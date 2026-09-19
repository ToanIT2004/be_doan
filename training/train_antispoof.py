"""Fine-tune MobileNetV3-Small for RGB face presentation-attack detection.

Expected data layout (split people/videos before extracting frames):

    data/antispoof/
      train/live/*.jpg
      train/spoof/*.jpg
      val/live/*.jpg
      val/spoof/*.jpg
      test/live/*.jpg
      test/spoof/*.jpg

The exported ONNX model always uses class order [spoof, live].
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class RemapTarget:
    def __init__(self, original_classes: dict[str, int]) -> None:
        required = {"live", "spoof"}
        if set(original_classes) != required:
            raise ValueError(
                f"Each split must contain exactly {sorted(required)}; "
                f"found {sorted(original_classes)}"
            )
        self.mapping = {
            original_classes["spoof"]: 0,
            original_classes["live"]: 1,
        }

    def __call__(self, target: int) -> int:
        return self.mapping[target]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/antispoof"))
    parser.add_argument("--output", type=Path, default=Path("models/antispoofing"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(root: Path, transform) -> datasets.ImageFolder:
    probe = datasets.ImageFolder(root)
    return datasets.ImageFolder(
        root,
        transform=transform,
        target_transform=RemapTarget(probe.class_to_idx),
    )


def make_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.30,
                contrast=0.30,
                saturation=0.20,
                hue=0.04,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.25,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    loaders = {}
    for split in ("train", "val", "test"):
        split_root = args.data / split
        if not split_root.is_dir():
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing dataset split: {split_root}")
        dataset = make_dataset(
            split_root,
            train_transform if split == "train" else eval_transform,
        )
        sampler = None
        if split == "train":
            remapped_targets = [dataset.target_transform(item) for item in dataset.targets]
            class_counts = np.bincount(remapped_targets, minlength=2)
            if np.any(class_counts == 0):
                raise ValueError("Training split must contain both spoof and live images")
            sample_weights = [1.0 / class_counts[target] for target in remapped_targets]
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print(
                f"train class counts: spoof={class_counts[0]} "
                f"live={class_counts[1]} (balanced sampler enabled)"
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"{split}: {len(dataset)} images")
    return loaders


def build_model(pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    count = 0
    attack_total = attack_accepted = 0
    live_total = live_rejected = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_total += float(criterion(logits, labels)) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum())
        count += labels.size(0)
        attack_mask = labels == 0
        live_mask = labels == 1
        attack_total += int(attack_mask.sum())
        live_total += int(live_mask.sum())
        attack_accepted += int(((predictions == 1) & attack_mask).sum())
        live_rejected += int(((predictions == 0) & live_mask).sum())

    apcer = attack_accepted / max(attack_total, 1)
    bpcer = live_rejected / max(live_total, 1)
    return {
        "loss": loss_total / max(count, 1),
        "accuracy": correct / max(count, 1),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
    }


def export_onnx(model: nn.Module, path: Path, image_size: int) -> None:
    model = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    loaders = make_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = build_model(not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_acer = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        metrics = evaluate(model, loaders["val"], criterion, device)
        print(
            f"epoch={epoch:03d} train_loss={running_loss / max(seen, 1):.4f} "
            f"val_loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
            f"APCER={metrics['apcer']:.4f} BPCER={metrics['bpcer']:.4f} "
            f"ACER={metrics['acer']:.4f}"
        )
        if metrics["acer"] < best_acer:
            best_acer = metrics["acer"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "image_size": args.image_size,
                    "class_names": ["spoof", "live"],
                    "val_metrics": metrics,
                },
                args.output / "mobilenetv3_pad.pt",
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    if "test" in loaders:
        metrics = evaluate(model, loaders["test"], criterion, device)
        print(
            "test "
            f"acc={metrics['accuracy']:.4f} APCER={metrics['apcer']:.4f} "
            f"BPCER={metrics['bpcer']:.4f} ACER={metrics['acer']:.4f}"
        )

    onnx_path = args.output / "mobilenetv3_pad.onnx"
    export_onnx(model, onnx_path, args.image_size)
    print(f"best_val_ACER={best_acer:.4f}")
    print(f"exported: {onnx_path}")


if __name__ == "__main__":
    main()
