"""Fine-tune MobileNetV3-Small for RGB face presentation-attack detection.

Supported data layouts (split people/videos before training):

    data/{train,validate,test}/<subject_id>/{live,spoof}/*.jpg

The legacy layout ``<split>/{live,spoof}/*.jpg`` and the split name ``val``
are also accepted. Labels are read from the live/spoof directory at any depth,
not from the first directory below a split.

The exported ONNX model always uses class order [spoof, live].
"""

import argparse
import copy
import json
import os
import random
from pathlib import Path

# Keep downloaded torchvision weights inside this project. This avoids
# permission errors on restricted Windows accounts; .torch/ is git-ignored.
os.environ.setdefault("TORCH_HOME", str(Path(".torch").resolve()))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_TO_INDEX = {"spoof": 0, "live": 1}


class AntiSpoofDataset(Dataset):
    """Read labels from a live/spoof path component below a split root."""

    def __init__(self, root: Path, transform) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_parts = {
                part.lower()
                for part in path.relative_to(root).parent.parts
                if part.lower() in CLASS_TO_INDEX
            }
            if len(label_parts) != 1:
                raise ValueError(
                    f"Image must be inside exactly one live/spoof directory: {path}"
                )
            label = CLASS_TO_INDEX[label_parts.pop()]
            self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No supported images found in dataset split: {root}")
        self.targets = [label for _, label in self.samples]
        if set(self.targets) != {0, 1}:
            raise ValueError(f"Dataset split must contain both live and spoof images: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
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


def resolve_split_root(data_root: Path, split: str) -> Path | None:
    names = ("val", "validate") if split == "val" else (split,)
    matches = [data_root / name for name in names if (data_root / name).is_dir()]
    if len(matches) > 1:
        raise ValueError(f"Use only one validation split directory: {matches}")
    return matches[0] if matches else None


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
        split_root = resolve_split_root(args.data, split)
        if split_root is None:
            if split == "test":
                continue
            expected = "val or validate" if split == "val" else split
            raise FileNotFoundError(
                f"Missing dataset split '{expected}' below: {args.data}"
            )
        dataset = AntiSpoofDataset(
            split_root, train_transform if split == "train" else eval_transform
        )
        sampler = None
        if split == "train":
            class_counts = np.bincount(dataset.targets, minlength=2)
            if np.any(class_counts == 0):
                raise ValueError("Training split must contain both spoof and live images")
            sample_weights = [1.0 / class_counts[target] for target in dataset.targets]
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
    # This is where the base model is pulled into training. DEFAULT loads
    # ImageNet weights; it never loads the previous anti-spoof checkpoint.
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
    best_val_metrics = None
    history = []
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
        train_loss = running_loss / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": metrics,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
            f"APCER={metrics['apcer']:.4f} BPCER={metrics['bpcer']:.4f} "
            f"ACER={metrics['acer']:.4f}"
        )
        if metrics["acer"] < best_acer:
            best_acer = metrics["acer"]
            best_val_metrics = metrics.copy()
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
    test_metrics = None
    if "test" in loaders:
        test_metrics = evaluate(model, loaders["test"], criterion, device)
        print(
            "test "
            f"acc={test_metrics['accuracy']:.4f} "
            f"APCER={test_metrics['apcer']:.4f} "
            f"BPCER={test_metrics['bpcer']:.4f} "
            f"ACER={test_metrics['acer']:.4f}"
        )

    onnx_path = args.output / "mobilenetv3_pad.onnx"
    export_onnx(model, onnx_path, args.image_size)
    metrics_path = args.output / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "device": str(device),
                "epochs": args.epochs,
                "image_size": args.image_size,
                "class_names": ["spoof", "live"],
                "best_validation": best_val_metrics,
                "test": test_metrics,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_val_ACER={best_acer:.4f}")
    print(f"exported: {onnx_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
