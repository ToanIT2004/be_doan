# Backend chấm công khuôn mặt

Backend FastAPI quản lý nhân viên, ảnh khuôn mặt, đơn nghỉ phép và chấm công. YOLO phát hiện khuôn mặt; MobileNetV3 PAD phân loại `live`/`spoof`/`uncertain`; chỉ khuôn mặt `live` mới được ArcFace so khớp với nhân viên đã đăng ký và ghi công. Dữ liệu được lưu trong PostgreSQL qua SQLAlchemy và Alembic.

Đọc [`AI_DOCUMENTATION.md`](AI_DOCUMENTATION.md) để hiểu kiến trúc AI, công thức, dữ liệu, cách train, metric và kịch bản thuyết trình.

Đọc [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) để hiểu kiến trúc frontend, backend, database, lưu trữ, các luồng nghiệp vụ và cách triển khai.

Bắt đầu từ thư mục gốc project; sau lệnh `cd be`, mọi lệnh còn lại chạy trong `be`. Đường dẫn model và upload trong cấu hình được tính tương đối từ thư mục này.

## Chuẩn bị

- Python, PostgreSQL và một database mà tài khoản ứng dụng có quyền tạo bảng.
- Checkpoint YOLO **đã huấn luyện để phát hiện khuôn mặt** tại `models/yolov8n-face.pt`. Model YOLO COCO thông thường không thay thế được checkpoint này.
- Model InsightFace `buffalo_l` tại `models/insightface/models/buffalo_l/` khi dùng cấu hình mặc định.

Workspace hiện có các file model trên. Nếu chạy ở máy khác, hãy chuẩn bị chúng trước khi gọi API phát hiện/nhận diện. Model chỉ được nạp khi có yêu cầu inference đầu tiên.

### Cài đặt Python

PowerShell trên Windows:

```powershell
cd be
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
cd be
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Nếu đã có virtual environment với đầy đủ dependency, chỉ cần kích hoạt môi trường đó.

### Database và biến môi trường

Tạo database, ví dụ bằng `psql` (đổi user/database cho phù hợp):

```powershell
psql -U postgres -c "CREATE DATABASE doan;"
```

Tạo file `be/.env`. Ví dụ tối thiểu, **không dùng nguyên mật khẩu hoặc khóa mẫu khi triển khai**:

```dotenv
DATABASE_URL=postgresql+psycopg2://DB_USER:DB_PASSWORD@localhost:5432/doan
JWT_SECRET_KEY=replace-with-a-long-random-secret
ACCESS_TOKEN_EXPIRE_MINUTES=60
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
APP_TIMEZONE=Asia/Ho_Chi_Minh
```

`DATABASE_URL` được `app.database` và Alembic cùng sử dụng; giá trị trong `.env` được ưu tiên hơn `sqlalchemy.url` ở `alembic.ini`. Backend đọc `.env` khi chạy từ thư mục `be`.

| Biến tùy chọn | Mặc định trong source | Công dụng |
| --- | --- | --- |
| `YOLO_FACE_MODEL_PATH` | `models/yolov8n-face.pt` | Checkpoint YOLO face. |
| `YOLO_FACE_CONFIDENCE` | `0.5` | Ngưỡng confidence của detector. |
| `YOLO_FACE_IMAGE_SIZE` | `640` | Kích thước inference YOLO. |
| `YOLO_PREVIEW_IMAGE_SIZE` | `416` | Kích thước inference nhẹ hơn cho WebSocket preview camera. |
| `YOLO_DEVICE` | `cpu` | Thiết bị inference, ví dụ `cpu` hoặc `0`. |
| `DETECTION_MAX_FRAME_BYTES` | `8388608` | Giới hạn byte cho một frame detection/recognition. |
| `DETECTION_MAX_PIXELS` | `16777216` | Giới hạn pixel của ảnh sau decode. |
| `ARCFACE_MODEL_NAME` | `buffalo_l` | Tên bộ model InsightFace. |
| `ARCFACE_MODEL_ROOT` | `models/insightface` | Thư mục gốc model InsightFace. |
| `ARCFACE_PROVIDERS` | `CPUExecutionProvider` | ONNX Runtime providers, ngăn cách bằng dấu phẩy. |
| `ARCFACE_CTX_ID` | `-1` | Context ID khi chuẩn bị model. |
| `ARCFACE_DET_SIZE` | `640` | Kích thước detector ArcFace. |
| `FACE_RECOGNITION_THRESHOLD` | `0.55` | Ngưỡng similarity để nhận diện. |
| `ANTI_SPOOF_MODEL_PATH` | `models/antispoofing/mobilenetv3_pad.onnx` | MobileNetV3 ONNX đã train cho PAD. |
| `ANTI_SPOOF_INPUT_SIZE` | `160` | Kích thước ảnh vào của model PAD; phải giống lúc train. |
| `ANTI_SPOOF_LIVE_THRESHOLD` | `0.75` | Từ ngưỡng này trở lên mới xét là `live`. |
| `ANTI_SPOOF_SPOOF_THRESHOLD` | `0.30` | Từ ngưỡng này trở xuống xét là `spoof`. |
| `ANTI_SPOOF_MAX_UNCERTAINTY` | `0.18` | Độ lệch điểm giữa các frame tối đa để chấp nhận `live`. |
| `ANTI_SPOOF_MIN_FRAMES` | `3` | Số frame hợp lệ tối thiểu của endpoint đa frame. |
| `ANTI_SPOOF_MAX_FRAMES` | `16` | Số frame tối đa của endpoint đa frame. |
| `APP_TIMEZONE` | `Asia/Ho_Chi_Minh` | Múi giờ xác định ngày/thời điểm chấm công. |
| `FILE_STORAGE_ROOT` | `app/uploads` | Thư mục lưu file upload. |
| `FILE_MAX_BYTES` | `10485760` | Giới hạn byte cho một file upload. |
| `CORS_ORIGINS` | `*` | Origin frontend, ngăn cách bằng dấu phẩy. |

`JWT_SECRET_KEY` có giá trị dự phòng trong source chỉ để phát triển; hãy đặt khóa riêng trong `.env`. Nếu đổi `FILE_STORAGE_ROOT`, phải cập nhật cả mount file tĩnh `/uploads` trong `app/main.py`; hiện mount này trỏ đến `app/uploads`.

### Migration và chạy server

Chạy các migration **đã có** trong repository:

```powershell
alembic upgrade head
alembic current
New-Item -ItemType Directory -Force app/uploads | Out-Null
python -m uvicorn app.main:app --reload
```

Trên macOS/Linux, dùng `mkdir -p app/uploads` thay cho lệnh `New-Item`. Không chạy `alembic revision --autogenerate` khi chỉ cài đặt: lệnh đó tạo migration mới và chỉ dùng khi chủ động thay đổi model/schema. API mặc định ở `http://127.0.0.1:8000`; Swagger UI ở `http://127.0.0.1:8000/docs`, OpenAPI JSON ở `/openapi.json`. `GET /` trả thông báo server đang chạy.

Repository chưa có bước seed admin. Để gọi các API `/admin/...` trên database mới, cần tạo một nhân viên có `role='admin'`. Tạo bcrypt hash bằng lệnh sau (nhập mật khẩu tại prompt, không đặt mật khẩu trực tiếp trong lệnh):

```powershell
python -c "from getpass import getpass; from app.auth.security import hash_password; print(hash_password(getpass('Admin password: ')))"
```

Sau đó dùng `psql -U postgres -d doan` để thêm bản ghi, thay `<bcrypt-hash>` bằng giá trị vừa tạo:

```sql
INSERT INTO employees (username, password, fullname, phone, role)
VALUES ('admin', '<bcrypt-hash>', 'Administrator', '0000000000', 'admin');
```

Chỉ cần làm bước này một lần. Sau khi có admin, có thể tạo nhân viên khác qua `POST /api/v1/admin/employee`; API tự hash mật khẩu.

## Xác thực và quyền truy cập

`POST /api/v1/auth/login` nhận JSON `{"username":"...","password":"..."}` và trả `accessToken`, `tokenType`, `employee`. Các API yêu cầu đăng nhập nhận header `Authorization: Bearer <accessToken>`. Token mặc định hết hạn sau 60 phút; source hiện không cung cấp API refresh token.

- **Admin**: quản lý nhân viên, ảnh mặt, enrollment, duyệt đơn và tra cứu chấm công theo nhân viên.
- **Nhân viên đã đăng nhập**: xem chấm công/đơn nghỉ của mình, tạo đơn nghỉ và gọi API nhận diện/chấm công.
- **Công khai theo source hiện tại**: API detection, upload/đọc/xóa file và các health endpoint; các endpoint này chưa có lớp xác thực riêng.

`GET /api/v1/admin/employee` loại các tài khoản có `role='admin'` khỏi **danh sách**. Những API quản lý theo ID vẫn tra cứu theo ID bình thường.

## Danh sách API

Có thể đặt biến Postman `{{doan}} = http://127.0.0.1:8000/api/v1`. Chi tiết request/response và mã lỗi từng route có ở `/docs`.

| Nhóm | Endpoint | Quyền | Ghi chú |
| --- | --- | --- | --- |
| Xác thực | `POST /auth/login` | Công khai | Đăng nhập username/password. |
| Xác thực | `GET /auth/me` | Bearer | Tài khoản hiện tại. |
| Nhân viên | `GET /admin/employee` | Admin | Danh sách không gồm role admin; hỗ trợ `skip`, `limit`. |
| Nhân viên | `GET /admin/employee/{employee_id}` | Admin | Chi tiết nhân viên. |
| Nhân viên | `POST /admin/employee` | Admin | Tạo nhân viên. |
| Nhân viên | `PUT /admin/employee/{employee_id}` | Admin | Cập nhật nhân viên. |
| Nhân viên | `DELETE /admin/employee/{employee_id}` | Admin | Xóa nhân viên. |
| Ảnh mặt | `GET /admin/employee/face-images/{employeeId}` | Admin | Ảnh của nhân viên. |
| Ảnh mặt | `POST /admin/employee/face-images` | Admin | JSON gồm `employeeId`, `imagePath`. |
| Ảnh mặt | `PUT /admin/employee/face-images/{id}` | Admin | Cập nhật ảnh. |
| Ảnh mặt | `DELETE /admin/employee/face-images/{id}` | Admin | Xóa bản ghi ảnh. |
| File | `POST /file/uploads/upload` | Công khai | Multipart: `image`, `type` (`face`/`avatar`). |
| File | `GET /file/uploads/{type}/{file_name}` | Công khai | Đọc file; URL tĩnh cũng có dạng `/uploads/{type}/{file_name}`. |
| File | `DELETE /file/uploads/delete/{type}/{file_name}` | Công khai | Xóa file. |
| Detection | `GET /detection/health` | Công khai | Trạng thái checkpoint YOLO. |
| Detection | `POST /detection/faces` | Công khai | Multipart `image`; trả box và confidence. |
| Detection | `WS /detection/ws/faces` | Công khai | Phát hiện theo frame. |
| Recognition | `GET /recognition/health` | Công khai | Trạng thái ArcFace và threshold. |
| Anti-spoof | `GET /recognition/anti-spoof/health` | Công khai | Trạng thái model MobileNetV3 PAD. |
| Recognition | `POST /recognition/admin/employee/{employee_id}/enroll` | Admin | Tạo/làm mới embedding từ ảnh đã đăng ký. |
| Recognition | `POST /recognition/attendance` | Bearer | PAD 3–16 frame có trọng số chất lượng, sau đó ArcFace và ghi công. |
| Chấm công | `GET /employee/attendance` | Bearer | Lịch sử của tài khoản đang đăng nhập; không truyền `employeeId`. |
| Chấm công | `GET /admin/attendance/{employeeId}` | Admin | Lịch sử theo nhân viên. |
| Nghỉ phép | `POST /employee/leave-request` | Bearer | Tạo đơn nghỉ. |
| Nghỉ phép | `PUT /employee/leave-request` | Bearer | Sửa đơn `pending` của mình; body cần `id`. |
| Nghỉ phép | `DELETE /employee/leave-request/{leave_request_id}` | Bearer | Xóa đơn `pending` của mình. |
| Nghỉ phép | `GET /employee/leave-request/my` | Bearer | Danh sách đơn của mình. |
| Nghỉ phép | `POST /admin/leave-request/browse` | Admin | Duyệt/từ chối; body gồm `id`, `status`, và `rejectReason` khi từ chối. |
| Nghỉ phép | `GET /admin/leave-request` | Admin | Tất cả đơn. |

`LeaveRequestCreate.leaveType` chấp nhận `annual_leave`, `sick_leave`, `unpaid_leave`, `personal_leave` hoặc `other`. Khi từ chối đơn, `rejectReason` là bắt buộc.

## Đăng ký mặt và chấm công

1. Admin tạo nhân viên qua `POST /admin/employee`.
2. Upload ảnh mặt qua `POST /file/uploads/upload` với `type=face`. Response `fileName` là URL dạng `/uploads/face/<tên-file>`; dùng giá trị này làm `imagePath` ở bước tiếp.
3. Admin tạo bản ghi ảnh qua `POST /admin/employee/face-images` với `employeeId`, `imagePath`. Source hiện giới hạn mỗi nhân viên một bản ghi ảnh mặt.
4. Admin gọi `POST /recognition/admin/employee/{employee_id}/enroll` để sinh embedding. Ảnh enrollment cần đúng một khuôn mặt.
5. Client có thể gọi detection REST/WebSocket để xác định mặt, rồi gửi 3–16 frame đến `POST /recognition/attendance` bằng Bearer token. Backend chạy MobileNetV3 PAD trước; chỉ trạng thái `live` mới tiếp tục qua ArcFace và ghi công.

Khi nhận diện thành công, response `faces[0].identity.status` là `recognized` và `faces[0].attendance` chứa bản ghi vừa ghi. Mặt không khớp trả `status=unknown`, `attendance=null`. API nhận diện dựa vào mặt trong frame, **không yêu cầu người được nhận diện trùng với tài khoản dùng token**; đây là luồng cho terminal dùng chung.

Bảng `attendances` có một dòng cho mỗi nhân viên mỗi ngày theo `APP_TIMEZONE`: lần nhận diện đầu tạo `checkIn`, các lần tiếp theo trong cùng ngày cập nhật `checkOut` trên dòng đó. Hai API GET chấm công trả `id`, `employeeId`, `attendanceDate`, `checkIn`, `checkOut`, theo ngày mới nhất trước.

### Ví dụ gọi bằng PowerShell

```powershell
$base = "http://127.0.0.1:8000/api/v1"
$login = Invoke-RestMethod -Method Post -Uri "$base/auth/login" `
  -ContentType "application/json" `
  -Body '{"username":"<username>","password":"<password>"}'

Invoke-RestMethod -Method Get -Uri "$base/employee/attendance" `
  -Headers @{ Authorization = "Bearer $($login.accessToken)" }
```

Thay `<username>`, `<password>` và đường dẫn ảnh trước khi chạy. Ảnh detection/recognition hỗ trợ JPEG, PNG và WebP.

### WebSocket detection

Kết nối `ws://127.0.0.1:8000/api/v1/detection/ws/faces`, sau đó gửi một trong hai dạng:

- Binary: bytes ảnh JPEG/PNG/WebP; server đánh số `frameId` tăng dần.
- Text JSON: `{"frameId":"123","image":"<base64 hoặc data URL>"}`.

Mỗi frame nhận một JSON response chứa `frameId`, kích thước `image`, danh sách `faces`, `count`, `inferenceMs`; lỗi trả object `error`. Nên chờ phản hồi trước khi gửi frame tiếp theo. Tọa độ `box` là pixel của ảnh gửi lên, nên client phải scale khi vẽ lên preview kích thước khác.

## Xử lý sự cố

| Hiện tượng | Kiểm tra |
| --- | --- |
| `401` khi login | Username/mật khẩu trong bảng `employees`; API không phân biệt role ở bước xác thực. |
| `401` ở API bảo vệ | Có Bearer token, token chưa hết hạn và nhân viên còn tồn tại. |
| `403` ở API `/admin/...` | Tài khoản đăng nhập phải có `role='admin'`. |
| Kết nối DB/migration lỗi | PostgreSQL đang chạy, `DATABASE_URL` đúng, đã `alembic upgrade head`; chạy lệnh từ `be`. |
| Detection/recognition trả `503` | Kiểm tra dependency, file model và các biến YOLO/ArcFace. Xem `/detection/health`, `/recognition/health`. |
| Recognition trả `400` | Frame phải là ảnh hợp lệ, đúng một khuôn mặt; ảnh enrollment cũng phải có một mặt. |
| File upload được nhưng URL `/uploads/...` không đọc được | Kiểm tra `FILE_STORAGE_ROOT` và mount file tĩnh trong `app/main.py`. |

Không đưa `.env`, token hoặc ảnh dữ liệu thật vào commit. Khi thay đổi model database, tạo migration bằng `alembic revision --autogenerate -m "mô tả thay đổi"`, kiểm tra file migration, rồi chạy `alembic upgrade head`.

## Huấn luyện và sử dụng MobileNetV3 chống giả mạo

Luồng bảo mật là `YOLO -> MobileNetV3 PAD -> ArcFace -> chấm công`. MobileNetV3 trả một trong ba trạng thái `live`, `spoof`, `uncertain`. ArcFace và thao tác ghi công chỉ chạy khi trạng thái là `live`. Hệ thống cố ý trả `503` nếu chưa có model PAD thay vì bỏ qua bước bảo mật.

### 1. Chuẩn bị môi trường train

Nên train trên máy có GPU. Tạo một môi trường riêng nếu bản PyTorch cần CUDA khác với backend:

```powershell
cd be
python -m pip install -r requirements.txt
```

Nếu dùng NVIDIA GPU, cài bản PyTorch tương ứng CUDA theo hướng dẫn chính thức của PyTorch trước khi chạy script. Kiểm tra:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Chuẩn bị dữ liệu đúng cách

Dataset hiện tại đã được chuẩn bị theo cấu trúc có thư mục người ở giữa:

```text
data/
  train/<subject_id>/live/*.jpg
  train/<subject_id>/spoof/*.jpg
  validate/<subject_id>/live/*.jpg
  validate/<subject_id>/spoof/*.jpg
  test/<subject_id>/live/*.jpg
  test/<subject_id>/spoof/*.jpg
```

Script train đọc trực tiếp cấu trúc này. Nhãn được lấy từ thư mục `live` hoặc
`spoof`, không lấy từ `<subject_id>`. Tên split validation `val` và `validate`
đều được hỗ trợ.

Chỉ dùng bước dưới đây khi đầu vào vẫn còn là video/ảnh gốc chưa crop:

Sắp xếp video/ảnh gốc theo cấu trúc sau. Phải chia theo **người và video nguồn** trước khi trích frame; tuyệt đối không để các frame của cùng video rơi vào nhiều split:

```text
data/antispoof_raw/
  train/live/<person-or-video-folders>/...
  train/spoof/<person-or-video-folders>/...
  val/live/...
  val/spoof/...
  test/live/...
  test/spoof/...
```

`live` là camera quay người thật. `spoof` gồm ảnh in và ảnh/video phát lại trên màn hình. Nên có nhiều camera, độ sáng, khoảng cách và thiết bị trình chiếu. Chỉ thu dữ liệu khuôn mặt khi có sự đồng ý của người tham gia.

Trích mỗi 5 frame và crop mặt bằng đúng YOLO/padding mà API sử dụng:

```powershell
python -m training.prepare_antispoof_data `
  --input data/antispoof_raw `
  --output data/antispoof `
  --sample-every 5
```

Kết quả có cấu trúc `data/antispoof/{train,val,test}/{live,spoof}`. Script bỏ qua frame không có đúng một khuôn mặt.

### 3. Train mới hoàn toàn và export ONNX

Xóa hai file `mobilenetv3_pad.pt` và `mobilenetv3_pad.onnx` cũ trước khi chạy
nếu muốn backend không thể vô tình dùng model cũ. Script không resume checkpoint
anti-spoof: mỗi lần chạy đều tạo MobileNetV3-Small mới, nạp trọng số ImageNet làm
điểm khởi đầu rồi thay classifier thành hai lớp.

```powershell
python -m training.train_antispoof `
  --data data `
  --output models/antispoofing `
  --epochs 25 `
  --batch-size 32 `
  --image-size 160
```

Script fine-tune MobileNetV3-Small, chọn checkpoint có ACER validation thấp nhất, báo cáo Accuracy/APCER/BPCER/ACER trên test (nếu có), rồi tạo:

```text
models/antispoofing/mobilenetv3_pad.pt
models/antispoofing/mobilenetv3_pad.onnx
models/antispoofing/training_metrics.json
```

Thứ tự output luôn là `[spoof, live]`. File ONNX có batch động để backend suy luận nhiều frame trong một lần. Không đánh giá chất lượng đề tài chỉ bằng accuracy; ưu tiên APCER, BPCER, ACER và kiểm thử camera chưa xuất hiện trong train.

Đoạn nạp backbone vào quá trình train nằm trong
`training/train_antispoof.py`, hàm `build_model`:

```python
weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
model = mobilenet_v3_small(weights=weights)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 2)
```

`DEFAULT` là trọng số ImageNet của torchvision, không phải checkpoint
anti-spoof cũ. Muốn train ngẫu nhiên hoàn toàn, thêm `--no-pretrained` (thường
không nên dùng với dataset nhỏ).

#### Kết quả đánh giá

Kết quả của model cũ không còn đại diện cho dataset hiện tại. Sau khi train xong,
xem dòng `test acc=... APCER=... BPCER=... ACER=...` trên terminal hoặc mở
`models/antispoofing/training_metrics.json`. File JSON lưu metric tốt nhất trên
validation, metric test và lịch sử của từng epoch. Sau đó cần hiệu chỉnh lại các
ngưỡng live/spoof trước khi dùng thực tế.

### 4. Cấu hình và chạy API

Thêm vào `.env` nếu muốn đổi mặc định:

```dotenv
ANTI_SPOOF_MODEL_PATH=models/antispoofing/mobilenetv3_pad.onnx
ANTI_SPOOF_INPUT_SIZE=160
ANTI_SPOOF_LIVE_THRESHOLD=0.75
ANTI_SPOOF_SPOOF_THRESHOLD=0.30
ANTI_SPOOF_MAX_UNCERTAINTY=0.18
ANTI_SPOOF_MIN_FRAMES=3
ANTI_SPOOF_MAX_FRAMES=16
```

`ANTI_SPOOF_INPUT_SIZE` phải trùng `--image-size`. Sau đó khởi động backend và kiểm tra:

```powershell
python -m uvicorn app.main:app --reload
Invoke-RestMethod http://127.0.0.1:8000/api/v1/recognition/anti-spoof/health
```

`modelExists=true` nghĩa là đường dẫn đúng. `modelLoaded` chuyển thành `true` sau lần inference đầu tiên.

### 5. Gọi endpoint khuyến nghị

Gửi 3–16 frame bằng cùng field multipart `frames`. Ví dụ sau khi đã có `$login.accessToken`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/recognition/attendance" `
  -H "Authorization: Bearer $($login.accessToken)" `
  -F "frames=@C:\frames\01.jpg;type=image/jpeg" `
  -F "frames=@C:\frames\02.jpg;type=image/jpeg" `
  -F "frames=@C:\frames\03.jpg;type=image/jpeg"
```

Ý nghĩa kết quả:

- `liveness.status=live`: backend mới chạy ArcFace; nếu nhận diện được thì ghi công.
- `liveness.status=spoof`: từ chối ảnh in/video phát lại; `identity.status=not_evaluated`.
- `liveness.status=uncertain`: dữ liệu nằm giữa hai ngưỡng hoặc các frame không ổn định; yêu cầu quét lại.

Luồng chấm công chỉ dùng endpoint đa frame `/recognition/attendance`; endpoint nhận diện một frame đã được loại bỏ để tránh một đường xử lý PAD kém ổn định.

### 6. Hiệu chỉnh ngưỡng

Các giá trị `0.30`, `0.75`, `0.18` chỉ là mặc định an toàn để bắt đầu tích hợp, không phải ngưỡng đã được chứng minh cho dữ liệu của nhóm. Dùng validation set để chọn ngưỡng theo APCER/BPCER mong muốn, khóa ngưỡng rồi mới báo cáo test set. Không dùng test set để chọn ngưỡng.
