# Phiếu Tin OCR — Tài liệu thay đổi

## Tổng quan

Phiếu tin là một pipeline OCR **hoàn toàn tách biệt** so với pipeline văn bản hành chính hiện có.  
Thay vì gửi kết quả qua webhook, kết quả được UPDATE trực tiếp vào bảng `dataslips` trên remote DB.

---

## Input / Output

### Đầu vào (`POST /api/submit-phieutin`)
```json
{
  "s3_bucket": "your-bucket-name",
  "s3_prefix": "phieutin/",
  "companyid": 122,
  "access_key": "...",
  "secret_key": "...",
  "endpoint": "http://your-s3-endpoint:9000"
}
```

> `s3_bucket` = tên bucket, `s3_prefix` = thư mục con trong bucket (kết thúc bằng `/`).  
> Pipeline sẽ **list toàn bộ PDF** trong `s3_bucket/s3_prefix` và xử lý song song từng file.

### Đầu ra (response ngay lập tức)
```json
{
  "error_code": 202,
  "error_message": "Đã nhận yêu cầu xử lý phiếu tin. Các file sẽ được xử lý song song và cập nhật vào DB khi hoàn thành.",
  "request_id": "celery-task-id"
}
```

### Kết quả xử lý (ghi vào DB)
- `status = 3` → OCR thành công, các trường đã được UPDATE
- `status = 4` → OCR thất bại
- Lookup bằng `WHERE name = <key>` — `key` là giá trị OCR từ class `key` của Detectron2

---

## Chain xử lý

```
POST /api/submit-phieutin
        ↓
phieutin_prepare_task           [phieutin_prepare_queue]
  - List toàn bộ PDF trong s3_bucket/s3_prefix
  - Spawn 1 chain per file
        ↓ (mỗi file)
phieutin_process_single_task    [phieutin_prepare_queue]
  - Download PDF từ S3
  - Convert PDF → ảnh
  - Deskew trang 1
        ↓
dispatch_and_process_task       [process_queue]  ← dùng chung với flow khác
  - Gọi phieutin_flow()
  - Detectron2 port 8010 detect 9 class
  - Binarize crop (Otsu) trước khi OCR
  - vLLM OCR song song (asyncio.gather)
  - Strip label prefix
        ↓
save_phieutin_to_db_task        [phieutin_db_queue]
  - UPDATE bảng dataslips WHERE name = <key>
  - status=3 (thành công) hoặc status=4 (thất bại)
  - Cleanup temp folder
```

---

## Detectron2 Classes (port 8010)

8 class, thứ tự **alphabetical** (phải khớp với thứ tự training JSON, model đã retrain bỏ class `tinhtrangvatly`):

| Class | Trường DB | Prefix strip |
|---|---|---|
| `hopso` | `hopso` | `b. Hộp số:` |
| `hososo` | `hososo` | `c. Hồ sơ số:` |
| `key` | `name` (dùng để lookup DB) | `Key:` |
| `quanlybienban` | `quanlybienban` | `9. Quản lý biên bản:` |
| `soto` | `soto` | `6. Số lượng tờ:` |
| `thoigianbaoquan` | `thoigianbaoquan` | `7. Thời hạn bảo quản:` |
| `thoigiantailieu` | `thoigiantailieu` | `5. Thời gian của tài liệu:` |
| `tieudehoso` | `tieudehoso` | `4. Tiêu đề hồ sơ:` |

> Không còn class cha parse JSON (bỏ `soluutru`, `thoigian`, `tenkho`, `tenphong`, `tenhoso`, `soluong`, `thoihan`, `tinhtrang`).

---

## Tiền xử lý ảnh

- **Toàn ảnh**: deskew trước khi OCR
- **Crop từng class**: binarize (Otsu threshold, grayscale → trắng đen) trước khi gửi vLLM

---

## Các file đã thay đổi

### 1. `app/models.py`
- `PhieutinRequest`: đổi từ `s3_path` + `dataslip_id` → `s3_bucket` + `s3_prefix`
- `PhieutinSubmitResponse`: giữ nguyên

---

### 2. `app/flows_new.py`
- `PROMPT_MAPPING_PHIEUTIN`: 9 entry, tên class trực tiếp (không còn tenkho, tenphong...)
- `_PHIEUTIN_PREFIX_MAP`: 9 entry tương ứng
- `_run_detection_and_ocr_phieutin`: trả dict 9 trường, không parse JSON sub-field
- Crop binarization (Otsu) thêm vào sau detection, trước vLLM
- `phieutin_flow`: `dataslips_data` map 9 trường + `name` = `key`

---

### 3. `app/tasks_new.py`
**`phieutin_prepare_task`** (sửa):
- Không còn download 1 file — thay bằng list toàn bộ PDF trong bucket/prefix
- Spawn 1 `celery_chain` per file: `phieutin_process_single_task → dispatch_and_process_task → save_phieutin_to_db_task`

**`phieutin_process_single_task`** (mới):
- Queue: `phieutin_prepare_queue`
- Download 1 PDF từ S3, convert, deskew
- Trả context cho bước tiếp theo

**`save_phieutin_to_db_task`** (sửa):
- `WHERE name = key` thay vì `WHERE id = dataslip_id`
- `status=3` (thành công), `status=4` (thất bại)
- UPDATE 7 cột: `hopso`, `hososo`, `tieudehoso`, `thoigiantailieu`, `soto`, `thoigianbaoquan`, `quanlybienban`

---

### 4. `app/main_celery.py`
- Import thêm `phieutin_process_single_task`
- Endpoint `POST /api/submit-phieutin`: truyền `s3_bucket`, `s3_prefix` thay vì `s3_path`, `dataslip_id`

---

### 5. `app/detectron2_phieutin_api.py`
- `thing_classes` cập nhật 8 class alphabetical (model đã retrain bỏ `tinhtrangvatly`):
  ```python
  ["hopso", "hososo", "key", "quanlybienban", "soto", "thoigianbaoquan", "thoigiantailieu", "tieudehoso"]
  ```

---

## Lưu ý triển khai

1. **Detectron2 port 8010 phải được start trước** khi phieutin workers hoạt động
2. **Biến môi trường remote DB** cần được set trong `.env`:
   - `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PW`
3. **Nếu `phieutin_prepare_task` hoặc `phieutin_process_single_task` fail**, chain dừng và `status=4` KHÔNG được set cho file đó — cần bổ sung error handler sau
4. **Workers phieutin** cần start riêng: `-Q phieutin_prepare_queue` và `-Q phieutin_db_queue`
5. **`thing_classes` trong Detectron2 API phải khớp chính xác thứ tự với file training JSON** — sai thứ tự → detect nhầm class
