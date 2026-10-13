# Document Arrangement and Digitization OCR

An asynchronous OCR service for digitizing Vietnamese administrative and archival documents. The service takes scanned PDF files from S3-compatible object storage, locates the key regions of each document with an object detection model, reads those regions with a Vietnamese vision-language model, normalizes the text, and returns structured metadata to the calling system through a webhook. A second, independent pipeline processes archival record slips ("phiếu tin") and writes the results directly to a PostgreSQL database.

---

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [System Architecture](#system-architecture)
4. [Inputs](#inputs)
5. [Processing Pipeline](#processing-pipeline)
6. [Expected Outcomes](#expected-outcomes)
7. [Repository Structure](#repository-structure)
8. [Prerequisites](#prerequisites)
9. [Installation](#installation)
10. [Configuration: Replacing Placeholder Values](#configuration-replacing-placeholder-values)
11. [Running the Service](#running-the-service)
12. [Usage](#usage)
13. [Operations and Monitoring](#operations-and-monitoring)
14. [Known Limitations](#known-limitations)

---

## Overview

The system automates metadata extraction for documents being archived and digitized. For each document it identifies and extracts fields such as:

- Document number and notation (Số, ký hiệu)
- Issuing date and place (Ngày tháng)
- Issuing authority (Tác giả / Cơ quan ban hành)
- Document type and summary (Tên loại và trích yếu)
- Signer (Người ký)
- Original or copy status (Bản gốc / Bản sao), determined from the ink color of the signature
- Confidentiality level (Độ mật), page count, and the handwritten sheet number of the document within its dossier (Trang số)

The output fields are not fixed. They are driven by configuration templates stored in the business database (`sampleconfigdocuments`), so each customer can define its own field set, ordering and export rules without code changes.

Main characteristics:

- **Asynchronous processing.** The API accepts a job, returns immediately with HTTP 202, and delivers the result to a callback URL when processing completes.
- **Horizontal scaling.** CPU-bound, GPU-bound and network-bound stages run in separate Celery queues, so each stage can be scaled independently.
- **Template-driven routing.** The document flow (government, record adjustment, CV, dossier, record slip) is selected automatically from the template name.
- **Vietnamese text normalization.** Post-processing covers spelling correction, administrative abbreviations, capitalization rules for organization names, and correction of place names against the national administrative unit catalog.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| API | FastAPI 0.115, Uvicorn 0.34, Pydantic 2 | REST endpoints for job submission, status, and worker management |
| Task queue | Celery 5.5, Kombu 5.5 | Multi-stage asynchronous pipeline |
| Message broker | RabbitMQ 3.9 | Task distribution across queues |
| Result backend | PostgreSQL 14 | Celery task state and the `ocr_results` archive table |
| Business database | PostgreSQL | Configuration templates (`sampleconfigdocuments`) and record slips (`dataslips`) |
| Object storage | S3-compatible storage (MinIO or AWS S3), Boto3 | Source document retrieval |
| Layout detection | Detectron2 (Faster R-CNN R50-FPN), PyTorch 2.5, CUDA | Detection of document regions (number, date, organization, signer, etc.) |
| Text recognition | Vintern-1B-v3.5 (`5CD-AI/Vintern-1B-v3_5`) served by vLLM | Vietnamese vision-language model for reading cropped regions |
| Image processing | OpenCV, Pillow, pdf2image (Poppler), PyMuPDF | PDF rasterization, deskewing, cropping, ink color detection |
| Text normalization | Custom rules, openpyxl, national administrative unit catalog (2021) | Spelling, abbreviation, capitalization and place-name correction |
| Monitoring | Flower (optional) | Worker inspection and the worker management API |
| Deployment | Bash scripts, systemd units, Docker Compose | Process orchestration on a Linux GPU host |

---

## System Architecture

```
                         +----------------------+
  Client system  ----->  |  FastAPI  (:9090)    |  ---- 202 Accepted + request_id
                         +----------+-----------+
                                    |  Celery chain
                                    v
                         +----------------------+
                         |  RabbitMQ  (:5672)   |
                         +----------+-----------+
          +---------------+---------+-----------+-----------------+
          v               v                     v                 v
   prepare_queue    process_queue         webhook_queue       db_queue
   (CPU)            (GPU)                 (network)           (CPU)
   download PDF,    detect regions  --->  POST result to      archive result in
   rasterize,       Detectron2 :8003      callback_url        ocr_results
   deskew, route    read regions    --->
                    vLLM :8005

   Record slip pipeline:
   phieutin_prepare_queue  ->  phieutin_process_queue  ->  phieutin_db_queue
   list PDFs in a bucket       Detectron2 :8010 + vLLM      UPDATE dataslips
```

| Service | Default port | Started by |
|---|---|---|
| FastAPI application | 9090 | `start-service.sh` |
| Detectron2 detection service (main documents) | 8003 | `start-service.sh` |
| Detectron2 detection service (record slips) | 8010 | `start-service.sh` |
| vLLM server (Vintern-1B-v3.5) | 8005 | `service/vllm.service` |
| RabbitMQ (AMQP / management UI) | 5672 / 15672 | `docker-compose.yaml` |
| PostgreSQL (result backend) | 5432 | `docker-compose.yaml` |
| Flower (optional) | 5555 | Started manually |

---

## Inputs

### Document OCR job

`POST /api/submit-ocr`

```json
{
  "s3_path": "s3://bucket-name/path/to/document.pdf",
  "customer_id": 71,
  "config_template_id": 102,
  "callback_url": "https://client.example.com/api/ocr/callback/14252",
  "access_key": "S3_ACCESS_KEY",
  "secret_key": "S3_SECRET_KEY",
  "endpoint": "http://s3-endpoint:9000",
  "document_id": 14252
}
```

| Field | Description |
|---|---|
| `s3_path` | Full `s3://bucket/key` path of the source document. Only PDF files are accepted. |
| `customer_id` | Customer (company) identifier in the business database. |
| `config_template_id` | Configuration template that defines the output fields and selects the processing flow. |
| `callback_url` | URL that receives the result through an HTTP POST when processing finishes. |
| `access_key`, `secret_key`, `endpoint` | Credentials and endpoint of the S3-compatible storage holding the document. They are supplied per request and are not stored by the service. |
| `document_id` | Identifier of the document in the client system, returned unchanged in the webhook payload. |

### Record slip OCR job

`POST /api/submit-phieutin`

```json
{
  "s3_bucket": "bucket-name",
  "s3_prefix": "phieutin/",
  "companyid": 122,
  "access_key": "S3_ACCESS_KEY",
  "secret_key": "S3_SECRET_KEY",
  "endpoint": "http://s3-endpoint:9000"
}
```

Every PDF under `s3_bucket/s3_prefix` is listed and processed in parallel, one task chain per file.

### Supporting inputs

- **Configuration templates.** These are read from the `sampleconfigdocuments` table of the business database (`POSTGRES_*` variables). The template name determines the processing flow, and the template object determines the output fields.
- **Model weights.** The Detectron2 weights are in `detectron_weights/` and are tracked with Git LFS. The Vintern-1B-v3.5 weights are loaded by vLLM.
- **Normalization dictionaries.** These are in `app/spellcheck_data/`:
  - `spell_dict.xlsx`: spelling and capitalization corrections, two columns: incorrect, correct.
  - `dvhc_2021.json`: the Vietnamese administrative unit catalog (63 provinces, 694 districts, 10,579 communes).
  - `TinhHuyenXa2021.xlsx`: the source of `dvhc_2021.json`. The code does not read it.

---

## Processing Pipeline

### Stage 1: Preparation and routing (`prepare_queue`)

1. Download the document from S3 storage. A missing object returns a webhook with error code 404.
2. Rasterize the PDF into page images (100 DPI) and deskew the first page.
3. Read the PDF metadata (`/Creator`, `/CreationDate`) to fill the creator and creation date.
4. Load the configuration template and select the processing flow from its name:

| Template name contains | Flow |
|---|---|
| `chỉnh lý` | `chinhly`: record adjustment |
| `chính quyền` | `government`: government administrative documents (default) |
| `lý lịch` | `cv`: personal records |
| `hồ sơ` | `hoso`: dynamic, template-based dossiers |
| `phiếu tin` | `phieutin`: record slips |

### Stage 2: Detection, recognition and normalization (`process_queue`)

1. **Layout detection.** Page images are sent to the Detectron2 service, which returns bounding boxes for 15 classes: `NO`, `DATE`, `ORG`, `THELOAI`, `VV`, `SENDTO`, `SIGNER`, `CHUCVU`, `RECEIVE`, `SAOY`, `PHULUC`, `TABLE`, `IMAGE`, `TEXT`, `BLANK`. The confidence threshold is 0.8.
2. **Signer search.** Header fields are taken from page 1. For documents longer than 10 pages, the signature block is searched in parallel from the beginning and the end of the document.
3. **Original or copy classification.** The ink color of the signature crop is analyzed in HSV space. Red or blue ink indicates an original, black ink indicates a copy.
4. **Text recognition.** Each region crop is sent concurrently to the vLLM server with a prompt specific to its label.
5. **Normalization.**
   - Spelling correction from `spell_dict.xlsx`.
   - Place-name correction after administrative unit keywords (`tỉnh`, `huyện`, `xã`, `phường`, etc.), matched against `dvhc_2021.json`. This corrects OCR errors in tone marks, vowel diacritics and `d`/`đ`. Ambiguous names are resolved with the province and district found in the same text.
   - Abbreviation normalization: `Ủy ban nhân dân` becomes `UBND`, `Hội đồng nhân dân` becomes `HĐND`, `về việc` becomes `v/v`.
   - Capitalization rules for issuing authorities and document types.
   - Separation of the document type from the summary.
6. **Field mapping.** The extracted values are mapped to the template fields (for example `SKH`, `NT`, `TG`, `TLVTY`, `NKVB`), with per-field confidence scores and date and number formatting. For example, `SKH` keeps both the number and the notation (`Số466-QĐ/TU` becomes `466-QĐ/TU`), while the older `SVB` and `KH` codes receive them separately.

### Stage 3: Result delivery (`webhook_queue`)

The structured result is POSTed to `callback_url` (up to 3 attempts), and the temporary working folder is removed.

### Stage 4: Archiving (`db_queue`)

The request, raw text and final JSON are upserted into the `ocr_results` table of the result backend database.

### Record slip pipeline

`phieutin_prepare_task` lists the PDFs in the bucket. `phieutin_process_single_task` downloads and rasterizes each file. A dedicated Detectron2 model on port 8010 detects 8 classes (`key`, `hopso`, `hososo`, `tieudehoso`, `thoigiantailieu`, `soto`, `thoigianbaoquan`, `quanlybienban`), the regions are read by vLLM, and `save_phieutin_to_db_task` updates the matching row in `public.dataslips`. The row is matched by `name = key`.

### Feature switches

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_SPELL_FIX` | `true` | Enables dictionary spelling correction and place-name correction. |
| `ENABLE_CAPITALIZATION_RULES` | `true` | Enables capitalization rules. When disabled, spelling correction keeps the original letter case of the text. |

---

## Expected Outcomes

### Immediate API response

```json
{
  "error_code": 202,
  "error_message": "Đã nhận yêu cầu xử lý OCR. Kết quả sẽ được gửi đến callback URL khi hoàn thành.",
  "request_id": "3f1c2a9e-0d6b-4c55-9a31-7e2b8f04c1d2"
}
```

### Webhook payload (sent to `callback_url`)

```json
{
  "meta": { "error_code": 200, "error_message": "Xử lý OCR thành công!" },
  "data": {
    "path": "s3://bucket-name/path/to/document.pdf",
    "raw_text": "Bùi Tiến Dũng Thái Bình, ngày 15 tháng 10 năm 2007 Số 466-QĐ/TU ...",
    "type_document": 102,
    "creator": "ScanPro",
    "created_at": "21:18:47 04/10/2026",
    "json": {
      "document_type": "Quyết định",
      "template_id": 102,
      "customer_id": 71,
      "pages_number": 1,
      "fields": [
        { "Name": "Số, ký hiệu", "Code": "SKH", "Value": "466-QĐ/TU", "Order": 1, "Percent": 99.81, "Page": 1 },
        { "Name": "Ngày tháng", "Code": "NT", "Value": "15/10/2007", "Order": 2, "Percent": 99.82, "Page": 1 },
        { "Name": "Tác giả", "Code": "TG", "Value": "Tỉnh Uỷ Thái Bình", "Order": 4, "Percent": 99.63, "Page": 1 },
        { "Name": "Người ký", "Code": "NKVB", "Value": "Bùi Tiến Dũng", "Order": 10, "Percent": 99.99, "Page": 1 }
      ],
      "confidence_score": 0.9353
    },
    "verified": false,
    "document_id": 14252
  },
  "request_id": "3f1c2a9e-0d6b-4c55-9a31-7e2b8f04c1d2"
}
```

Each entry in `fields` follows the configuration template and also contains `Children`, `Required`, `Type`, `Note` and `IsExportExcel`. These are omitted above for brevity. Field values depend on the template and on the normalization settings.

### Error responses

The same payload structure is used for failures, with `json` left empty:

| `error_code` | Condition |
|---|---|
| 404 | The source file does not exist in the storage bucket. |
| 500 | Download, conversion or processing failed after all retries. |

### Persisted results

| Table | Database | Content |
|---|---|---|
| `ocr_results` | Result backend (`CELERY_RESULT_BACKEND`) | One row per request: input, raw text, final JSON, verification state. Created automatically. |
| `public.dataslips` | Business database (`POSTGRES_*`) | Record slip fields. `status = 3` on success, `status = 4` on failure. Rows with `status = 5` (already verified) are never overwritten. |

The job state can also be queried at any time with `GET /api/request-status/{request_id}`.

---

## Repository Structure

```
.
|-- app/
|   |-- main_celery.py              FastAPI application and endpoints
|   |-- celery_app.py               Celery application, queues and routing
|   |-- tasks_new.py                Celery tasks (prepare, process, webhook, db, phieutin)
|   |-- flows_new.py                Processing flows (government, chinhly, cv, hoso)
|   |-- flow_phieutin.py            Record slip flow
|   |-- detectron2_api.py           Detection service for documents (port 8003)
|   |-- detectron2_phieutin_api.py  Detection service for record slips (port 8010)
|   |-- vision_service.py           vLLM client
|   |-- get_config_schema.py        Template loading and field mapping
|   |-- utils.py                    S3, PDF, image utilities and text normalization
|   |-- detect_color.py             Signature ink color detection
|   |-- worker_manager.py           Worker start/stop/restart through Flower
|   |-- models.py                   Pydantic request and response models
|   |-- config.py                   Static configuration
|   `-- spellcheck_data/            Spelling dictionary and administrative unit catalog
|-- detectron_weights/              Detectron2 weights (Git LFS)
|-- docs/                           Technical reports and change notes
|-- service/                        systemd unit files for the application and vLLM
|-- logs/                           Runtime logs
|-- temp_processing/                Temporary working folders (one per request)
|-- docker-compose.yaml             RabbitMQ and PostgreSQL
|-- start-service.sh                Starts the API, detection services and all workers
|-- stop-service.sh                 Stops the API, detection services and workers
|-- requirements.txt                Python dependencies
`-- .env                            Environment configuration (placeholder values)
```

---

## Prerequisites

- Linux host with an NVIDIA GPU, a CUDA-compatible driver, and enough GPU memory for both Detectron2 and vLLM. vLLM is configured to use 80% of GPU memory by default.
- Python 3.10, managed with Conda (Miniconda or Anaconda).
- Poppler utilities (`poppler-utils`), required by `pdf2image`.
- Docker and Docker Compose, for RabbitMQ and PostgreSQL.
- Git LFS, to download the Detectron2 weights.
- Network access to the S3-compatible storage, the business PostgreSQL database, and the client callback endpoints.

---

## Installation

1. Clone the repository and download the model weights:

   ```bash
   git lfs install
   git clone https://github.com/Khiem0203/Document-arrangement-and-digitization-OCR.git
   cd Document-arrangement-and-digitization-OCR
   git lfs pull
   ```

   Each file in `detectron_weights/` should be about 315 MB. If a file is only a few hundred bytes, it is an LFS pointer and `git lfs pull` has not completed.

2. Create the Python environment:

   ```bash
   conda create -n ocr python=3.10 -y
   conda activate ocr
   pip install -r requirements.txt
   ```

   Detectron2 is installed from source, so a CUDA toolkit matching the installed PyTorch build (CUDA 11.8 or 12.x) must be available.

3. Install vLLM in a separate virtual environment, which `service/vllm.service` expects:

   ```bash
   python3 -m venv ~/.venv
   ~/.venv/bin/pip install vllm
   ```

4. Start RabbitMQ and PostgreSQL:

   ```bash
   docker compose up -d
   ```

5. Replace all placeholder values as described in the next section.

---

## Configuration: Replacing Placeholder Values

All credentials, hosts and paths in this repository are placeholders. Before running the service, replace them in the files below. Placeholders follow three patterns:

- `your_...` for credentials and names
- `your-...` for hosts
- `/path/to/...` for file system paths

The following command lists every remaining placeholder:

```bash
grep -rn -E "your_|your-|/path/to" --include="*.py" --include="*.sh" --include="*.service" --include="*.yaml" --include=".env" .
```

### 1. `.env` (required)

This is the primary configuration file. `start-service.sh` loads it from `PROJECT_DIR`.

| Variable | Placeholder | Replace with |
|---|---|---|
| `CELERY_BROKER_URL` | `amqp://your_rabbitmq_user:your_rabbitmq_password@localhost:5672//` | RabbitMQ user, password and host |
| `CELERY_RESULT_BACKEND` | `db+postgresql://your_celery_db_user:your_celery_db_password@localhost:5432/celery_results` | Result backend database credentials and host. URL-encode special characters in the password, for example `@` becomes `%40`. |
| `POSTGRES_HOST` | `localhost` | Host of the business database (templates and `dataslips`) |
| `POSTGRES_DB` | `your_ocr_db` | Business database name |
| `POSTGRES_USER` / `POSTGRES_PW` | `your_db_user` / `your_db_password` | Business database credentials |
| `S3_ENDPOINT` | `http://localhost:9000/` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `your_s3_access_key` / `your_s3_secret_key` | Storage credentials |
| `S3_REGION` / `S3_SSL` | `us-east-1` / `False` | Storage region and TLS setting |
| `LLM_MODEL` | `/path/to/models--5CD-AI--Vintern-1B-v3_5` | Model name sent to vLLM. It must exactly match the name the vLLM server serves (see [Known Limitations](#known-limitations)). |
| `DETECTRON_WEIGHTS` | `/path/to/OCR_CELERY/detectron_weights/main_model/model_final.pth` | Absolute path to the main Detectron2 weights |
| `DETECTRON_PHIEUTIN` | `/path/to/OCR_CELERY/detectron_weights/phieutin_model/model_final.pth` | Absolute path to the record slip weights |
| `DETECTRON2_API_URL` | `http://localhost:8003/detectron2/detect` | Change only if the detection service runs on another host |
| `DETECTRON2_PHIEUTIN_API` | `http://localhost:8010/detectron2_phieutin/detect` | Change only if the detection service runs on another host |
| `VISION_MODEL_SERVER` | `http://localhost:8005/v1/chat/completions` | vLLM chat completions endpoint. Change only if vLLM runs on another host or port. |
| `ENABLE_SPELL_FIX` / `ENABLE_CAPITALIZATION_RULES` | `true` | Feature switches, see [Feature switches](#feature-switches) |

Optional variables that the code reads but `.env` does not define:

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_API_URL` | Not set | Overrides `VISION_MODEL_SERVER` when both are set. If neither is set, `http://localhost:8005/v1/chat/completions` is used. |
| `FLOWER_URL` | `http://localhost:5555` | Flower endpoint used by the worker management API |
| `PROJECT_DIR` | `/path/to/OCR_CELERY` | Project root, used for the startup connection log |

### 2. `docker-compose.yaml`

| Line | Placeholder | Replace with |
|---|---|---|
| 11 | `RABBITMQ_DEFAULT_USER=your_rabbitmq_user` | Same user as in `CELERY_BROKER_URL` |
| 12 | `RABBITMQ_DEFAULT_PASS=your_rabbitmq_password` | Same password as in `CELERY_BROKER_URL` |
| 30 | `POSTGRES_USER=your_celery_db_user` | Same user as in `CELERY_RESULT_BACKEND` |
| 31 | `POSTGRES_PASSWORD=your_celery_db_password` | Same password as in `CELERY_RESULT_BACKEND` |

### 3. `start-service.sh`

| Line | Placeholder | Replace with |
|---|---|---|
| 8 | `CONDA_PATH="/path/to/miniconda3/condabin/conda"` | Path to the `conda` executable |
| 10 | `PROJECT_DIR="/path/to/OCR_CELERY"` | Absolute path of this repository on the server |

Worker counts per queue (`DEFAULT_*_WORKERS`) and concurrency (`CPU_CONCURRENCY`, `GPU_CONCURRENCY`) are set in the same file.

### 4. `service/start-ocr.service` and `service/vllm.service`

| File | Placeholder | Replace with |
|---|---|---|
| Both | `User=your_user`, `Group=your_group` | Linux user and group that run the services |
| `start-ocr.service` | `/path/to/OCR_CELERY` (`WorkingDirectory`, `ExecStart`) | Absolute path of this repository |
| `start-ocr.service` | `/path/to/anaconda3/envs/ocr/bin` (`PATH`) | `bin` directory of the `ocr` Conda environment |
| `vllm.service` | `/path/to` (`WorkingDirectory`), `/path/to/.venv/bin` (`PATH`, `ExecStart`) | Home directory and the vLLM virtual environment |

### 5. Hard-coded values in application code

These values are not read from `.env` and must be edited directly in the source.

| File | Line | Placeholder | Replace with |
|---|---|---|---|
| `app/main_celery.py` | 205-209 | `host="localhost"`, `database="celery_results"`, `user="your_celery_db_user"`, `password="your_celery_db_password"` | Result backend credentials, used by `GET /api/suggestions` |
| `app/worker_manager.py` | 52 | `CELERY_BIN = "/path/to/anaconda3/envs/ocr/bin/celery"` | Path to the `celery` executable of the `ocr` environment |
| `app/celery_app.py` | 13 | `"/path/to/OCR_CELERY"` (default of `PROJECT_DIR`) | Repository path, or set `PROJECT_DIR` in the environment |
| `app/detectron2_api.py` | 30 | Default of `DETECTRON_WEIGHTS` | Used only when the variable is not set |
| `app/detectron2_phieutin_api.py` | 29 | Default of `DETECTRON_PHIEUTIN` | Used only when the variable is not set |
| `app/tasks_new.py` | 563-567, 599-603, 649-653 | Defaults of `POSTGRES_*` | Used only when the variables are not set |
| `app/config.py` | 9 | `http://localhost:15672/...` | RabbitMQ management URL, if RabbitMQ is on another host |

### 6. Documentation files

`docs/README.md` and `docs/phieutin.md` contain placeholder hosts (`your-git-server`, `your-s3-endpoint`, `your-bucket-name`). They do not affect runtime behavior.

Do not commit real credentials. After you fill in `.env`, stop git from tracking local changes to it:

```bash
git update-index --skip-worktree .env
```

---

## Running the Service

### Manual start

```bash
# 1. Infrastructure
docker compose up -d

# 2. vision-language model server
~/.venv/bin/vllm serve "5CD-AI/Vintern-1B-v3_5" \
  --trust-remote-code --max-model-len=2048 \
  --gpu-memory-utilization 0.8 --port 8005

# 3. API, detection services and all Celery workers
bash start-service.sh
```

`start-service.sh` loads `.env` and waits for PostgreSQL to become available. It then starts the FastAPI application (port 9090), both Detectron2 services (ports 8003 and 8010), and the workers for all seven queues. Logs are written to `logs/` and grouped by component.

### Running as systemd services

```bash
sudo cp service/vllm.service service/start-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm start-ocr
```

### Stopping

```bash
bash stop-service.sh
```

This stops the Celery workers, the API and the detection services. RabbitMQ and PostgreSQL keep running. Stop them with `docker compose down`.

All Celery workers load the application code once at startup. After any change to the code or to `.env`, restart the workers.

---

## Usage

Submit a document:

```bash
curl -X POST http://localhost:9090/api/submit-ocr \
  -H "Content-Type: application/json" \
  -d '{
        "s3_path": "s3://bucket-name/path/to/document.pdf",
        "customer_id": 71,
        "config_template_id": 102,
        "callback_url": "https://client.example.com/api/ocr/callback/1",
        "access_key": "S3_ACCESS_KEY",
        "secret_key": "S3_SECRET_KEY",
        "endpoint": "http://s3-endpoint:9000",
        "document_id": 1
      }'
```

Check the job status:

```bash
curl http://localhost:9090/api/request-status/<request_id>
```

Submit a batch of record slips:

```bash
curl -X POST http://localhost:9090/api/submit-phieutin \
  -H "Content-Type: application/json" \
  -d '{"s3_bucket": "bucket-name", "s3_prefix": "phieutin/", "companyid": 122,
       "access_key": "S3_ACCESS_KEY", "secret_key": "S3_SECRET_KEY",
       "endpoint": "http://s3-endpoint:9000"}'
```

### API reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/submit-ocr` | Submit a document OCR job |
| POST | `/api/submit-phieutin` | Submit a record slip batch |
| GET | `/api/request-status/{request_id}` | Celery state and result of a job |
| POST | `/api/crop-ocr` | Synchronous OCR of a single base64 image |
| GET | `/api/suggestions` | Previously extracted values for a field (for example `ORG`), for autocompletion |
| POST | `/api/create_profile_name` | Generate a dossier title from file information |
| GET | `/api/queue/process-queue/status` | Length of the processing queue |
| GET | `/api/workers/status` | Worker status (requires Flower) |
| POST | `/api/workers/start`, `/stop`, `/restart`, `/start-preset` | Worker lifecycle management (requires Flower) |
| GET | `/api/workers/presets` | Available worker presets |
| GET | `/health` | Liveness check |

Interactive API documentation is available at `http://localhost:9090/docs`.

---

## Operations and Monitoring

- **Logs.** Logs are written to `logs/` with subfolders per component (`api`, `process`, `chinhly`, `phieutin`, `db`). The infrastructure connection check performed at worker startup is recorded in `logs/startup_connections.log`.
- **Queues.** The RabbitMQ management UI is available at `http://<host>:15672`.
- **Workers.** Flower provides worker inspection and is required for the `/api/workers/*` endpoints:

  ```bash
  celery -A app.celery_app flower --port=5555
  ```

- **Temporary files.** `temp_processing/<request_id>/` is removed after the webhook is sent. Folders of failed jobs remain and can be cleaned up periodically.
- **Reliability.** Workers use late acknowledgement (`task_acks_late`) and a prefetch of 1. A task interrupted by a worker crash is returned to the queue.

---

## Known Limitations

- **Model name must match.** The value of `LLM_MODEL` is sent as the `model` field of every vLLM request. It must be identical to the name the vLLM server serves. With the provided `vllm.service`, which serves `5CD-AI/Vintern-1B-v3_5`, set `LLM_MODEL="5CD-AI/Vintern-1B-v3_5"`, or start vLLM from the local snapshot path used in `LLM_MODEL`.
- **PDF only.** Only PDF input is accepted. Other content types are rejected during preparation.
- **Spelling correction only touches words in the dictionary.** Place-name correction only applies after an administrative unit keyword, and only to names of two or more syllables. Names absent from the 2021 catalog, such as units created by later mergers, are not corrected.
- **`confidence_score` is not computed from the model.** The document-level `confidence_score` in the webhook payload is a randomly generated placeholder. Per-field `Percent` values come from the detection model.
- **GPU sharing.** Detectron2 and vLLM share the GPU. Increasing `GPU_CONCURRENCY` or the number of `process_queue` workers requires enough GPU memory.
