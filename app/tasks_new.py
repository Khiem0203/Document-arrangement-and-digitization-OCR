import logging
import os
import shutil
import torch
import gc
from celery import chain
from celery.utils.log import get_task_logger
import requests
from .celery_app import celery_app
from botocore.exceptions import ClientError
from .utils import download_from_s3, convertPDFImage, send_webhook_sync, get_template_name, get_pdf_metadata, deskew, process_text, variants, insert_newline_after_title
from .models import OCRWebhookResponse, OCRMetaResponse, OCRDataResponse
from .flows_new import government_flow, curriculum_vitae_flow, chinhly_flow, _ocr_cropped_image_processing, hoso_flow
from .flow_phieutin import phieutin_flow
import cv2
from dotenv import load_dotenv
import psycopg2
import json
from datetime import datetime
import time
from urllib.parse import urlparse, unquote
load_dotenv()


celery_logger = get_task_logger(__name__)


def _celery_db_connect(connect_timeout: int = 5):
    """Parse CELERY_RESULT_BACKEND và trả về psycopg2 connection.
    Đảm bảo mọi thay đổi host/port chỉ cần sửa 1 chỗ trong .env.
    """
    url = os.getenv("CELERY_RESULT_BACKEND", "")
    # Strip SQLAlchemy prefix: db+postgresql:// -> postgresql://
    url = url.replace("db+postgresql://", "postgresql://", 1)
    # Strip query string params không hợp lệ với psycopg2
    url = url.split("?")[0]
    p = urlparse(url)
    password = unquote(p.password) if p.password else None
    return psycopg2.connect(
        host=p.hostname,
        port=p.port or 5432,
        database=p.path.lstrip("/"),
        user=p.username,
        password=password,
        connect_timeout=connect_timeout,
    )


# ==============================================================================
#  TASK 1: Routing and Preparation
# ==============================================================================
@celery_app.task(name="routing_and_preparation_task", bind=True, max_retries=0)
def routing_and_preparation_task(self, s3_path: str, customer_id: int, config_template_id: int, access_key: str, secret_key: str, endpoint: str, callback_url: str, document_id: int):
    """
    Downloads file, converts to images, determines the processing flow,
    and passes context to the next task.
    """
    celery_logger.info(f"Routing & Prep Task ({self.request.id}): Starting for {s3_path}")
    temp_folder = f'./temp_processing/{self.request.id}'

    try:
        file_path, content_type = download_from_s3(s3_path, temp_folder, access_key, secret_key, endpoint)
        if content_type != 'application/pdf':
            raise ValueError(f"Unsupported file type: {content_type}")
        else:
            page_image_folder = os.path.join(temp_folder, "pages")
            image_paths = convertPDFImage(file_path, page_image_folder)
            if image_paths is None or len(image_paths) == 0:
                raise ValueError("PDF conversion resulted in no images.")
            else:
                img_input = cv2.imread(image_paths[0])
                result = deskew(img_input)
                cv2.imwrite(image_paths[0], result)

            template_name = get_template_name(customer_id, config_template_id, celery_logger)
            creator = get_pdf_metadata(file_path).get('/Creator', 'Unknown')
            created_at_raw = get_pdf_metadata(file_path).get('/CreationDate', '').replace("D:", "").replace("'", "")
            created_at_raw = created_at_raw[:14] if len(created_at_raw) >= 14 else ''

            from datetime import datetime
            # If PDF has valid CreationDate metadata, use it; otherwise use current time
            if created_at_raw and len(created_at_raw) == 14:
                try:
                    created_at = datetime.strptime(created_at_raw, "%Y%m%d%H%M%S")
                except ValueError:
                    celery_logger.warning(f"Routing & Prep Task ({self.request.id}): Invalid CreationDate format '{created_at_raw}', using current time")
                    created_at = datetime.now()
            else:
                celery_logger.info(f"Routing & Prep Task ({self.request.id}): No CreationDate metadata found, using current time")
                created_at = datetime.now()

            formatted_created_at = created_at.strftime("%H:%M:%S %d/%m/%Y")
            flow_type = 'government' # Default flow

            # Check for more specific keywords first
            if 'chỉnh lý' in template_name.lower():
                flow_type = 'chinhly'
            elif 'chính quyền' in template_name.lower():
                flow_type = 'government'
            elif 'lý lịch' in template_name.lower():
                flow_type = 'cv'
            elif 'hồ sơ' in template_name.lower():
                flow_type = 'hoso'
            elif 'phiếu tin' in template_name.lower():
                flow_type = 'phieutin'


            celery_logger.info(f"Routing & Prep Task ({self.request.id}): Determined flow_type='{flow_type}'")
            celery_logger.info(f"Creator metadata extracted: {creator}")
            celery_logger.info(f"Created_at metadata extracted: {get_pdf_metadata(file_path).get('/CreationDate', '')}")

        return {
            "flow_type": flow_type,
            "image_paths": image_paths,
            "temp_folder": temp_folder,
            "s3_path": s3_path,
            "customer_id": customer_id,
            "config_template_id": config_template_id,
            "request_id": self.request.id,
            "creator": creator,
            "created_at": formatted_created_at,
            "callback_url": callback_url,  # Add callback_url to context
            "document_id": document_id  # Add document_id to context
        }
    # MODIFIED: Added specific handling for S3 ClientError
    except ClientError as e:
        # MODIFIED: The correct error code for a missing S3 key is 'NoSuchKey'
        if e.response['Error']['Code'] == 'NoSuchKey':
            # This is a permanent "File Not Found" error. Do not retry.
            celery_logger.error(f"Routing & Prep Task ({self.request.id}) failed: File not found on S3 at {s3_path}.")
            error_response = OCRWebhookResponse(
                meta=OCRMetaResponse(error_code=404, error_message=f"File not found on S3: {s3_path}"),
                data=OCRDataResponse(path=s3_path, raw_text="", type_document=config_template_id, creator="", created_at="", json={}, document_id=document_id),
                request_id=self.request.id
            ).dict()
            celery_logger.info(f"the current response is {error_response}")
            send_webhook_sync(callback_url, error_response)
            # After sending the webhook, we do not raise an exception, allowing the task to end gracefully.
        else:
            # Another S3-related error occurred (e.g., credentials, permissions). Treat as 500 and retry.
            celery_logger.error(f"Routing & Prep Task ({self.request.id}) failed with S3 error: {e}", exc_info=True)
            if self.request.retries >= self.max_retries:
                error_response = OCRWebhookResponse(
                    meta=OCRMetaResponse(error_code=500, error_message=f"Processing failed due to S3 error: {str(e)}"),
                    data=OCRDataResponse(path=s3_path, raw_text="", type_document=config_template_id, creator="", created_at="", json={}, document_id=document_id),
                    request_id=self.request.id
                ).dict()
                send_webhook_sync(callback_url, error_response)
            raise self.retry(exc=e, countdown=10)

    except Exception as e:
        # Handles all other errors (e.g., PDF conversion, network issues)
        celery_logger.error(f"Routing & Prep Task ({self.request.id}) failed: {e}", exc_info=True)
        if self.request.retries >= self.max_retries:
            celery_logger.error(f"Task {self.request.id} failed permanently. Sending error webhook.")
            error_response = OCRWebhookResponse(
                meta=OCRMetaResponse(error_code=500, error_message=f"Processing failed in preparation stage: {str(e)}"),
                data=OCRDataResponse(path=s3_path, raw_text="", type_document=config_template_id, creator="", created_at="", json={}, document_id=document_id),
                request_id=self.request.id
            ).dict()
            send_webhook_sync(callback_url, error_response)
        raise self.retry(exc=e, countdown=10)

    finally:
        # Cleanup is now in a finally block to ensure it runs even after a 404
        if 'temp_folder' in locals() and os.path.exists(temp_folder):
            pass
            # shutil.rmtree(temp_folder)


# ==============================================================================
#  TASK 2: Dispatcher (The new central processing task)
# ==============================================================================
@celery_app.task(name="dispatch_and_process_task", bind=True, max_retries=0)
def dispatch_and_process_task(self, context: dict):
    """
    Receives context from the prep task and dispatches to the correct flow logic.
    """
    flow_type = context.get("flow_type")
    # author = context.get("author", "Unknown")
    celery_logger.info(f"Dispatcher Task ({self.request.id}): Received context for flow: {flow_type}")

    try:
        # --- UPDATED DISPATCHING LOGIC ---
        if flow_type == 'chinhly':
            result = chinhly_flow(context, celery_logger)
        elif flow_type == 'government':
            result = government_flow(context, celery_logger)
        elif flow_type == 'cv':
            result = curriculum_vitae_flow(context, celery_logger)
        elif flow_type == 'hoso':
            result = hoso_flow(context, celery_logger)
        elif flow_type == 'phieutin':
            # Phieutin xử lý lỗi riêng — không raise, trả error context để chain tiếp tục
            if context.get("phieutin_error"):
                # Bước trước đã lỗi, bỏ qua OCR, truyền thẳng context xuống save task
                celery_logger.warning(f"Dispatcher Task ({self.request.id}): upstream phieutin error, skipping OCR.")
                return context
            try:
                result = phieutin_flow(context, celery_logger)
                context.update(result)
                celery_logger.info(f"The current result is {result}")
                return context
            except Exception as phieutin_exc:
                celery_logger.error(f"Dispatcher Task ({self.request.id}): phieutin_flow failed: {phieutin_exc}", exc_info=True)
                context["phieutin_error"]     = True
                context["phieutin_error_msg"] = str(phieutin_exc)
                return context

        context.update(result)
        celery_logger.info(f"The current result is {result}")
        return context

    except Exception as e:
        celery_logger.error(f"Dispatcher Task ({self.request.id}) failed during flow '{flow_type}': {e}", exc_info=True)

        # Send error webhook if all retries are exhausted (chỉ áp dụng cho flow không phải phieutin)
        callback_url = context.get("callback_url")
        if self.request.retries >= self.max_retries:
            celery_logger.error(f"Task {self.request.id} failed permanently. Sending error webhook.")
            if callback_url:
                error_response = OCRWebhookResponse(
                    meta=OCRMetaResponse(error_code=500, error_message=f"Lỗi xử lý ở luồng '{flow_type}': {str(e)}"),
                    data=OCRDataResponse(
                        path=context.get("s3_path", "Unknown"),
                        raw_text="",
                        type_document=context.get("config_template_id", 0),
                        creator=context.get("creator", ""),
                        created_at=context.get("created_at", ""),
                        json={},
                        document_id=context.get("document_id")
                    ),
                    request_id=context.get("request_id")
                ).dict()
                send_webhook_sync(callback_url, error_response)
                celery_logger.info(f"webhook error successfully submitted")

        raise self.retry(exc=e, countdown=15)



# ==============================================================================
#  TASK 3: Final Webhook and Cleanup Task (Largely Unchanged)
# ==============================================================================
@celery_app.task(bind=True, name="send_webhook_task", max_retries=0)
def send_webhook_task(self, context: dict):
    """Receives final data, sends the webhook, and performs cleanup."""
    request_id = self.request.id
    temp_folder = context.get("temp_folder")
    callback_url = context.get("callback_url", "")
    flow_type = context.get("flow_type", "unknown")

    try:
        celery_logger.info(f"Webhook Task ({request_id}): Preparing to send result.")
        celery_logger.info(f"Webhook Task ({request_id}): Callback URL: {callback_url}")

        webhook_data = OCRWebhookResponse(
            meta=OCRMetaResponse(error_code=200, error_message="Xử lý OCR thành công!"),
            data=OCRDataResponse(
                path=context["s3_path"],
                raw_text=context.get("raw_text", ""),
                type_document=context["config_template_id"],
                creator=context.get("creator", "Unknown"),
                created_at=context.get("created_at", ""),
                json=context.get("final_json", {}),
                verified=False,  # Added: initial OCR results, not yet verified
                document_id=context.get("document_id")
            ),
            request_id=request_id
        ).dict()
        celery_logger.info(f"Webhook json sent: {webhook_data}")
        send_webhook_sync(callback_url, webhook_data)

        # Return the full context so the next task can access the OCR results
        return context

    except Exception as e:
        celery_logger.error(f"Webhook Task ({request_id}) failed: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=20)

    #COMMENT BLOCK THIS BLOCK IF NEEDED
    else:
        pass
    finally:
        if temp_folder and os.path.exists(temp_folder):
            try:
                shutil.rmtree(temp_folder)
                celery_logger.info(f"Webhook Task ({request_id}): Cleaned up temp folder {temp_folder}.")
            except Exception as cleanup_error:
                celery_logger.error(f"Webhook Task ({request_id}): Cleanup failed for {temp_folder}: {cleanup_error}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ==============================================================================
#  TASK 4: OCR Task
# ==============================================================================

@celery_app.task(name="ocr_process_task", bind=True, max_retries=0)
def ocr_process_task(self, image_base64: str):
    """
    Receives context from the prep task and dispatches to the correct flow logic.
    """
    try:
        ocr_result = _ocr_cropped_image_processing(image_base64, celery_logger)
        return ocr_result
    except Exception as e:
        celery_logger.error(f"OCR Task error: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=15)


# ==============================================================================
#  TASK 5: Save Results to Database
# ==============================================================================
@celery_app.task(bind=True, name="save_results_to_db_task", max_retries=3)
def save_results_to_db_task(self, context: dict):
    """
    Saves OCR processing results to PostgreSQL database after webhook is sent.
    """
    request_id = context.get("request_id", self.request.id)
    callback_url = context.get("callback_url", "")

    try:
        celery_logger.info(f"Save DB Task ({request_id}): Saving OCR results to local database.")
        celery_logger.info(f"Context keys available: {list(context.keys())}")
        celery_logger.info(f"Context data - s3_path: {context.get('s3_path')}, customer_id: {context.get('customer_id')}, config_template_id: {context.get('config_template_id')}, callback_url: {callback_url}")

        # Connect to local PostgreSQL database (same as Celery result backend)
        conn = _celery_db_connect()
        cursor = conn.cursor()

        # Create table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ocr_results (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(255) UNIQUE NOT NULL,
            s3_path TEXT,
            customer_id INTEGER,
            config_template_id INTEGER,
            callback_url TEXT,
            raw_text TEXT,
            final_json JSONB,
            verified_json JSONB,
            creator VARCHAR(255),
            created_at VARCHAR(255),
            flow_type VARCHAR(50),
            image_paths TEXT[],
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP,
            status VARCHAR(50) DEFAULT 'completed',
            document_id INTEGER
        );
        """
        cursor.execute(create_table_query)

        # Add new columns if they don't exist (for existing tables)
        alter_table_queries = [
            "ALTER TABLE ocr_results ADD COLUMN IF NOT EXISTS verified_json JSONB;",
            "ALTER TABLE ocr_results ADD COLUMN IF NOT EXISTS verified_at TIMESTAMP;",
            "ALTER TABLE ocr_results ADD COLUMN IF NOT EXISTS image_paths TEXT[];"
        ]

        for alter_query in alter_table_queries:
            try:
                cursor.execute(alter_query)
                celery_logger.info(f"Executed: {alter_query}")
            except Exception as e:
                celery_logger.warning(f"Could not execute {alter_query}: {e}")

        # Insert OCR results
        insert_query = """
        INSERT INTO ocr_results
        (request_id, s3_path, customer_id, config_template_id, callback_url,
         raw_text, final_json, creator, created_at, flow_type, image_paths, status, document_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (request_id)
        DO UPDATE SET
            raw_text = EXCLUDED.raw_text,
            final_json = EXCLUDED.final_json,
            image_paths = EXCLUDED.image_paths,
            processed_at = CURRENT_TIMESTAMP,
            status = EXCLUDED.status,
            document_id = EXCLUDED.document_id;
        """

        cursor.execute(insert_query, (
            request_id,
            context.get("s3_path"),
            context.get("customer_id"),
            context.get("config_template_id"),
            callback_url,
            context.get("raw_text", ""),
            json.dumps(context.get("final_json", {})),
            context.get("creator", "Unknown"),
            context.get("created_at", ""),
            context.get("flow_type", "unknown"),
            context.get("image_paths", []),
            "completed",
            context.get("document_id")
        ))

        conn.commit()
        cursor.close()
        conn.close()

        celery_logger.info(f"Save DB Task ({request_id}): Successfully saved results to database.")

        # Return the full context so the next task can access the OCR results
        return context

    except Exception as e:
        celery_logger.error(f"Save DB Task ({request_id}) failed: {e}", exc_info=True)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=30)
        else:
            celery_logger.error(f"Save DB Task ({request_id}): Failed permanently after {self.max_retries} retries.")
            # Still return context so the chain can continue
            return context


def _log_dataslip_status(cursor, key_value: str, request_id: str, context_label: str):
    """Query và log status hiện tại của record (3=thành công, 4=thất bại, 5=đã kiểm tra) để debug việc skip update."""
    try:
        cursor.execute("SELECT status FROM public.dataslips WHERE name = %s", (key_value,))
        row = cursor.fetchone()
        if row is None:
            celery_logger.info(f"Phieutin DB Task ({request_id}) [{context_label}]: no record found for name='{key_value}'")
        else:
            celery_logger.info(f"Phieutin DB Task ({request_id}) [{context_label}]: current status={row[0]} for name='{key_value}'")
    except Exception as e:
        celery_logger.warning(f"Phieutin DB Task ({request_id}) [{context_label}]: failed to check status for name='{key_value}': {e}")


# ==============================================================================
#  PHIEUTIN TASK 1: List all PDFs in bucket and spawn one chain per file
# ==============================================================================
@celery_app.task(name="phieutin_prepare_task", bind=True, max_retries=0)
def phieutin_prepare_task(self, s3_bucket: str, s3_prefix: str, companyid: int, access_key: str, secret_key: str, endpoint: str):
    """Phieutin step 1: List all PDFs in S3 bucket/prefix, spawn one chain per file."""
    celery_logger.info(f"Phieutin Prepare Task ({self.request.id}): Listing s3://{s3_bucket}/{s3_prefix}")

    try:
        import boto3
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("S3_REGION", "us-east-1"),
            use_ssl=os.getenv("S3_SSL", "False").lower() in ('true', '1', 't'),
        )

        paginator = s3_client.get_paginator('list_objects_v2')
        pdf_keys = []
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.pdf'):
                    pdf_keys.append(key)

        celery_logger.info(f"Phieutin Prepare Task ({self.request.id}): Found {len(pdf_keys)} PDF(s).")

        from celery import chain as celery_chain
        for key in pdf_keys:
            s3_path = f"s3://{s3_bucket}/{key}"
            celery_chain(
                phieutin_process_single_task.s(
                    s3_path=s3_path,
                    companyid=companyid,
                    access_key=access_key,
                    secret_key=secret_key,
                    endpoint=endpoint,
                ).set(queue='phieutin_prepare_queue'),
                dispatch_and_process_task.s().set(queue='phieutin_process_queue'),
                save_phieutin_to_db_task.s().set(queue='phieutin_db_queue'),
            ).apply_async()

        return {"dispatched": len(pdf_keys), "files": pdf_keys}

    except Exception as e:
        celery_logger.error(f"Phieutin Prepare Task ({self.request.id}) failed: {e}", exc_info=True)
        raise


# ==============================================================================
#  PHIEUTIN TASK 1b: Process single PDF file
# ==============================================================================
@celery_app.task(name="phieutin_process_single_task", bind=True, max_retries=0)
def phieutin_process_single_task(self, s3_path: str, companyid: int, access_key: str, secret_key: str, endpoint: str):
    """Download one PDF from S3, convert to images, deskew page 1."""
    celery_logger.info(f"Phieutin Single Task ({self.request.id}): Processing {s3_path}")
    temp_folder = f'./temp_processing/{self.request.id}'

    try:
        file_path, content_type = download_from_s3(s3_path, temp_folder, access_key, secret_key, endpoint)
        if content_type != 'application/pdf':
            raise ValueError(f"Unsupported file type: {content_type}")

        page_image_folder = os.path.join(temp_folder, "pages")
        image_paths = convertPDFImage(file_path, page_image_folder)
        if not image_paths:
            raise ValueError("PDF conversion resulted in no images.")

        img_input = cv2.imread(image_paths[0])
        deskewed = deskew(img_input)
        cv2.imwrite(image_paths[0], deskewed)

        celery_logger.info(f"Phieutin Single Task ({self.request.id}): Prepared {len(image_paths)} page(s).")
        return {
            "flow_type": "phieutin",
            "image_paths": image_paths,
            "temp_folder": temp_folder,
            "s3_path": s3_path,
            "companyid": companyid,
            "request_id": self.request.id,
        }

    except Exception as e:
        celery_logger.error(f"Phieutin Single Task ({self.request.id}) failed: {e}", exc_info=True)
        # Không raise — trả error context để chain tiếp tục chạy đến save_phieutin_to_db_task
        # Lúc này chưa có key (chưa OCR) nên DB không thể update status=4, chỉ log lỗi
        return {
            "phieutin_error":     True,
            "phieutin_error_msg": str(e),
            "flow_type":          "phieutin",
            "s3_path":            s3_path,
            "temp_folder":        temp_folder,
            "request_id":         self.request.id,
            "dataslips_data":     {},
        }

    finally:
        pass


# ==============================================================================
#  PHIEUTIN TASK 3: Save to Remote DB
# ==============================================================================
@celery_app.task(bind=True, name="save_phieutin_to_db_task", max_retries=3)
def save_phieutin_to_db_task(self, context: dict):
    """
    Phieutin step 3: UPDATE dataslips record on remote DB.
    Finds record by name = key (OCR result).
    Sets status=3 on success, status=4 on failure.
    """
    request_id = context.get("request_id", self.request.id)
    dataslips_data = context.get("dataslips_data", {})
    temp_folder = context.get("temp_folder")
    key_value = dataslips_data.get("key", "")

    # ==============================================================================
    #  Xử lý khi upstream task lỗi (phieutin_error=True)
    # ==============================================================================
    if context.get("phieutin_error"):
        error_msg = context.get("phieutin_error_msg", "Unknown error")
        s3_path   = context.get("s3_path", "")
        celery_logger.error(f"Phieutin DB Task ({request_id}): upstream error for '{s3_path}': {error_msg}")

        if key_value:
            # Có key → set status=4 trong DB
            try:
                conn = psycopg2.connect(
                    host=os.getenv("POSTGRES_HOST", "localhost"),
                    port=5432,
                    database=os.getenv("POSTGRES_DB", "your_ocr_db"),
                    user=os.getenv("POSTGRES_USER", "your_db_user"),
                    password=os.getenv("POSTGRES_PW", "your_db_password")
                )
                cur = conn.cursor()
                _log_dataslip_status(cur, key_value, request_id, "upstream_error_before_update")
                cur.execute(
                    "UPDATE public.dataslips SET status=4, updatedat=NOW() WHERE name=%s AND status <> 5",
                    (key_value,)
                )
                conn.commit()
                rows_affected = cur.rowcount
                cur.close()
                conn.close()
                if rows_affected:
                    celery_logger.info(f"Phieutin DB Task ({request_id}): status=4 set for name='{key_value}'")
                else:
                    celery_logger.info(f"Phieutin DB Task ({request_id}): skip update for name='{key_value}' — already verified (status=5)")
            except Exception as db_err:
                celery_logger.error(f"Phieutin DB Task ({request_id}): Failed to set status=4: {db_err}")
        else:
            # Không có key (lỗi ở bước download/convert) → không thể update DB
            celery_logger.warning(f"Phieutin DB Task ({request_id}): No key available, cannot update DB status=4 for '{s3_path}'")

        # Cleanup temp nếu có
        if temp_folder and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)
        return context
    # ==============================================================================

    celery_logger.info(f"Phieutin DB Task ({request_id}): Searching dataslips WHERE name = '{key_value}'")

    try:
        remote_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=5432,
            database=os.getenv("POSTGRES_DB", "your_ocr_db"),
            user=os.getenv("POSTGRES_USER", "your_db_user"),
            password=os.getenv("POSTGRES_PW", "your_db_password")
        )
        cursor = remote_conn.cursor()

        _log_dataslip_status(cursor, key_value, request_id, "success_before_update")

        cursor.execute("ALTER TABLE public.dataslips ADD COLUMN IF NOT EXISTS path TEXT;")

        # AND status <> 5: không ghi đè record đã được người dùng kiểm tra & nhập tay
        update_query = """
        UPDATE public.dataslips
        SET hopso = %s, hososo = %s,
            tieudehoso = %s, thoigiantailieu = %s,
            soto = %s, thoigianbaoquan = %s,
            quanlybienban = %s,
            path = %s,
            status = 3, updatedat = NOW()
        WHERE name = %s AND status <> 5;
        """

        cursor.execute(update_query, (
            dataslips_data.get("hopso", ""),
            dataslips_data.get("hososo", ""),
            dataslips_data.get("tieudehoso", ""),
            dataslips_data.get("thoigiantailieu", ""),
            dataslips_data.get("soto", ""),
            dataslips_data.get("thoigianbaoquan", ""),
            dataslips_data.get("quanlybienban", ""),
            dataslips_data.get("path", ""),
            key_value,
        ))

        remote_conn.commit()
        rows_affected = cursor.rowcount
        cursor.close()
        remote_conn.close()
        if rows_affected:
            celery_logger.info(f"Phieutin DB Task ({request_id}): status=3 set for name='{key_value}'")
        else:
            celery_logger.info(f"Phieutin DB Task ({request_id}): skip update for name='{key_value}' — already verified (status=5)")
        return context

    except Exception as e:
        celery_logger.error(f"Phieutin DB Task ({request_id}) failed: {e}", exc_info=True)
        try:
            fail_conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=5432,
                database=os.getenv("POSTGRES_DB", "your_ocr_db"),
                user=os.getenv("POSTGRES_USER", "your_db_user"),
                password=os.getenv("POSTGRES_PW", "your_db_password")
            )
            fail_cursor = fail_conn.cursor()
            _log_dataslip_status(fail_cursor, key_value, request_id, "exception_before_update")
            fail_cursor.execute(
                "UPDATE public.dataslips SET status = 4, updatedat = NOW() WHERE name = %s AND status <> 5",
                (key_value,)
            )
            fail_conn.commit()
            fail_rows_affected = fail_cursor.rowcount
            fail_cursor.close()
            fail_conn.close()
            if fail_rows_affected:
                celery_logger.info(f"Phieutin DB Task ({request_id}): status=4 set for name='{key_value}'")
            else:
                celery_logger.info(f"Phieutin DB Task ({request_id}): skip update for name='{key_value}' — already verified (status=5)")
        except Exception as db_err:
            celery_logger.error(f"Phieutin DB Task ({request_id}): Failed to set status=4: {db_err}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=30)
        return context

    finally:
        if temp_folder and os.path.exists(temp_folder):
            try:
                shutil.rmtree(temp_folder)
                celery_logger.info(f"Phieutin DB Task ({request_id}): Cleaned up temp folder {temp_folder}")
            except Exception as cleanup_error:
                celery_logger.error(f"Phieutin DB Task ({request_id}): Cleanup failed: {cleanup_error}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
