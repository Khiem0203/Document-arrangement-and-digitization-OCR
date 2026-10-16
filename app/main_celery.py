from fastapi import FastAPI, HTTPException
from typing import Optional
import logging
import datetime
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from celery import chain
from .celery_app import celery_app
import requests
import os
from dotenv import load_dotenv
from .config import rabbitmq_url
import psycopg2

load_dotenv()   

from .vision_service import call_llm_for_profile_name
# Import the new task names

from .tasks_new import (
    routing_and_preparation_task,
    dispatch_and_process_task,
    send_webhook_task,
    ocr_process_task,
    save_results_to_db_task,
    phieutin_prepare_task,
    save_phieutin_to_db_task,
)

from .models import (
    OCRRequest, SubmitResponse, PhieutinRequest, PhieutinSubmitResponse,
    StatusResponse, ImageProcessingRequest,
    ProfileRequest, ProfileResponse,
    QueueStatusResponse, QueueControlResponse, WorkerInfo,
    # Worker Management Models
    WorkerStartRequest, WorkerStopRequest, WorkerRestartRequest,
    WorkerStartResponse, WorkerStopResponse, WorkerRestartResponse,
    WorkerStatusResponse, WorkerPresetsResponse, WorkerPreset,
    SuggestionResponse, SuggestionItem
)

# Import worker manager
from .worker_manager import (
    start_workers, stop_workers, restart_workers,
    get_worker_status, get_presets,
    pause_queue, resume_queue, get_pause_status,
    force_pause_queue
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Separate logger for /api/crop-ocr
_crop_ocr_logger = logging.getLogger("crop_ocr")
_crop_ocr_logger.setLevel(logging.INFO)
_crop_ocr_logger.propagate = False
_crop_ocr_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "crop_ocr")
os.makedirs(_crop_ocr_log_dir, exist_ok=True)
_crop_ocr_handler = logging.FileHandler(
    os.path.join(_crop_ocr_log_dir, "crop_ocr.log"),
    encoding="utf-8",
)
_crop_ocr_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
_crop_ocr_logger.addHandler(_crop_ocr_handler)


# env_var = requests.get("http://your-config-server:8989/config/ocr-service")
# env_var.raise_for_status()
# env_var = env_var.json() 
# for key, value in env_var.items():
#     os.environ[key] = str(value)
# logger.info(f"OCR Service Env: {os.getenv('VISION_MODEL')}")
# logger.info(f"OCR Service Env: {os.getenv('CELERY_BROKER_URL')}")
# logger.info(f"OCR Service Env: {os.getenv('CELERY_RESULT_BACKEND')}")


app = FastAPI(
    title="Dynamic OCR Webhook Service (Celery)",
    description="Asynchronous OCR processing API with dynamic workflow routing",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _init_spellfix_state():
    """/api/crop-ocr chạy trong tiến trình API: kiểm tra ProtonX một lần lúc khởi động."""
    from .utils import check_spellfix
    check_spellfix(force=True)


@app.post("/api/create_profile_name", response_model=ProfileResponse,status_code=200)
async def create_profile_name(request: ProfileRequest):
    """
    Receive a list of file subject and return the name of the profile
    """
    logger.info(f"Received profile creation request for customer_id: {request.customer_id}, config_template_id: {request.config_template_id} at {datetime.datetime.now().isoformat()}")
    profile_name = call_llm_for_profile_name(request.info)
    logger.info(f"profile name is {profile_name}")
    return ProfileResponse(
        customer_id=request.customer_id,
        config_template_id=request.config_template_id,
        profile_name=profile_name)

@app.post("/api/submit-ocr", response_model=SubmitResponse, status_code=202)
async def submit_ocr_request(request: OCRRequest):
    """
    Submits an OCR processing job. This starts a 3-stage pipeline
    with dynamic workflow routing in the second stage.
    """
    logger.info(f"Received OCR request for S3 path: {request.s3_path}, Template ID: {request.config_template_id}")
    
    # --- The new, simplified chain definition ---
    workflow = chain(
        # Task 1: Lightweight, I/O-bound, with routing logic
        routing_and_preparation_task.s(
            s3_path=request.s3_path,
            customer_id=request.customer_id,
            config_template_id=request.config_template_id,
            access_key=request.access_key,
            secret_key=request.secret_key,
            endpoint=request.endpoint,
            callback_url=str(request.callback_url),
            document_id=request.document_id
        ).set(queue='prepare_queue'),

        # Task 2: Heavy GPU task (dispatcher calls the correct flow)
        dispatch_and_process_task.s().set(queue='process_queue'),

        # Task 3: Lightweight, network-bound (callback_url now in context)
        send_webhook_task.s().set(queue='webhook_queue'),

        # Task 4: Save results to database (callback_url now in context)
        save_results_to_db_task.s().set(queue='db_queue'),
    )
    
    task_chain = workflow.apply_async()
    
    return SubmitResponse(
        error_code=202,
        error_message="Đã nhận yêu cầu xử lý OCR. Kết quả sẽ được gửi đến callback URL khi hoàn thành.",
        request_id=task_chain.id 
    )


@app.post("/api/submit-phieutin", response_model=PhieutinSubmitResponse, status_code=202)
async def submit_phieutin_request(request: PhieutinRequest):
    """
    Submits a Phieutin OCR job. Runs a 3-task chain:
    phieutin_prepare_task → dispatch_and_process_task → save_phieutin_to_db_task
    Result is written directly to the dataslips table (status=10 success, status=20 failure).
    """
    logger.info(f"Received Phieutin request: bucket={request.s3_bucket}, prefix={request.s3_prefix}")

    task = phieutin_prepare_task.apply_async(
        kwargs=dict(
            s3_bucket=request.s3_bucket,
            s3_prefix=request.s3_prefix,
            companyid=request.companyid,
            access_key=request.access_key,
            secret_key=request.secret_key,
            endpoint=request.endpoint,
        ),
        queue='phieutin_prepare_queue',
    )

    return PhieutinSubmitResponse(
        error_code=202,
        error_message="Đã nhận yêu cầu xử lý phiếu tin. Các file sẽ được xử lý song song và cập nhật vào DB khi hoàn thành.",
        request_id=task.id
    )


@app.get("/api/suggestions", response_model=SuggestionResponse)
async def get_suggestions(
    field_type: str = "ORG", 
    q: str = "", 
    customer_id: Optional[int] = None,
    limit: int = 10
):
    """
    Get word suggestions for manual data entry based on historical OCR results.
    field_type: 'SIGNER' or 'ORG'
    q: partial text to search (case-insensitive prefix)
    """
    logger.info(f"Suggestions requested: field_type={field_type}, q='{q}', customer_id={customer_id}")
    
    # Define codes to search for based on type
    if field_type.upper() == "ORG":
        codes = ("CQBH", "OrganName", "TG", "ORG")
    elif field_type.upper() == "SIGNER":
        codes = ("NK", "Signer", "NKVB", "SIGNER")
    elif field_type.upper() == "VV":
        codes = ("TTNDVB", "Subject", "VV", "TLVTY")
    elif field_type.upper() in ("THELOAI", "TIEUDE"):
        codes = ("TLVB", "TypeName", "THELOAI", "TLVTY")
    else:
        # Generic case: try to use field_type as code itself
        codes = (field_type, field_type.upper())

    try:
        # Use connection details from tasks_new logic
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="celery_results",
            user="your_celery_db_user",
            password="your_celery_db_password"
        )
        cursor = conn.cursor()

        # Build query - Use Postgres JSONB functions to extract and aggregate
        query_sql = """
        WITH field_values AS (
            SELECT 
                f->>'Value' as val,
                f->>'Code' as code
            FROM 
                ocr_results,
                jsonb_array_elements(final_json->'fields') f
            WHERE 
                (final_json->'fields') IS NOT NULL
                {customer_filter}
        )
        SELECT 
            val, 
            COUNT(*) as freq
        FROM 
            field_values
        WHERE 
            val IS NOT NULL AND val != ''
            AND code IN %s
            AND val ILIKE %s
        GROUP BY 
            val
        ORDER BY 
            freq DESC
        LIMIT %s;
        """
        
        customer_filter = "AND customer_id = %s" if customer_id else ""
        query_sql = query_sql.format(customer_filter=customer_filter)
        
        params = []
        if customer_id:
            params.append(customer_id)
        params.extend([codes, f"{q}%", limit])

        cursor.execute(query_sql, tuple(params))
        rows = cursor.fetchall()

        suggestions = [
            SuggestionItem(value=row[0], frequency=row[1])
            for row in rows
        ]

        cursor.close()
        conn.close()

        return SuggestionResponse(field_type=field_type, suggestions=suggestions)

    except Exception as e:
        logger.error(f"Error fetching suggestions: {e}", exc_info=True)
        # Return empty list on error to avoid breaking frontend
        return SuggestionResponse(field_type=field_type, suggestions=[])


@app.post("/api/crop-ocr")
async def crop_ocr_request(image: ImageProcessingRequest):
    """
    OCR cropped image
    """
    import time
    import asyncio
    from functools import partial
    t0 = time.time()
    _crop_ocr_logger.info("Received request")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(ocr_process_task, image_base64=image.image)
        )
        elapsed = time.time() - t0
        _crop_ocr_logger.info(f"OK | elapsed={elapsed:.2f}s | result={result!r}")
    except Exception as e:
        elapsed = time.time() - t0
        _crop_ocr_logger.error(f"ERROR | elapsed={elapsed:.2f}s | error={e}", exc_info=True)
        result = ""
    return {"result": result.strip()}

# ... (get_request_status and health_check remain the same) ...
@app.get("/api/request-status/{request_id}", response_model=StatusResponse)
async def get_request_status(request_id: str):
    """
    Checks the status of a submitted OCR job.
    """
    task_result = AsyncResult(request_id, app=celery_app)
    
    result_info = task_result.info
    if isinstance(task_result.info, Exception):
        result_info = str(task_result.info)

    return StatusResponse(
        request_id=request_id,
        status=task_result.state,
        result=result_info
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}


# ==============================================================================
# Queue Management APIs
# ==============================================================================

@app.get("/api/queue/process-queue/status", response_model=QueueStatusResponse)
async def get_process_queue_status():
    """
    API 1: Get status of process_queue including active workers and pending tasks.

    Returns:
    - queue_name: Name of the queue
    - total_workers: Total number of workers handling this queue
    - active_workers: Number of workers currently processing tasks
    - total_active_tasks: Number of tasks currently being executed
    - total_reserved_tasks: Number of tasks waiting in worker's queue
    - messages_in_queue: Number of messages in RabbitMQ queue
    - workers_detail: List of workers with their task counts
    - status: Overall queue status (idle/busy/paused)
    """
    try:
        from celery import current_app
        inspect = current_app.control.inspect()

        # Get active and reserved tasks
        active_tasks = inspect.active() or {}
        reserved_tasks = inspect.reserved() or {}

        # Collect information about workers handling process_queue
        workers_detail = []
        total_active = 0
        total_reserved = 0
        active_worker_count = 0

        # Get all workers
        all_workers = set(list(active_tasks.keys()) + list(reserved_tasks.keys()))

        for worker in all_workers:
            # Check if this worker handles process_queue
            if 'process_queue' not in worker:
                continue

            active_count = 0
            reserved_count = 0

            # Count active tasks
            if worker in active_tasks:
                for task in active_tasks[worker]:
                    if 'process_queue' in task.get('delivery_info', {}).get('routing_key', ''):
                        active_count += 1

            # Count reserved tasks
            if worker in reserved_tasks:
                for task in reserved_tasks[worker]:
                    if 'process_queue' in task.get('delivery_info', {}).get('routing_key', ''):
                        reserved_count += 1

            total_active += active_count
            total_reserved += reserved_count

            if active_count > 0:
                active_worker_count += 1

            workers_detail.append(WorkerInfo(
                worker_name=worker,
                active_tasks=active_count,
                reserved_tasks=reserved_count
            ))

        # Try to get RabbitMQ queue length (optional, may fail if management API not enabled)
        messages_in_queue = 0
        try:
            from requests.auth import HTTPBasicAuth
            response = requests.get(rabbitmq_url, auth=HTTPBasicAuth("guest", "guest"), timeout=2)
            if response.status_code == 200:
                queue_info = response.json()
                messages_in_queue = queue_info.get("messages_ready", 0)
        except Exception as e:
            logger.warning(f"Could not get RabbitMQ queue length: {e}")

        # Determine status
        if total_active > 0 or total_reserved > 0 or messages_in_queue > 0:
            status = "busy"
        else:
            status = "idle"

        return QueueStatusResponse(
            queue_name="process_queue",
            total_workers=len(workers_detail),
            active_workers=active_worker_count,
            total_active_tasks=total_active,
            total_reserved_tasks=total_reserved,
            messages_in_queue=messages_in_queue,
            workers_detail=workers_detail,
            status=status
        )

    except Exception as e:
        logger.error(f"Error getting process_queue status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting queue status: {str(e)}")


# ==============================================================================
# Worker Management APIs
# ==============================================================================

@app.post("/api/workers/start", response_model=WorkerStartResponse)
async def api_start_workers(request: WorkerStartRequest):
    """Start Celery workers for a queue."""
    try:
        logger.info(f"Starting {request.num_workers} worker(s) for {request.queue_name} (concurrency={request.concurrency})")
        result = start_workers(
            queue_name=request.queue_name,
            num_workers=request.num_workers,
            concurrency=request.concurrency
        )
        return WorkerStartResponse(**result)
    except Exception as e:
        logger.error(f"Error starting workers: {e}", exc_info=True)
        return WorkerStartResponse(
            status="error",
            message=f"Failed to start workers: {str(e)}",
            pids=[]
        )


@app.post("/api/workers/stop", response_model=WorkerStopResponse)
async def api_stop_workers(request: WorkerStopRequest):
    """Stop all workers for a queue via Flower shutdown API."""
    try:
        logger.info(f"Stopping all workers for {request.queue_name}")
        result = stop_workers(queue_name=request.queue_name)
        return WorkerStopResponse(**result)
    except Exception as e:
        logger.error(f"Error stopping workers: {e}", exc_info=True)
        return WorkerStopResponse(
            status="error",
            message=f"Failed to stop workers: {str(e)}",
            stopped=[]
        )


@app.post("/api/workers/restart", response_model=WorkerRestartResponse)
async def api_restart_workers(request: WorkerRestartRequest):
    """Restart workers: shutdown via Flower then start new ones."""
    try:
        logger.info(f"Restarting workers for {request.queue_name}")
        result = restart_workers(
            queue_name=request.queue_name,
            num_workers=request.num_workers,
            concurrency=request.concurrency
        )
        return WorkerRestartResponse(**result)
    except Exception as e:
        logger.error(f"Error restarting workers: {e}", exc_info=True)
        return WorkerRestartResponse(
            status="error",
            message=f"Failed to restart workers: {str(e)}",
            stop_result={},
            start_result={}
        )


@app.get("/api/workers/status", response_model=WorkerStatusResponse)
async def api_worker_status(queue_name: str = None):
    """Get worker status from Flower API (real-time, reliable)."""
    try:
        logger.info(f"Getting worker status (filter: {queue_name})")
        result = get_worker_status(queue_name=queue_name)
        return WorkerStatusResponse(**result)
    except Exception as e:
        logger.error(f"Error getting worker status: {e}", exc_info=True)
        return WorkerStatusResponse(
            total_workers=0,
            workers=[],
            filter=queue_name
        )


@app.get("/api/workers/presets", response_model=WorkerPresetsResponse)
async def api_worker_presets():
    """Get available worker configuration presets."""
    try:
        presets_dict = get_presets()
        # Convert to Pydantic models
        presets = {
            name: WorkerPreset(**config)
            for name, config in presets_dict.items()
        }
        return WorkerPresetsResponse(presets=presets)
    except Exception as e:
        logger.error(f"Error getting presets: {e}", exc_info=True)
        return WorkerPresetsResponse(presets={})


@app.post("/api/workers/start-preset")
async def api_start_preset(request: WorkerStopRequest):
    """Start workers using preset configuration."""
    try:
        presets = get_presets()

        if request.queue_name not in presets:
            raise HTTPException(
                status_code=400,
                detail=f"No preset found for queue '{request.queue_name}'. Available: {list(presets.keys())}"
            )

        preset = presets[request.queue_name]
        logger.info(f"Starting workers for {request.queue_name} with preset: {preset}")

        result = start_workers(
            queue_name=request.queue_name,
            num_workers=preset["num_workers"],
            concurrency=preset["concurrency"]
        )

        return WorkerStartResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting preset workers: {e}", exc_info=True)
        return WorkerStartResponse(
            status="error",
            message=f"Failed to start workers: {str(e)}",
            pids=[]
        )
