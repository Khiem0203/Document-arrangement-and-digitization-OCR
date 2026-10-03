# app/models.py
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, Any, Optional

# This file contains all shared Pydantic models to prevent circular imports.

# --- API Request/Response Models ---

class OCRRequest(BaseModel):
    s3_path: str
    customer_id: int
    config_template_id: int
    callback_url: HttpUrl
    access_key: str = Field(..., description="S3-compatible access key")
    secret_key: str = Field(..., description="S3-compatible secret key")
    endpoint: str = Field(..., description="S3-compatible endpoint URL")
    document_id: int

class SubmitResponse(BaseModel):
    error_code: int
    error_message: str
    request_id: str


class StatusResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Any] = None


# --- Webhook Payload Models ---

class OCRMetaResponse(BaseModel):
    error_code: int
    error_message: str

class OCRDataResponse(BaseModel):
    path: str
    raw_text: str
    type_document: int
    creator: str
    created_at: str
    json: Dict[str, Any]
    verified: bool = False
    document_id: Optional[int] = None

class OCRWebhookResponse(BaseModel):
    meta: OCRMetaResponse
    data: OCRDataResponse
    request_id: str


class ImageProcessingRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data")


class ProfileRequest(BaseModel):
    customer_id: int
    config_template_id: int
    info: str

class ProfileResponse(BaseModel):
    customer_id: int
    config_template_id: int
    profile_name: str


# --- Queue Management Models ---

class WorkerInfo(BaseModel):
    worker_name: str
    active_tasks: int
    reserved_tasks: int

class QueueStatusResponse(BaseModel):
    queue_name: str
    total_workers: int
    active_workers: int
    total_active_tasks: int
    total_reserved_tasks: int
    messages_in_queue: int
    workers_detail: list[WorkerInfo]
    status: str  # "idle", "busy", "paused"

class QueueControlResponse(BaseModel):
    queue_name: str
    action: str  # "paused" or "resumed"
    status: str  # "success" or "error"
    message: str


# --- Worker Management Models ---

class WorkerStartRequest(BaseModel):
    queue_name: str = Field(..., description="Queue name (e.g., process_queue)")
    num_workers: int = Field(..., ge=1, le=10, description="Number of workers to start (1-10)")
    concurrency: int = Field(..., ge=1, le=8, description="Concurrency per worker (1-8)")

class WorkerStopRequest(BaseModel):
    queue_name: str = Field(..., description="Queue name to stop workers")

class WorkerRestartRequest(BaseModel):
    queue_name: str = Field(..., description="Queue name to restart")
    num_workers: Optional[int] = Field(None, ge=1, le=10, description="Number of workers (optional, uses preset if not provided)")
    concurrency: Optional[int] = Field(None, ge=1, le=8, description="Concurrency (optional, uses preset if not provided)")

class WorkerPIDInfo(BaseModel):
    worker_id: int
    pid: int
    log_file: str

class WorkerStartResponse(BaseModel):
    status: str  # "success", "partial", "skipped"
    message: str
    pids: list[WorkerPIDInfo]
    failed: Optional[list[Dict[str, Any]]] = None

class WorkerStopResponse(BaseModel):
    status: str  # "success" or "not_found"
    message: str
    stopped: list[Dict[str, Any]]
    not_found: Optional[list[Dict[str, Any]]] = None

class WorkerRestartResponse(BaseModel):
    status: str
    message: str
    stop_result: Dict[str, Any]
    start_result: Dict[str, Any]

class WorkerStatusResponse(BaseModel):
    total_workers: int
    workers: list[Dict[str, Any]]
    filter: Optional[str] = None
    source: Optional[str] = None

class WorkerPreset(BaseModel):
    num_workers: int
    concurrency: int
    description: str

class WorkerPresetsResponse(BaseModel):
    presets: Dict[str, WorkerPreset]

class SuggestionItem(BaseModel):
    value: str
    frequency: int

class SuggestionResponse(BaseModel):
    field_type: str
    suggestions: list[SuggestionItem]


class PhieutinRequest(BaseModel):
    s3_bucket: str = Field(..., description="S3 bucket name")
    s3_prefix: str = Field(..., description="Prefix/folder path in bucket (e.g. 'phieutin/')")
    companyid: int
    access_key: str = Field(..., description="S3-compatible access key")
    secret_key: str = Field(..., description="S3-compatible secret key")
    endpoint: str = Field(..., description="S3-compatible endpoint URL")

class PhieutinSubmitResponse(BaseModel):
    error_code: int
    error_message: str
    request_id: str


