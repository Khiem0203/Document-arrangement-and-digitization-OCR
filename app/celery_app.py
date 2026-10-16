from celery import Celery
from celery.signals import worker_ready, worker_process_init
from kombu import Exchange, Queue
import os
import logging

from dotenv import load_dotenv
load_dotenv()

_log = logging.getLogger(__name__)

_STARTUP_LOG = os.path.join(
    os.getenv("PROJECT_DIR", "/path/to/OCR_CELERY"),
    "logs", "startup_connections.log",
)


def _startup_log(msg: str):
    """Ghi vào file startup log tập trung và đồng thời vào worker log."""
    _log.info(msg)
    try:
        os.makedirs(os.path.dirname(_STARTUP_LOG), exist_ok=True)
        with open(_STARTUP_LOG, "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


@worker_ready.connect
def _check_connections(sender, **kwargs):
    import psycopg2
    from urllib.parse import urlparse, unquote

    worker_name = getattr(sender, "hostname", "unknown")
    _startup_log("=" * 60)
    _startup_log(f"  WORKER STARTUP — {worker_name}")
    _startup_log("=" * 60)

    # --- PostgreSQL (celery_results) ---
    backend_url = os.getenv("CELERY_RESULT_BACKEND", "")
    url = backend_url.replace("db+postgresql://", "postgresql://", 1).split("?")[0]
    p = urlparse(url)
    pg_host = p.hostname or "localhost"
    pg_port = p.port or 5432
    pg_db   = p.path.lstrip("/")
    try:
        password = unquote(p.password) if p.password else None
        conn = psycopg2.connect(
            host=pg_host, port=pg_port, database=pg_db,
            user=p.username, password=password, connect_timeout=5,
        )
        conn.close()
        _startup_log(f"  [OK]   PostgreSQL  {pg_host}:{pg_port}/{pg_db}")
    except Exception as e:
        _startup_log(f"  [FAIL] PostgreSQL  {pg_host}:{pg_port}/{pg_db}  →  {e}")

    # --- Model services ---
    import requests

    def _base(url: str) -> str:
        u = urlparse(url)
        return f"{u.scheme}://{u.netloc}"

    vllm = os.getenv("VLLM_API_URL", "http://localhost:8005/v1/chat/completions")
    det = os.getenv("DETECTRON2_API_URL", "http://localhost:8003/detectron2/detect")
    det_pt = os.getenv("DETECTRON2_PHIEUTIN_API", "http://localhost:8010/detectron2_phieutin/detect")
    checks = [
        ("vLLM", f"{_base(vllm)}/health"),
        ("Detectron2", f"{_base(det)}/docs"),
        ("Detectron2 PT", f"{_base(det_pt)}/docs"),
    ]
    for name, url in checks:
        try:
            requests.get(url, timeout=3).raise_for_status()
            _startup_log(f"  [OK]   {name:<12} {url}")
        except Exception as e:
            _startup_log(f"  [FAIL] {name:<12} {url}  →  {e}")

    from .utils import ENABLE_ML_SPELL_FIX, spellfix_health_url
    if not ENABLE_ML_SPELL_FIX:
        _startup_log("  [SKIP] ProtonX       ENABLE_ML_SPELL_FIX=false  →  flow: OCR → quy tắc")
    else:
        url = spellfix_health_url()
        try:
            requests.get(url, timeout=3).raise_for_status()
            _startup_log(f"  [OK]   ProtonX      {url}  →  flow: OCR → ProtonX → quy tắc")
        except Exception as e:
            _startup_log(f"  [FAIL] ProtonX      {url}  →  {e}")
            _startup_log("         ProtonX bị bỏ qua cho tới khi khởi động lại worker  →  flow: OCR → quy tắc")

    _startup_log("=" * 60)


@worker_process_init.connect
def _init_spellfix_state(**kwargs):
    """Mỗi tiến trình con kiểm tra ProtonX một lần; request sau dùng lại kết quả, không kiểm tra lại."""
    from .utils import check_spellfix
    check_spellfix(force=True)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
if not CELERY_BROKER_URL or not CELERY_RESULT_BACKEND:
    raise ValueError("CELERY_BROKER_URL and CELERY_RESULT_BACKEND must be set.")

celery_app = Celery(
    "worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['app.tasks_new']
)

# --- NEW Simplified Queue setup for the 3-stage pipeline ---
prepare_exchange          = Exchange('prepare_exchange',          type='direct')
process_exchange          = Exchange('process_exchange',          type='direct')
webhook_exchange          = Exchange('webhook_exchange',          type='direct')
db_exchange               = Exchange('db_exchange',               type='direct')
phieutin_prepare_exchange = Exchange('phieutin_prepare_exchange', type='direct')
phieutin_process_exchange = Exchange('phieutin_process_exchange', type='direct')
phieutin_db_exchange      = Exchange('phieutin_db_exchange',      type='direct')

celery_app.conf.task_queues = (
    Queue('prepare_queue',           prepare_exchange,          routing_key='prepare_task'),
    Queue('process_queue',           process_exchange,          routing_key='process_task'),
    Queue('webhook_queue',           webhook_exchange,          routing_key='webhook_task'),
    Queue('db_queue',                db_exchange,               routing_key='db_task'),
    Queue('phieutin_prepare_queue',  phieutin_prepare_exchange, routing_key='phieutin_prepare_task'),
    Queue('phieutin_process_queue',  phieutin_process_exchange, routing_key='phieutin_process_task'),
    Queue('phieutin_db_queue',       phieutin_db_exchange,      routing_key='phieutin_db_task'),
)

# --- NEW Simplified Routes ---
celery_app.conf.task_routes = {
    'routing_and_preparation_task':  {'queue': 'prepare_queue',          'routing_key': 'prepare_task'},
    'dispatch_and_process_task':     {'queue': 'process_queue',          'routing_key': 'process_task'},
    'send_webhook_task':             {'queue': 'webhook_queue',          'routing_key': 'webhook_task'},
    'save_results_to_db_task':       {'queue': 'db_queue',               'routing_key': 'db_task'},
    'phieutin_prepare_task':         {'queue': 'phieutin_prepare_queue', 'routing_key': 'phieutin_prepare_task'},
    'phieutin_process_single_task':  {'queue': 'phieutin_prepare_queue', 'routing_key': 'phieutin_prepare_task'},
    'save_phieutin_to_db_task':      {'queue': 'phieutin_db_queue',      'routing_key': 'phieutin_db_task'},
    'ocr_process_task':              {'queue': 'process_queue',          'routing_key': 'process_task'},
}

# --- Other Celery settings ---
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    database_engine_options={'pool_recycle': 3600},

    # Prefetch & acks_late: cho phép trả task về RabbitMQ khi pause/stop
    worker_prefetch_multiplier=1,   # Chỉ lấy 1 task/process (mặc định 4)
    task_acks_late=True,            # Acknowledge SAU khi hoàn thành (không phải khi nhận)
    task_reject_on_worker_lost=True,  # Worker chết → task trả về RabbitMQ
)
