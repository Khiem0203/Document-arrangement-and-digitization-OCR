#!/bin/bash
# start-service.sh - Runs the application with a 3-stage dynamic routing pipeline.

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

CONDA_PATH="/path/to/miniconda3/condabin/conda"
CONDA_ENV_NAME="ocr"
PROJECT_DIR="/path/to/OCR_CELERY"

# Derived paths — không cần sửa
CONDA_ENV_BIN="$(dirname "$CONDA_PATH")/../envs/$CONDA_ENV_NAME/bin"
PYTHON="$CONDA_ENV_BIN/python"
UVICORN="$CONDA_ENV_BIN/uvicorn"
CELERY="$CONDA_ENV_BIN/celery"

# --- Worker counts ---
DEFAULT_PREPARE_WORKERS=2
DEFAULT_PROCESS_WORKERS=4
DEFAULT_WEBHOOK_WORKERS=4
DEFAULT_DB_WORKERS=4
DEFAULT_PHIEUTIN_PREPARE_WORKERS=1
DEFAULT_PHIEUTIN_PROCESS_WORKERS=2
DEFAULT_PHIEUTIN_DB_WORKERS=2

CPU_CONCURRENCY=1
GPU_CONCURRENCY=1

# ==============================================================================
#  LOAD .env
# ==============================================================================
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded env from $ENV_FILE"
else
    echo "Warning: $ENV_FILE not found, using shell environment variables."
fi

# Validate required vars
if [ -z "$CELERY_BROKER_URL" ] || [ -z "$CELERY_RESULT_BACKEND" ]; then
    echo "Error: CELERY_BROKER_URL and CELERY_RESULT_BACKEND must be set." >&2
    exit 1
fi

echo "Using CELERY_BROKER_URL:    $CELERY_BROKER_URL"
echo "Using CELERY_RESULT_BACKEND: $CELERY_RESULT_BACKEND"

# Parse PostgreSQL host/port/user/db từ CELERY_RESULT_BACKEND
# Format: db+postgresql://user:pass@host:port/db
_PG_URL="${CELERY_RESULT_BACKEND#db+postgresql://}"
_PG_URL="${_PG_URL#postgresql://}"
PG_HOST=$(echo "$_PG_URL" | sed 's/.*@//' | cut -d: -f1 | cut -d/ -f1)
PG_PORT=$(echo "$_PG_URL" | sed 's/.*@//' | cut -d: -f2 | cut -d/ -f1)
PG_USER=$(echo "$_PG_URL" | cut -d: -f1)
PG_DB=$(echo "$_PG_URL"   | sed 's/.*\///' | cut -d? -f1)
PG_PORT="${PG_PORT:-5432}"

# ==============================================================================
#  APPLICATION STARTUP
# ==============================================================================
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR/api" "$LOG_DIR/chinhly" "$LOG_DIR/phieutin" "$LOG_DIR/process" "$LOG_DIR/db" "$LOG_DIR/spellfix"

# Activate conda
if [ -f "$(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh" ]; then
    source "$(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Check your Conda installation."
    exit 1
fi

# Wait for PostgreSQL
echo "Waiting for PostgreSQL at $PG_HOST:$PG_PORT/$PG_DB ..."
for i in {1..30}; do
    if pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" --timeout=1 > /dev/null 2>&1; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "  Not ready yet, waiting 5s... ($i/30)"
    sleep 5
done

if ! pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" --timeout=1 > /dev/null 2>&1; then
    echo "Error: Timed out waiting for PostgreSQL." >&2
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then echo "Error: Failed to activate Conda env '$CONDA_ENV_NAME'."; exit 1; fi

cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory $PROJECT_DIR"; exit 1; }

# ==============================================================================
#  SERVICE FUNCTIONS
# ==============================================================================

start_fastapi_server() {
    local log_file="$LOG_DIR/api/api_server_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting FastAPI API Server on port 9090..."
    nohup "$UVICORN" app.main_celery:app --host 0.0.0.0 --port 9090 > "$log_file" 2>&1 &
    echo "  PID: $!  Log: $log_file"
}

start_detectron2_service() {
    local log_file="$LOG_DIR/chinhly/detectron2_service_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting Detectron2 Microservice on port 8003..."
    nohup "$UVICORN" app.detectron2_api:app --host 0.0.0.0 --port 8003 > "$log_file" 2>&1 &
    echo "  PID: $!  Log: $log_file"
}

start_spellfix_service() {
    local log_file="$LOG_DIR/spellfix/spellfix_service_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting ProtonX Spellfix Microservice on port 8006..."
    nohup "$UVICORN" app.spellfix_api:app --host 0.0.0.0 --port 8006 > "$log_file" 2>&1 &
    echo "  PID: $!  Log: $log_file"
}

start_detectron2_phieutin_service() {
    local log_file="$LOG_DIR/phieutin/detectron2_phieutin_service_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting Detectron2 Phieutin Microservice on port 8010..."
    nohup "$UVICORN" app.detectron2_phieutin_api:app --host 0.0.0.0 --port 8010 > "$log_file" 2>&1 &
    echo "  PID: $!  Log: $log_file"
}

_get_log_subdir() {
    case "$1" in
        prepare_queue|webhook_queue) echo "chinhly" ;;
        process_queue)               echo "process" ;;
        db_queue)                    echo "db" ;;
        phieutin_prepare_queue|\
        phieutin_process_queue|\
        phieutin_db_queue)           echo "phieutin" ;;
        *)                           echo "chinhly" ;;
    esac
}

start_celery_workers() {
    local queue_name=$1 num_workers=$2 concurrency=$3
    local subdir; subdir=$(_get_log_subdir "$queue_name")
    if [ "$num_workers" -eq 0 ]; then
        echo "Skipping workers for queue '$queue_name' (count is 0)."
        return
    fi
    echo "Starting $num_workers worker(s) for queue: '$queue_name' (concurrency: $concurrency)"
    for ((i=1; i<=num_workers; i++)); do
        local log_file="$LOG_DIR/$subdir/${queue_name}_worker_${i}_$(date +%Y%m%d_%H%M%S).log"
        "$CELERY" -A app.celery_app worker \
            --loglevel=INFO \
            --queues="$queue_name" \
            --hostname="${queue_name}_worker_${i}@%h" \
            --concurrency="$concurrency" \
            --logfile="$log_file" \
            --detach
        echo "  -> Worker $i started. Log: $log_file"
    done
}

# ==============================================================================
#  MAIN
# ==============================================================================
echo "========================================"
echo "  Starting OCR Pipeline Services"
echo "========================================"

start_detectron2_service
start_detectron2_phieutin_service
start_spellfix_service
sleep 3

# Chờ ProtonX nạp xong model trước khi khởi động API/worker: worker chỉ kiểm tra ProtonX một lần
# lúc khởi động, nếu lúc đó chưa sẵn sàng thì ProtonX bị bỏ qua tới lần khởi động lại sau.
if [ "${ENABLE_ML_SPELL_FIX:-true}" = "true" ]; then
    _SPELLFIX_BASE=$(echo "${SPELLFIX_API_URL:-http://localhost:8006/spellfix/correct}" | sed -E 's#^(https?://[^/]+).*#\1#')
    _SPELLFIX_WAIT=${SPELLFIX_WAIT_SECONDS:-120}
    echo "Waiting for ProtonX spellfix at $_SPELLFIX_BASE/health (max ${_SPELLFIX_WAIT}s) ..."
    _waited=0
    until curl -sf "$_SPELLFIX_BASE/health" > /dev/null 2>&1; do
        if [ "$_waited" -ge "$_SPELLFIX_WAIT" ]; then
            echo "Warning: ProtonX not ready after ${_SPELLFIX_WAIT}s -> workers will bypass ProtonX (flow: OCR -> rules)."
            break
        fi
        sleep 5
        _waited=$((_waited + 5))
    done
    [ "$_waited" -lt "$_SPELLFIX_WAIT" ] && echo "ProtonX spellfix is ready."
fi

start_fastapi_server
sleep 3

echo "--- Starting Celery Worker Groups ---"
start_celery_workers "prepare_queue"          "$DEFAULT_PREPARE_WORKERS"          "$CPU_CONCURRENCY"
start_celery_workers "process_queue"          "$DEFAULT_PROCESS_WORKERS"          "$GPU_CONCURRENCY"
start_celery_workers "webhook_queue"          "$DEFAULT_WEBHOOK_WORKERS"          "$CPU_CONCURRENCY"
start_celery_workers "db_queue"               "$DEFAULT_DB_WORKERS"               "$CPU_CONCURRENCY"
start_celery_workers "phieutin_prepare_queue"  "$DEFAULT_PHIEUTIN_PREPARE_WORKERS"  "$CPU_CONCURRENCY"
start_celery_workers "phieutin_process_queue"  "$DEFAULT_PHIEUTIN_PROCESS_WORKERS"  "$GPU_CONCURRENCY"
start_celery_workers "phieutin_db_queue"       "$DEFAULT_PHIEUTIN_DB_WORKERS"       "$CPU_CONCURRENCY"

echo ""
echo "All application services have been started."
echo "--------------------------------"
echo "RAM Usage:"
free -h | awk '/^Mem:/ {print "  Used: "$3" / Total: "$2" (Free: "$4")"}'
echo "VRAM Usage (GPU):"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | awk -F', ' '{print "  Total: "$1" MiB, Used: "$2" MiB, Free: "$3" MiB"}'
echo "--------------------------------"
echo "To stop: ./stop-service.sh"
echo "--------------------------------"
