"""
Worker Manager Module - Flower API Backend

Manages Celery workers through Flower's HTTP API (port 5555).
Uses Flower for reliable worker control instead of PID-based management.

Flower API endpoints used:
- GET  /api/workers                                  → List workers & status
- POST /api/worker/shutdown/{name}                   → Shutdown worker
- POST /api/worker/pool/restart/{name}               → Restart worker pool
- POST /api/worker/queue/add-consumer/{name}?queue=  → Resume queue consumption
- POST /api/worker/queue/cancel-consumer/{name}?queue= → Pause queue consumption
- GET  /api/queues/length                            → Queue lengths
"""

import os
import subprocess
import time
import json
import httpx
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
STATE_DIR = os.path.join(BASE_DIR, "worker_state")

# Log subdirectory per queue
_QUEUE_LOG_SUBDIR = {
    "prepare_queue":          "chinhly",
    "webhook_queue":          "chinhly",
    "process_queue":          "process",
    "db_queue":               "db",
    "phieutin_prepare_queue": "phieutin",
    "phieutin_db_queue":      "phieutin",
}

for _subdir in set(_QUEUE_LOG_SUBDIR.values()):
    os.makedirs(os.path.join(LOG_DIR, _subdir), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# Flower API
FLOWER_URL = os.getenv("FLOWER_URL", "http://localhost:5555")
FLOWER_TIMEOUT = 10  # seconds

# Celery configuration (for starting new workers only)
CELERY_BIN = "/path/to/anaconda3/envs/ocr/bin/celery"
CELERY_APP = "app.celery_app"

# Worker presets
WORKER_PRESETS = {
    "prepare_queue":          {"num_workers": 4, "concurrency": 2, "description": "Preparation & Routing (CPU/IO-bound)"},
    "process_queue":          {"num_workers": 2, "concurrency": 1, "description": "Processing (GPU-bound)"},
    "webhook_queue":          {"num_workers": 4, "concurrency": 2, "description": "Webhook (Network-bound)"},
    "db_queue":               {"num_workers": 4, "concurrency": 2, "description": "Database Operations (I/O-bound)"},
    "phieutin_prepare_queue": {"num_workers": 2, "concurrency": 2, "description": "Phieutin Preparation (CPU/IO-bound)"},
    "phieutin_db_queue":      {"num_workers": 2, "concurrency": 2, "description": "Phieutin DB Update (I/O-bound)"},
}


# ==============================================================================
# Flower API helpers
# ==============================================================================

def _flower_get(path: str) -> Optional[dict]:
    """GET request to Flower API."""
    try:
        resp = httpx.get(f"{FLOWER_URL}{path}", timeout=FLOWER_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Flower GET {path} failed: {e}")
        return None


def _flower_post(path: str, **kwargs) -> bool:
    """POST request to Flower API. Returns True on success."""
    try:
        resp = httpx.post(f"{FLOWER_URL}{path}", timeout=FLOWER_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Flower POST {path} failed: {e}")
        return False


def _get_workers_for_queue(queue_name: str) -> List[str]:
    """Get list of worker names that consume from a specific queue."""
    data = _flower_get("/api/workers?refresh=true")
    if not data:
        return []

    workers = []
    for worker_name, info in data.items():
        active_queues = info.get("active_queues") or []
        for q in active_queues:
            if q.get("name") == queue_name:
                workers.append(worker_name)
                break
    return workers


def _get_all_workers() -> dict:
    """Get all workers from Flower."""
    return _flower_get("/api/workers?refresh=true") or {}


def _wait_for_worker_online(worker_hostname: str, timeout: int = 30) -> bool:
    """Wait until a worker appears in Flower's worker list."""
    start = time.time()
    while time.time() - start < timeout:
        data = _flower_get("/api/workers")
        if data:
            for name in data:
                if worker_hostname.replace("%h", "") in name:
                    return True
        time.sleep(2)
    return False


# ==============================================================================
# Core operations
# ==============================================================================

def start_workers(queue_name: str, num_workers: int, concurrency: int) -> Dict:
    """
    Start Celery workers for a specific queue.
    Uses subprocess (Flower can't start new workers).
    Verifies workers appear in Flower after starting.
    """
    if num_workers <= 0:
        return {"status": "skipped", "message": f"num_workers={num_workers}", "pids": []}

    pids = []
    failed = []

    for i in range(1, num_workers + 1):
        worker_hostname = f"{queue_name}_worker_{i}@%h"
        subdir = _QUEUE_LOG_SUBDIR.get(queue_name, "chinhly")
        log_file = os.path.join(
            LOG_DIR, subdir,
            f"{queue_name}_worker_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        cmd = [
            CELERY_BIN, "-A", CELERY_APP, "worker",
            "--loglevel=INFO",
            f"--queues={queue_name}",
            f"--hostname={worker_hostname}",
            f"--concurrency={concurrency}",
            f"--logfile={log_file}",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
                cwd=BASE_DIR
            )
            pids.append({"worker_id": i, "pid": process.pid, "log_file": log_file})
            logger.info(f"Started {queue_name} worker {i} (PID: {process.pid})")

        except Exception as e:
            failed.append({"worker_id": i, "error": str(e)})
            logger.error(f"Failed to start {queue_name} worker {i}: {e}")

    if failed:
        return {
            "status": "partial",
            "message": f"Started {len(pids)}/{num_workers} workers, {len(failed)} failed",
            "pids": pids, "failed": failed
        }

    return {
        "status": "success",
        "message": f"Started {len(pids)} worker(s) for {queue_name}",
        "pids": pids
    }


def stop_workers(queue_name: str) -> Dict:
    """
    Stop all workers for a queue via Flower shutdown API.
    Falls back to SIGTERM if Flower fails.
    """
    workers = _get_workers_for_queue(queue_name)

    if not workers:
        logger.warning(f"No workers found for {queue_name} in Flower")
        return {
            "status": "not_found",
            "message": f"No workers found for {queue_name}",
            "stopped": []
        }

    stopped = []
    failed = []

    for worker_name in workers:
        ok = _flower_post(f"/api/worker/shutdown/{worker_name}")
        if ok:
            stopped.append({"worker": worker_name})
            logger.info(f"Shutdown {worker_name} via Flower")
        else:
            failed.append({"worker": worker_name, "reason": "Flower shutdown failed"})
            logger.error(f"Failed to shutdown {worker_name}")

    return {
        "status": "success" if stopped else "error",
        "message": f"Stopped {len(stopped)}/{len(workers)} worker(s) for {queue_name}",
        "stopped": stopped,
        "not_found": failed if failed else None
    }


def restart_workers(queue_name: str, num_workers: int = None, concurrency: int = None) -> Dict:
    """Restart workers: stop via Flower, then start new ones."""
    stop_result = stop_workers(queue_name)

    # Wait for workers to fully stop
    time.sleep(3)

    if num_workers is None or concurrency is None:
        if queue_name in WORKER_PRESETS:
            preset = WORKER_PRESETS[queue_name]
            num_workers = num_workers or preset["num_workers"]
            concurrency = concurrency or preset["concurrency"]
        else:
            num_workers = num_workers or len(stop_result.get("stopped", []))
            concurrency = concurrency or 1

    start_result = start_workers(queue_name, num_workers, concurrency)

    return {
        "status": "success",
        "message": f"Restarted {queue_name}",
        "stop_result": stop_result,
        "start_result": start_result
    }


def get_worker_status(queue_name: str = None) -> Dict:
    """
    Get worker status from Flower API (reliable, real-time).
    """
    all_workers = _get_all_workers()
    workers = []

    for worker_name, info in all_workers.items():
        # Determine which queues this worker consumes
        active_queues = [q.get("name", "") for q in (info.get("active_queues") or [])]

        # Filter by queue if requested
        if queue_name and queue_name not in active_queues:
            continue

        stats = info.get("stats") or {}
        workers.append({
            "worker_name": worker_name,
            "queue": ", ".join(active_queues) if active_queues else "none",
            "status": "running",
            "stats": {
                "pool_size": stats.get("pool", {}).get("max-concurrency", "?"),
                "tasks_completed": stats.get("total", {}).get("tasks.total", 0),
            }
        })

    return {
        "total_workers": len(workers),
        "workers": workers,
        "filter": queue_name,
        "source": "flower"
    }


# ==============================================================================
# Pause / Resume via Flower cancel-consumer / add-consumer
# ==============================================================================

def pause_queue(queue_name: str) -> Dict:
    """
    Pause a queue by removing it from all workers' consumers via Flower.
    Workers stay alive but stop consuming new tasks from this queue.
    """
    workers = _get_workers_for_queue(queue_name)

    if not workers:
        # Check if there's already a pause state (paused before)
        if _load_pause_state(queue_name):
            return {
                "status": "already_paused",
                "message": f"Queue '{queue_name}' is already paused"
            }
        return {
            "status": "not_found",
            "message": f"No workers consuming '{queue_name}'"
        }

    # Save state before pausing
    if queue_name in WORKER_PRESETS:
        preset = WORKER_PRESETS[queue_name]
        _save_pause_state(queue_name, preset["num_workers"], preset["concurrency"])
    else:
        _save_pause_state(queue_name, len(workers), 1)

    # Cancel consumer on each worker
    paused_workers = []
    failed_workers = []

    for worker_name in workers:
        ok = _flower_post(
            f"/api/worker/queue/cancel-consumer/{worker_name}",
            params={"queue": queue_name}
        )
        if ok:
            paused_workers.append(worker_name)
            logger.info(f"Cancelled consumer '{queue_name}' on {worker_name}")
        else:
            failed_workers.append(worker_name)
            logger.error(f"Failed to cancel consumer '{queue_name}' on {worker_name}")

    if paused_workers:
        return {
            "status": "paused",
            "message": f"Paused '{queue_name}' on {len(paused_workers)} worker(s)",
            "paused_workers": paused_workers,
            "failed_workers": failed_workers if failed_workers else None
        }

    return {
        "status": "error",
        "message": f"Failed to pause '{queue_name}' on all workers",
        "failed_workers": failed_workers
    }


def resume_queue(queue_name: str) -> Dict:
    """
    Resume a paused queue by adding it back as consumer on workers via Flower.
    """
    # Get all workers (they should still be alive, just not consuming this queue)
    all_workers = _get_all_workers()

    if not all_workers:
        # No workers alive at all - need to start new ones
        logger.warning(f"No workers found in Flower, starting new ones for {queue_name}")
        pause_state = _load_pause_state(queue_name)
        if pause_state:
            num_workers = pause_state["num_workers"]
            concurrency = pause_state["concurrency"]
        elif queue_name in WORKER_PRESETS:
            preset = WORKER_PRESETS[queue_name]
            num_workers = preset["num_workers"]
            concurrency = preset["concurrency"]
        else:
            return {"status": "error", "message": f"No workers and no preset for '{queue_name}'"}

        start_result = start_workers(queue_name, num_workers, concurrency)
        _delete_pause_state(queue_name)
        return {
            "status": "resumed",
            "message": f"Started {len(start_result.get('pids', []))} new worker(s) for '{queue_name}'",
            "start_result": start_result
        }

    # Check if workers already consume this queue
    already_consuming = _get_workers_for_queue(queue_name)
    if already_consuming:
        _delete_pause_state(queue_name)
        return {
            "status": "already_running",
            "message": f"'{queue_name}' already has {len(already_consuming)} consumer(s)",
            "workers": already_consuming
        }

    # Find workers that should consume this queue
    # Use workers with matching hostname pattern, or any available worker
    target_workers = []
    for worker_name in all_workers:
        if queue_name in worker_name:
            target_workers.append(worker_name)

    # If no matching workers found, use all workers
    if not target_workers:
        target_workers = list(all_workers.keys())

    # Add consumer on target workers
    resumed_workers = []
    failed_workers = []

    for worker_name in target_workers:
        ok = _flower_post(
            f"/api/worker/queue/add-consumer/{worker_name}",
            params={"queue": queue_name}
        )
        if ok:
            resumed_workers.append(worker_name)
            logger.info(f"Added consumer '{queue_name}' on {worker_name}")
        else:
            failed_workers.append(worker_name)
            logger.error(f"Failed to add consumer '{queue_name}' on {worker_name}")

    if resumed_workers:
        _delete_pause_state(queue_name)
        return {
            "status": "resumed",
            "message": f"Resumed '{queue_name}' on {len(resumed_workers)} worker(s)",
            "resumed_workers": resumed_workers,
            "failed_workers": failed_workers if failed_workers else None
        }

    return {
        "status": "error",
        "message": f"Failed to resume '{queue_name}' on all workers",
        "failed_workers": failed_workers
    }


def force_pause_queue(queue_name: str, terminate_active: bool = True) -> Dict:
    """
    Force pause: cancel consumer + revoke reserved tasks + terminate active tasks.
    With acks_late=True, revoked/terminated tasks return to RabbitMQ automatically.

    Args:
        queue_name: Queue to force pause
        terminate_active: If True, also SIGTERM active (running) tasks
    """
    from .celery_app import celery_app

    # Step 1: Cancel consumer (stop fetching new tasks)
    pause_result = pause_queue(queue_name)
    logger.info(f"Force pause step 1 - cancel consumer: {pause_result}")

    # Step 2: Get active/reserved tasks via Celery inspect
    inspect = celery_app.control.inspect()
    active_tasks = inspect.active() or {}
    reserved_tasks = inspect.reserved() or {}

    revoked_reserved = []
    revoked_active = []
    errors = []

    # Step 3: Revoke reserved (prefetched) tasks for this queue
    for worker_name, tasks in reserved_tasks.items():
        if queue_name not in worker_name:
            continue
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            try:
                celery_app.control.revoke(task_id, terminate=False)
                revoked_reserved.append(task_id)
                logger.info(f"Revoked reserved task {task_id} on {worker_name}")
            except Exception as e:
                errors.append({"task_id": task_id, "error": str(e)})
                logger.error(f"Failed to revoke reserved task {task_id}: {e}")

    # Step 4: Terminate active (running) tasks if requested
    if terminate_active:
        for worker_name, tasks in active_tasks.items():
            if queue_name not in worker_name:
                continue
            for task in tasks:
                task_id = task.get("id")
                if not task_id:
                    continue
                try:
                    celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')
                    revoked_active.append(task_id)
                    logger.info(f"Terminated active task {task_id} on {worker_name}")
                except Exception as e:
                    errors.append({"task_id": task_id, "error": str(e)})
                    logger.error(f"Failed to terminate active task {task_id}: {e}")

    total_revoked = len(revoked_reserved) + len(revoked_active)
    return {
        "status": "force_paused",
        "message": (
            f"Force paused '{queue_name}': "
            f"{len(revoked_reserved)} reserved + {len(revoked_active)} active tasks revoked"
        ),
        "pause_result": pause_result,
        "revoked_reserved": revoked_reserved,
        "revoked_active": revoked_active,
        "total_revoked": total_revoked,
        "errors": errors if errors else None
    }


def get_pause_status(queue_name: str) -> Dict:
    """Check if a queue is paused (no workers consuming it)."""
    consumers = _get_workers_for_queue(queue_name)
    pause_state = _load_pause_state(queue_name)

    # Queue is paused if: has pause state AND no consumers
    is_paused = len(consumers) == 0 and pause_state is not None

    # Get queue length from Flower
    queue_length = 0
    queue_data = _flower_get("/api/queues/length")
    if queue_data and "active_queues" in queue_data:
        for q in queue_data["active_queues"]:
            if q.get("name") == queue_name:
                queue_length = q.get("messages", 0)
                break

    return {
        "queue_name": queue_name,
        "is_paused": is_paused,
        "consumers": consumers,
        "consumer_count": len(consumers),
        "queue_length": queue_length,
        "pause_state": pause_state,
        "source": "flower"
    }


# ==============================================================================
# State persistence (for pause/resume)
# ==============================================================================

def _save_pause_state(queue_name: str, num_workers: int, concurrency: int) -> None:
    state_file = os.path.join(STATE_DIR, f"{queue_name}_pause_state.json")
    state = {
        "queue_name": queue_name,
        "num_workers": num_workers,
        "concurrency": concurrency,
        "paused_at": datetime.now().isoformat()
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved pause state for {queue_name}: {state}")


def _load_pause_state(queue_name: str) -> Optional[Dict]:
    state_file = os.path.join(STATE_DIR, f"{queue_name}_pause_state.json")
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading pause state: {e}")
        return None


def _delete_pause_state(queue_name: str) -> None:
    state_file = os.path.join(STATE_DIR, f"{queue_name}_pause_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        logger.info(f"Deleted pause state for {queue_name}")


def get_presets() -> Dict:
    """Get available worker presets."""
    return WORKER_PRESETS
