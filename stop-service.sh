#!/bin/bash
# stop-services.sh - Stops all non-Dockerized application components.

echo "========================================"
echo "  Stopping OCR Pipeline Services"
echo "========================================"

# --- Gracefully stop Celery workers ---
# The 'pkill -f' command finds processes whose command line matches the string.
# This is a reliable way to target the specific Celery workers for this app.
echo "--> Sending TERM signal to Celery workers..."
pkill -f "celery -A app.celery_app worker"
# pkill -f "celery -A app.celery_app flower --port=5555"
if [ $? -eq 0 ]; then
    echo "    Signal sent. Waiting for workers to shut down gracefully..."
    sleep 5 # Give workers a few seconds to finish current tasks if possible
else
    echo "    No running Celery workers found."
fi

# --- Stop the FastAPI/Uvicorn server ---
echo "--> Sending TERM signal to FastAPI (Uvicorn) server..."
pkill -f "uvicorn app.main_celery:app"
if [ $? -eq 0 ]; then
    echo "    Signal sent."
else
    echo "    FastAPI server not found."
fi

# --- Stop the Detectron2 Microservice ---
echo "--> Sending TERM signal to Detectron2 service..."
pkill -f "uvicorn app.detectron2_api:app"
if [ $? -eq 0 ]; then
    echo "    Signal sent."
else
    echo "    Detectron2 service not found."
fi

# --- Stop the Detectron2 Phieutin Microservice ---
echo "--> Sending TERM signal to Detectron2 Phieutin service..."
pkill -f "uvicorn app.detectron2_phieutin_api:app"
if [ $? -eq 0 ]; then
    echo "    Signal sent."
else
    echo "    Detectron2 Phieutin service not found."
fi

# --- Stop the ProtonX Spellfix Microservice ---
echo "--> Sending TERM signal to Spellfix service..."
pkill -f "uvicorn app.spellfix_api:app"
if [ $? -eq 0 ]; then
    echo "    Signal sent."
else
    echo "    Spellfix service not found."
fi


echo ""
echo "----------------------------------------"
echo "All application services have been sent the stop signal."
echo ""
echo "Note: This script does NOT stop the Dockerized services (RabbitMQ & PostgreSQL)."
echo "To stop them, run the following command in your project directory:"
echo "  docker-compose down"
echo "----------------------------------------"
