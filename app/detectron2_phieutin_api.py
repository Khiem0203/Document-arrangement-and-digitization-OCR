from fastapi import FastAPI, UploadFile, File, HTTPException
# Note: detectron_process is defined below in this file, not imported
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from pydantic import BaseModel, Field
import os
import cv2
import base64
import numpy as np
_predictor = None

class ImageDetect(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")
    image_path: str = Field(..., description="Path to the image file")

app=FastAPI()
thing_classes = ["hopso", "hososo", "key", "quanlybienban", "soto", "thoigianbaoquan", "thoigiantailieu", "tieudehoso"]
def get_detectron_predictor():
    """
    Initializes and returns a singleton Detectron2 predictor instance.
    This function ensures the model is loaded only once per worker process.
    """
    global _predictor
    if _predictor is None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.getenv("DETECTRON_PHIEUTIN", "/path/to/OCR_CELERY/detectron_weights/phieutin_model/model_final.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.DEVICE = "cuda"
        _predictor = DefaultPredictor(cfg)
        MetadataCatalog.get("phieutin_model_metadata").set(thing_classes=thing_classes)
        custom_metadata = MetadataCatalog.get("phieutin_model_metadata")
    return _predictor

def detectron_process(image_base64, image_path) -> list:
    """
    Processes an image using the lazily-initialized Detectron2 predictor.
    MUST match the original detectron.py logic exactly!
    """
    # Decode image from base64
    img_binary = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_binary, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[Skipped] Cannot decode image from base64")
        return []

    # Get predictor (singleton)
    predictor = get_detectron_predictor()

    # Run inference
    outputs = predictor(img)
    instances = outputs["instances"]

    # Build result list - EXACTLY like detectron.py original
    result = []
    for box, label, confidence_score in zip(
        instances.pred_boxes.tensor.cpu().numpy(),
        instances.pred_classes.cpu().numpy(),
        instances.scores.cpu().numpy()
    ):
        x1, y1, x2, y2 = box
        class_name = MetadataCatalog.get("phieutin_model_metadata").thing_classes[label]
        # Return format: [x1, y1, x2, y2, class_name, confidence_score]
        # NO extra labels like "MAIN/COPY", "NUMBERPAGE", "SECURITY" (those are handled in flows.py)
        result.append([int(x1), int(y1), int(x2), int(y2), class_name, float(confidence_score)])

    return result


@app.post("/detectron2_phieutin/detect")
def detect_image(input: ImageDetect):
    try:
        detections = detectron_process(image_base64=input.image_base64, image_path=input.image_path)
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
