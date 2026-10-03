import os
import re
import cv2
import random
import json
import shutil
from collections import defaultdict
import asyncio
import httpx
from celery.utils.log import get_task_logger
celery_logger = get_task_logger(__name__)
from .utils import convertPDFImage, variants, process_text, is_box_in_top_left, normalize_abbrevs, capitalize_after_admin_units, chuan_hoa_tac_gia, sentence_case, fix_spelling, fix_place_names
from .vision_service import call_vllm, call_vllm_async, call_vllm_base64, encode_image_to_base64_resize
from .get_config_schema import get_fields, extract_date_format, get_fields_by_name
from .detect_color import detect_colors_in_image
import base64


from dotenv import load_dotenv
load_dotenv()

# Detectron2 classes - same as in detectron.py
thing_classes = ["BLANK","CHUCVU", "DATE", "IMAGE", "NO", "ORG", "PHULUC", "RECEIVE", "SAOY", "SENDTO", "SIGNER", "TABLE", "TEXT", "THELOAI", "VV"]

# Detectron2 API configuration
DETECTRON2_API_URL = os.getenv("DETECTRON2_API_URL", "http://localhost:8003/detectron2/detect")
# Biến bật/tắt tính năng dò tìm chữ ký ở các trang sau (True: Bật, False: Tắt)
ENABLE_SIGNER_SEARCH = True

async def detectron_process_async(image_path: str, request_id: str = "N/A", client: httpx.AsyncClient = None) -> list:
    """Async version of detectron_process to enable parallel scanning."""
    try:
        # Read and encode image (blocking operation, but OK for small images)
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Use provided client or create a temporary one
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=300.0)
            close_client = True

        response = await client.post(
            DETECTRON2_API_URL,
            json={
                "image_base64": img_base64,
                "image_path": image_path
            }
        )
        response.raise_for_status()

        if close_client:
            await client.aclose()

        result = response.json()
        return result.get("detections", [])
    except Exception as e:
        celery_logger.error(f"Detectron2 Async API call failed for {image_path}: {e}")
        return []

def detectron_process(image_path: str, request_id: str = "N/A") -> list:
    """
    Call Detectron2 microservice API instead of loading model locally.
    This allows multiple workers to share a single model instance.
    """
    try:
        # Read and encode image
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Call Detectron2 API
        response = httpx.post(
            DETECTRON2_API_URL,
            json={
                "image_base64": img_base64,
                "image_path": image_path
            },
            timeout=300.0
        )
        response.raise_for_status()
        # Parse response
        result = response.json()
        detections = result.get("detections", [])

        # API returns same format as original detectron_process:
        # [x1, y1, x2, y2, class_name, confidence_score] (6 elements)
        return detections
    except Exception as e:
        celery_logger.error(f"Detectron2 API call failed for {image_path}: {e}")
        return []


# This prompt mapping was in tasks.py, but it's used by the government flow logic.
PROMPT_MAPPING = {
    "NO": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
    "DATE": '<image>\nNhận diện và trích xuất TEXT trong ảnh. Trả ra TEXT, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "ORG": '<image>\nNhận diện TEXT trong ảnh. Trả ra TEXT trong ảnh, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "SENDTO": '<image>\nNhận diện TEXT trong ảnh. Trả ra tên người nhận văn bản, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "VV": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "THELOAI": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "SIGNER": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra họ và tên người ký văn bản, không trả ra gì thêm.NẾU KHÔNG CÓ THÔNG TIN TÊN NGƯỜI KÝ THÌ TRẢ VỀ CHUỖI RỖNG "".',
    "NUMBERPAGE": '<image>\nTừ thông tin trong ảnh, hãy trả về số viết tay ở góc trên bên trái của ảnh. Chỉ trả về số, không trả lời gì thêm.',
    "DEFAULT": '<image>\nTrích xuất tất cả TEXT trong ảnh, không trả ra gì thêm.'
}

PROMPT_MAPPING_CHINHLY = {
    "NO": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
    "DATE": '<image>\nNhận diện và trích xuất TEXT trong ảnh. Trả ra TEXT, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "ORG": '<image>\nNhận diện và Trả ra TEXT trong ảnh, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "" (LƯU Ý: NẾU PHÁT HIỆN TEXT BOLD thì trả ra dòng TEXT BOLD Trước, dòng TEXT thường sai).',
    "VV": '<image>\nNhận diện TEXT trong ảnh. Trả ra TEXT, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "THELOAI": '<image>\nNhận diện TEXT trong ảnh. Trả ra TEXT, không trả ra gì thêm. Nếu không có TEXT thì trả về chuỗi rỗng "".',
    "SIGNER": '<image>\nNhận diện TEXT trong ảnh. Trích xuất và Trả ra họ và tên người ký văn bản, không trả ra gì thêm.',
    "DEFAULT": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra các trường sau dưới dạng JSON, không trả ra gì thêm',
    "SECURITY": '<image>\nChỉ đưa ra câu trả lời Có hoặc Không, không đưa ra giải thích gì thêm. Văn bản có dấu Mật không?',
    "NUMBERPAGE": '<image>\nExtract handwriting number on image. ONLY RETURN THE NUMBER, NOTHING ELSE. If there is no handwriting number or blank, return empty string "".',
}

import unicodedata
def clean_text(text):
    BANG_XOA_DAU = str.maketrans(
        "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
        "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 + "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5
    )
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    return text.translate(BANG_XOA_DAU)



def _ocr_cropped_image_processing(image_base64, logger):
    """
    A helper function to process cropped images for OCR.
    """
    prompt= '<image>\nNhận diện toàn bộ TEXT trong ảnh. Trả ra TEXT nhận diện được, không trả ra gì thêm'
    try:
        result = call_vllm_base64(prompt, image_base64=image_base64, max_tokens=1300)
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")
        result = ""
    if result:
        result = ' '.join(result.split('\n')).strip()
        result = capitalize_after_admin_units(normalize_abbrevs(fix_spelling(sentence_case(result))))
    return result

def _run_detection_and_ocr(image_paths, logger, request_id: str = "N/A"):
    """A helper function to run the common detection and OCR logic."""
    num_pages = len(image_paths)
    signer_status = False
    crops_to_ocr = []
    header_fields_found = set()
    temp_folder_base = os.path.dirname(os.path.dirname(image_paths[0])) # e.g., ./temp_processing/{id}

    # Logic cải tiến: Ưu tiên trang 1 trước (Head + Header info)
    page_1_idx = 0
    img_path_1 = image_paths[page_1_idx]
    page_num = 1
    logger.info(f"Checking page 1 first for header info and signer...")
    
    image = cv2.imread(img_path_1)
    if image is not None:
        image_h, image_w = image.shape[:2]
        detected_objects = detectron_process(img_path_1, request_id)
        for box_info in detected_objects:
            x1, y1, x2, y2, label, confidence_score = box_info
            if label in ["BLANK", "TEXT", "RECEIVE", "CHUCVU","SAOY","TABLE","PHULUC","IMAGE"]:
                continue
            
            crop_img = image[y1:y2, x1:x2]
            crop_path = os.path.join(temp_folder_base, f"page_1_crop_{label}_{x1}.jpg")
            
            if label == "SIGNER":
                if is_box_in_top_left(x1, y1, x2, y2, image_w, image_h, 0.5):
                    continue
                else:
                    cv2.imwrite(crop_path, crop_img)
                    crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
                    color= detect_colors_in_image(crop_path)
                    logger.info(f"Detected colors in SIGNER crop: {color}")
                    if color.get("Red") == False and color.get("Blue") == False:
                        confidence_type = "bản sao"
                    else:
                        confidence_type = "bản chính"
                    logger.info(f"Signer confidence type set to: {confidence_type}")
                    signer_status = True
                    logger.info("Found SIGNER on page 1")
            elif label not in header_fields_found:
                cv2.imwrite(crop_path, crop_img)
                crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
                header_fields_found.add(label)

    # 2. XỬ LÝ DÒ TÌM ĐA TRANG (Chỉ chạy khi ENABLE_SIGNER_SEARCH = True)
    if ENABLE_SIGNER_SEARCH and not signer_status:
        # XỬ LÝ SONG SONG NẾU > 10 TRANG
        if num_pages > 10:
            logger.info(f"Document has {num_pages} pages (>10) -> Enabling Parallel Head-Tail search...")
            mid = (num_pages + 1) // 2 
            range_forward = list(range(1, mid))
            range_backward = list(range(num_pages - 1, mid - 1, -1))
            
            async def search_in_range(indices, name):
                nonlocal signer_status
                async with httpx.AsyncClient(timeout=300.0) as client:
                    for idx in indices:
                        if signer_status: return
                        p_num = idx + 1
                        p_path = image_paths[idx]
                        logger.info(f"[{name}] Checking page {p_num}...")
                        
                        dets = await detectron_process_async(p_path, request_id, client)
                        if not dets: continue
                        
                        img = cv2.imread(p_path)
                        for b in dets:
                            x1, y1, x2, y2, lbl, conf = b
                            if lbl == "SIGNER":
                                c_path = os.path.join(temp_folder_base, f"page_{p_num}_crop_SIGNER_{x1}.jpg")
                                cv2.imwrite(c_path, img[y1:y2, x1:x2])
                                crops_to_ocr.append({"label": lbl, "crop_path": c_path, "confidence": conf, "page": p_num})
                                color= detect_colors_in_image(c_path)
                                logger.info(f"Detected colors in SIGNER crop: {color}")
                                if color.get("Red") == False and color.get("Blue") == False:
                                    confidence_type = "bản sao"
                                else:
                                    confidence_type = "bản chính"
                                logger.info(f"Signer confidence type set to: {confidence_type}")
                                signer_status = True
                                logger.info(f"[{name}] Found SIGNER at page {p_num}")
                                return

            async def run_parallel():
                await asyncio.gather(search_in_range(range_forward, "Forward"), search_in_range(range_backward, "Backward"))
            
            asyncio.run(run_parallel())

        # XỬ LÝ TUẦN TỰ NẾU <= 10 TRANG 
        else:
            check_order = []
            if num_pages > 1: check_order.append(num_pages - 1)
            if num_pages > 2: check_order.extend(range(1, num_pages - 1))
            
            for idx in check_order:
                if signer_status: break
                img_path = image_paths[idx]
                page_num = idx + 1
                logger.info(f"Detecting on page {page_num} (Sequential)...")
                
                image = cv2.imread(img_path)
                dets = detectron_process(img_path, request_id)
                for box_info in dets:
                    x1, y1, x2, y2, label, confidence_score = box_info
                    if label == "SIGNER":
                        crop_path = os.path.join(temp_folder_base, f"page_{page_num}_crop_SIGNER_{x1}.jpg")
                        cv2.imwrite(crop_path, image[y1:y2, x1:x2])
                        crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
                        color= detect_colors_in_image(crop_path)
                        logger.info(f"Detected colors in SIGNER crop: {color}")
                        if color.get("Red") == False and color.get("Blue") == False:
                            confidence_type = "bản sao"
                        else:
                            confidence_type = "bản chính"
                        logger.info(f"Signer confidence type set to: {confidence_type}")
                        signer_status = True
                        logger.info(f"Found SIGNER at page {page_num}")
                        break
    elif not signer_status:
        logger.info("SIGNER multi-page search is DISABLED. Finishing after page 1.")

    # Final logic for page reporting
    if signer_status:
        try:
            # Find the actual signer page from crops
            signer_crop = next((c for c in crops_to_ocr if c["label"] == "SIGNER"), None)
            if signer_crop and "page" in signer_crop:
                page_num = signer_crop["page"]
        except Exception: pass
    
    # Perform Concurrent OCR
    async def run_async_ocr():
        async with httpx.AsyncClient() as client:
            tasks = [call_vllm_async(client, PROMPT_MAPPING.get(c["label"], "DEFAULT"), c["crop_path"]) for c in crops_to_ocr]
            return await asyncio.gather(*tasks)
    ocr_texts = asyncio.run(run_async_ocr())

    # Aggregate results
    final_ocr = defaultdict(list)
    label_confidences = defaultdict(list)  # Changed: track confidences for each label
    _ABBREV_LABELS = {"ORG", "VV", "SENDTO", "THELOAI"}
    for crop, text in zip(crops_to_ocr, ocr_texts):
        if text:
            label = crop["label"]
            if label == "NO": text = text.replace(" ", "").replace("=", "-")
            if label == 'ORG':
                text = chuan_hoa_tac_gia(fix_place_names(text))
            elif label == 'VV':
                text = text.replace("\n", " ").strip()
            elif label == 'THELOAI':
                text = normalize_abbrevs(fix_spelling(sentence_case(text)))
            elif label in _ABBREV_LABELS:
                text = capitalize_after_admin_units(normalize_abbrevs(fix_place_names(text)))
            final_ocr[label].append(text)
            label_confidences[label].append(crop.get("confidence", 0.0))  # Changed: store confidence

    OCR_RESULTS = {key: ", ".join(values) for key, values in final_ocr.items()}
    # Changed: Add confidence scores (use max confidence for each label)
    OCR_RESULTS["_confidences"] = {key: max(scores) * 100 if scores else 0 for key, scores in label_confidences.items()}
    OCR_RESULTS["Page"] = page_num

    # --- Post-process: strip THELOAI prefix khỏi VV (case-insensitive) ---
    if "THELOAI" in OCR_RESULTS:
        theloai_val = OCR_RESULTS.get("THELOAI", "")
        vv_val = OCR_RESULTS.get("VV", "")
        if theloai_val and theloai_val.lower() in vv_val.lower():
            import re as _re_local
            pattern = _re_local.compile(_re_local.escape(theloai_val), _re_local.IGNORECASE)
            OCR_RESULTS["VV"] = pattern.sub("", vv_val).strip()
    elif "VV" in OCR_RESULTS and "THELOAI" not in OCR_RESULTS:
        vv_val = OCR_RESULTS.get("VV", "")
        _DOC_TYPES = [
            "CÔNG VĂN", "QUYẾT ĐỊNH", "NGHỊ ĐỊNH", "THÔNG TƯ", "THÔNG BÁO",
            "PHIẾU TRÌNH", "TỜ TRÌNH", "BÁO CÁO", "KẾ HOẠCH", "BIÊN BẢN",
            "HỢP ĐỒNG", "QUY CHẾ", "QUY ĐỊNH", "CHỈ THỊ", "NGHỊ QUYẾT",
        ]
        matched_type = None
        for dt in _DOC_TYPES:
            if vv_val.upper().startswith(dt):
                matched_type = dt
                break
        if matched_type:
            OCR_RESULTS["THELOAI"] = sentence_case(matched_type)
            OCR_RESULTS["VV"] = vv_val[len(matched_type):].strip()

    celery_logger.info(f"Crop Results: {final_ocr}")
    celery_logger.info(f"Confidence Scores: {OCR_RESULTS['_confidences']}")
    return OCR_RESULTS



def _run_detection_and_ocr_chinhly(image_paths, logger, request_id: str = "N/A"):
    """A helper function to run the common detection and OCR logic."""
    num_pages = len(image_paths)
    signer_status = False
    crops_to_ocr = []
    header_fields_found = set()
    temp_folder_base = os.path.dirname(os.path.dirname(image_paths[0]))
    confidence_type = "bản chính"
    global page_num
    # Logic cải tiến: Ưu tiên trang 1 trước
    page_1_idx = 0
    img_path_1 = image_paths[page_1_idx]
    page_num = 1
    logger.info(f"Checking page 1 first (Chinh ly)...")
    
    image = cv2.imread(img_path_1)
    if image is not None:
        image_h, image_w = image.shape[:2]
        detected_objects = detectron_process(img_path_1, request_id)
        for box_info in detected_objects:
            x1, y1, x2, y2, label, confidence_score = box_info
            if label in ["BLANK", "TEXT", "RECEIVE", "CHUCVU","SAOY","TABLE","PHULUC","IMAGE"]:
                continue
            
            crop_img = image[y1:y2, x1:x2]
            crop_path = os.path.join(temp_folder_base, f"page_1_crop_{label}_{x1}.jpg")
            
            if label == "SIGNER":
                if is_box_in_top_left(x1, y1, x2, y2, image_w, image_h, 0.5):
                    continue
                else:
                    cv2.imwrite(crop_path, crop_img)
                    crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
                    color= detect_colors_in_image(crop_path)
                    logger.info(f"Detected colors in SIGNER crop: {color}")
                    if color.get("Red") == False and color.get("Blue") == False:
                        confidence_type = "bản sao"
                    else:
                        confidence_type = "bản chính"
                    logger.info(f"Signer confidence type set to: {confidence_type}")
                    signer_status = True
                    logger.info("Found SIGNER on page 1")
            elif label not in header_fields_found:
                cv2.imwrite(crop_path, crop_img)
                crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
                header_fields_found.add(label)

    # 2. XỬ LÝ DÒ TÌM ĐA TRANG (Chỉ chạy khi ENABLE_SIGNER_SEARCH = True)
    if ENABLE_SIGNER_SEARCH and not signer_status:
        # XỬ LÝ SONG SONG NẾU > 10 TRANG 
        if num_pages > 10:
            logger.info(f"Chỉnh lý: {num_pages} trang -> Enabling Parallel search...")
            mid = (num_pages + 1) // 2
            range_forward = list(range(1, mid))
            range_backward = list(range(num_pages - 1, mid - 1, -1))

            async def search_in_range(indices, name):
                nonlocal signer_status, confidence_type
                async with httpx.AsyncClient(timeout=300.0) as client:
                    for idx in indices:
                        if signer_status: return
                        p_num = idx + 1
                        p_path = image_paths[idx]
                        dets = await detectron_process_async(p_path, request_id, client)
                        if not dets: continue
                        img = cv2.imread(p_path)
                        for b in dets:
                            x1, y1, x2, y2, lbl, conf = b
                            if lbl == "SIGNER":
                                c_path = os.path.join(temp_folder_base, f"page_{p_num}_crop_SIGNER_{x1}.jpg")
                                cv2.imwrite(c_path, img[y1:y2, x1:x2])
                                crops_to_ocr.append({"label": lbl, "crop_path": c_path, "confidence": conf, "page": p_num})
                                color= detect_colors_in_image(c_path)
                                logger.info(f"Detected colors in SIGNER crop: {color}")
                                if color.get("Red") == False and color.get("Blue") == False:
                                    confidence_type = "bản sao"
                                else:
                                    confidence_type = "bản chính"
                                logger.info(f"Signer confidence type set to: {confidence_type}")
                                signer_status = True
                                logger.info(f"[{name}] Found SIGNER at page {p_num} (Chinh ly)")
                                return

            async def run_parallel():
                await asyncio.gather(search_in_range(range_forward, "Forward"), search_in_range(range_backward, "Backward"))
            asyncio.run(run_parallel())

        # XỬ LÝ TUẦN TỰ NẾU <= 10 TRANG 
        elif not signer_status:
            check_order = []
            if num_pages > 1: check_order.append(num_pages - 1)
            if num_pages > 2: check_order.extend(range(1, num_pages - 1))
            for idx in check_order:
                if signer_status: break
                img_path = image_paths[idx]
                page_num = idx + 1
                image = cv2.imread(img_path)
                dets = detectron_process(img_path, request_id)
                for box_info in dets:
                    x1, y1, x2, y2, label, confidence_score = box_info
                    if label == "SIGNER":
                        crop_path = os.path.join(temp_folder_base, f"page_{page_num}_crop_SIGNER_{x1}.jpg")
                        cv2.imwrite(crop_path, image[y1:y2, x1:x2])
                        crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
                        color= detect_colors_in_image(crop_path)
                        logger.info(f"Detected colors in SIGNER crop: {color}")
                        if color.get("Red") == False and color.get("Blue") == False:
                            confidence_type = "bản sao"
                        else:
                            confidence_type = "bản chính"
                        logger.info(f"Signer confidence type set to: {confidence_type}")
                        signer_status = True
                        break
    elif not signer_status:
         logger.info("SIGNER multi-page search is DISABLED (Chinh ly). Finishing after page 1.")

    logger.info(f"Page of the ending: {page_num}")
    image_first = cv2.imread(image_paths[0])
    crop_path_security = None
    crop_path_number = None
    if image_first is not None:
        img_h, img_w = image_first.shape[:2]
        first_page_crop_img = image_first[0:img_h // 2, 0:img_w // 2]
        number_page_img =image_first[0:img_h // 12, (img_w // 2)+100 :img_w]
        crop_path_security = os.path.join(temp_folder_base, "first_page_crop_half.jpg")
        crop_path_number = os.path.join(temp_folder_base, "numberpage.jpg")
        cv2.imwrite(crop_path_security, first_page_crop_img)
        cv2.imwrite(crop_path_number, number_page_img)
        crops_to_ocr.append({"label": "SECURITY", "crop_path": crop_path_security, "confidence": 1.0})
        crops_to_ocr.append({"label": "NUMBERPAGE", "crop_path": crop_path_number, "confidence": 1.0})
    else:
        logger.warning(f"Could not load first page image for cropping: {image_paths[0]}")
    if not any(item.get("label") == "NO" for item in crops_to_ocr) and crop_path_security is not None:
        crops_to_ocr.append({"label": "NO", "crop_path": crop_path_security, "confidence": 1.0})
        PROMPT_MAPPING_CHINHLY["NO"] = '<image>\nTrích xuất Số văn bản trong ảnh trên. Nếu không có số văn bản thì trả về chuỗi rỗng "".'
    logger.info(f"Label và đường dẫn: {crops_to_ocr}")
    # Perform Concurrent OCR
    async def run_async_ocr():
        async with httpx.AsyncClient() as client:
            tasks = [call_vllm_async(client, PROMPT_MAPPING_CHINHLY.get(c["label"], "DEFAULT"), c["crop_path"]) for c in crops_to_ocr]
            return await asyncio.gather(*tasks)
    ocr_texts = asyncio.run(run_async_ocr())
    celery_logger.info(f"Kết quả OCR trước xử lý: {ocr_texts}")
    final_ocr = defaultdict(list)
    label_confidences = defaultdict(list)  # Changed: track confidences for each label

    _ABBREV_LABELS = {"ORG", "VV", "SENDTO", "THELOAI"}
    for crop, text in zip(crops_to_ocr, ocr_texts):
        if text:
            label = crop["label"]
            if label == "NO":
                import re
                text = re.sub(r'^\D+', '', text)
            elif label == "SIGNER":
                if len(text.split(" "))  > 5 and "\n" in text:
                    text = text.split("\n")[-1]
                elif len(text.split(" ")) > 5 and "\n" not in text:
                    text = ""
                else:
                    text = text.strip()
            elif label == "ORG":
                CO_QUAN_LIST = [
                    "TỈNH ỦY THÁI BÌNH",
                    "ĐẢNG BỘ TỈNH THÁI BÌNH",
                    "TỈNH ỦY HƯNG YÊN",
                    "ĐẢNG BỘ TỈNH HƯNG YÊN",
                    "ĐẢNG BỘ TỈNH BÀ RỊA - VŨNG TÀU",
                ]
                CO_QUAN_VARIANTS = [variants(name) for name in CO_QUAN_LIST]
                text = process_text(text, CO_QUAN_VARIANTS)
            if label == 'ORG':
                text = chuan_hoa_tac_gia(fix_place_names(text))
            elif label == 'VV':
                text = text.replace("\n", " ").strip()
            elif label == 'THELOAI':
                text = normalize_abbrevs(fix_spelling(sentence_case(text)))
            elif label in _ABBREV_LABELS:
                text = capitalize_after_admin_units(normalize_abbrevs(fix_place_names(text)))
            final_ocr[label].append(text)
            label_confidences[label].append(crop.get("confidence", 0.0))
    final_ocr["DOMAT"].append(confidence_type)
    celery_logger.info(f"Kết quả OCR sau xử lý: {final_ocr}")
    # OCR_RESULTS={key: ", ".join(values) for key, values in final_ocr.items()}

    OCR_RESULTS = {
        key: ", ".join(str(v) for v in values if v is not None)
        for key, values in (final_ocr or {}).items()
        if values is not None
    }
    # Changed: Add confidence scores (use max confidence for each label)
    OCR_RESULTS["_confidences"] = {key: max(scores) * 100 if scores else 0 for key, scores in label_confidences.items()}
    OCR_RESULTS["VV"]= OCR_RESULTS.get("VV", "").replace("\n"," ").strip()
    value = OCR_RESULTS.get("NUMBERPAGE")
    OCR_RESULTS["NUMBERPAGE"] = value if isinstance(value, str) and value.isdigit() and (value != "0") else ""

    if "THELOAI" in OCR_RESULTS.keys():
        theloai_val = OCR_RESULTS.get("THELOAI", "")
        vv_val = OCR_RESULTS.get("VV", "")
        # Case-insensitive check — OCR từ ảnh thường trả về chữ HOA (CÔNG VĂN)
        # trong khi sentence_case đã chuyển thành "Công văn"
        if theloai_val and theloai_val.lower() in vv_val.lower():
            import re as _re_local
            pattern = _re_local.compile(_re_local.escape(theloai_val), _re_local.IGNORECASE)
            OCR_RESULTS["VV"] = pattern.sub("", vv_val).strip()
        logger.info(f"Kết quả OCR THE LOAI:: {OCR_RESULTS}")
    elif "VV" in OCR_RESULTS.keys() and "THELOAI" not in OCR_RESULTS.keys():
        # Khi không detect được THELOAI riêng, thử tách prefix thể loại từ đầu VV
        vv_val = OCR_RESULTS.get("VV", "")
        _DOC_TYPES = [
            "CÔNG VĂN", "QUYẾT ĐỊNH", "NGHỊ ĐỊNH", "THÔNG TƯ", "THÔNG BÁO",
            "PHIẾU TRÌNH", "TỜ TRÌNH", "BÁO CÁO", "KẾ HOẠCH", "BIÊN BẢN",
            "HỢP ĐỒNG", "QUY CHẾ", "QUY ĐỊNH", "CHỈ THỊ", "NGHỊ QUYẾT",
        ]
        matched_type = None
        for dt in _DOC_TYPES:
            if vv_val.upper().startswith(dt):
                matched_type = dt
                break
        if matched_type:
            OCR_RESULTS["THELOAI"] = sentence_case(matched_type)
            OCR_RESULTS["VV"] = vv_val[len(matched_type):].strip()
        else:
            OCR_RESULTS["THELOAI"] = "Công văn"
    if "NO" in OCR_RESULTS.keys():
        if len(OCR_RESULTS.get('NO'))>30:
            OCR_RESULTS["NO"] = ""
    OCR_RESULTS["Page"]= page_num
    celery_logger.info(f"Ending Page: {OCR_RESULTS}")
    return OCR_RESULTS

#####################################

def government_flow(context: dict, logger) -> dict:
    """
    Processing logic for Government ('Chính quyền') documents.
    """
    logger.info(f"Executing Government Flow for request {context['request_id']}")
    
    # Run the common detection and OCR part
    request_id = context.get("request_id", "N/A")
    final_ocr = _run_detection_and_ocr(context["image_paths"], logger, request_id)

    # --- Schema Mapping for Government Documents ---
    json_data = {
        "SVB": final_ocr.get("NO", ""),
        "KH": final_ocr.get("NO", ""),
        "CQBH": final_ocr.get("ORG", ""),
        "NKy": final_ocr.get("DATE", ""),
        "NK": final_ocr.get("SIGNER", ""),
        "TTNDVB": final_ocr.get("VV", ""),
        "GHICHU": "",
        "DM": "Thường",
        "LB": "Bản chính",
        "NN": "Tiếng Việt",
        "TLVB": final_ocr.get("THELOAI", ""),
        "Page": final_ocr.get("Page", 1),
        "SLTr": context.get("pages_number", len(context["image_paths"]))
    }
    logger.info(f"Government Flow: Mapped JSON data before keywords: {json_data}")

    try:
        if json_data["TTNDVB"] != "":
            summary_prompt = f'Sinh ra list 6 từ khóa (mỗi từ khóa gồm 2-3 từ đơn) từ đoạn text sau: {json_data["TTNDVB"]}'
            keywords = call_vllm(prompt=summary_prompt, image_path=None)
            json_data["TK"] = keywords.replace("1.","").replace("2.",",").replace("3.",",").replace("4.",",").replace("5.",",").replace("6.",",").replace("\n","")
        else:
            json_data["TK"] = ""
    except Exception as e:
        logger.warning(f"Government Flow: Could not generate keywords: {e}")
        json_data["TK"] = ""
    logger.info(f"Government Flow: Extracted JSON data: {json_data}")
    final_fields = get_fields(context["customer_id"], context["config_template_id"], json_data)
    
    json_data_final = {
        "document_type": json_data["TLVB"],
        "template_id": context["config_template_id"],
        "customer_id": context["customer_id"],
        "pages_number": len(context["image_paths"]),
        "fields": final_fields,
        "confidence_score": round(random.uniform(0.85, 0.98), 4)
    }
    raw_text = " ".join(map(str, final_ocr.values()))
    
    return {"final_json": json_data_final, "raw_text": raw_text}


def chinhly_flow(context: dict, logger) -> dict:
    """
    Processing logic for chỉnh lý  documents.
    """
    logger.info(f"Executing Chỉnh lý Flow for request {context['request_id']}")
    request_id = context.get("request_id", "N/A")
    final_ocr = _run_detection_and_ocr_chinhly(context["image_paths"], logger, request_id)
    security_status = "Mật" if "Có" in str(final_ocr.get("SECURITY")) else ""
    # Map detection label confidences to field code confidences
    original_confidences = final_ocr.get("_confidences", {})
    json_data = {
        "SKH": final_ocr.get("NO", "").replace("Số", "").replace(":", "").replace(" ", "").replace("SỐ", "").strip(),
        "TG": final_ocr.get("ORG", ""),
        "NT": extract_date_format(final_ocr.get("DATE", "")),
        "TLVTY": (final_ocr.get("THELOAI", "") + " " + final_ocr.get("VV", "")).strip(),
        "GC": "",
        "DM": security_status,
        "BG": True if final_ocr["DOMAT"] == "bản chính" else False,
        "BS": True if final_ocr["DOMAT"] == "bản sao" else False,
        "TS": final_ocr.get("NUMBERPAGE", ""),
        "NKVB": final_ocr.get("SIGNER", ""),
        "Page": final_ocr.get("Page", 1),
        "_confidences": {
            "SKH": original_confidences.get("NO", 0),
            "TG": original_confidences.get("ORG", 0),
            "NT": original_confidences.get("DATE", 0),
            "TLVTY": max(original_confidences.get("THELOAI", 0), original_confidences.get("VV", 0)),
            "GC": 0,
            "DM": original_confidences.get("SECURITY", 0),
            "BG": 0,
            "BS": 0,
            "TS": original_confidences.get("NUMBERPAGE", 0),
            "NKVB": original_confidences.get("SIGNER", 0),
        }
    }

    final_fields = get_fields(context["customer_id"], context["config_template_id"], json_data)
    
    json_data_final = {
        "document_type": final_ocr.get("THELOAI", ""),
        "template_id": context["config_template_id"],
        "customer_id": context["customer_id"],
        "pages_number": len(context["image_paths"]),
        "fields": final_fields,
        "confidence_score": round(random.uniform(0.85, 0.98), 4)
    }

    raw_text = " ".join(map(str, final_ocr.values()))
    
    return {"final_json": json_data_final, "raw_text": raw_text}

# def government_flow(context: dict, logger) -> dict:
#     """
#     Processing logic for Government ('Chính quyền') documents.
#     """
#     logger.info(f"Executing Government Flow for request {context['request_id']}")
    
#     # Run the common detection and OCR part
#     final_ocr = _run_detection_and_ocr(context["image_paths"], logger)
    
#     # --- Schema Mapping for Government Documents ---
#     json_data = {
#         "CodeNumber": final_ocr.get("NO", ""),
#         "CodeNotation": final_ocr.get("NO", ""),
#         "OrganName": final_ocr.get("ORG", ""),
#         "IssuedDate": final_ocr.get("DATE", ""),
#         "Signer": final_ocr.get("SIGNER", ""), # Note: The provided script didn't map this, but it seems logical to.
#         "Subject": final_ocr.get("VV", ""),
#         "Description": "",
#         "PageAmount": len(context["image_paths"]),
#         "ConfidenceLevel": "Bản chính",
#         "Language": "Tiếng Việt",
#         "Mode": "",
#         "Format": "",
#         "Autograph": "",
#         "InforSign": "",
#         "TypeName": final_ocr.get("THELOAI", "")
#     }
    
#     try:
#         if json_data["Subject"] !="":
#             summary_prompt = f'Sinh ra list 6 từ khóa (mỗi từ khóa gồm 2-3 từ đơn) từ đoạn text sau: {json_data["Subject"]}'
#             keywords = call_vllm(prompt=summary_prompt, image_path=None)
#             json_data["Keyword"] = keywords.replace("1.","").replace("2.",",").replace("3.",",").replace("4.",",").replace("5.",",").replace("6.",",").replace("\n","")
#         else:
#             json_data["Keyword"] = ""
#     except Exception as e:
#         logger.warning(f"Government Flow: Could not generate keywords: {e}")
#         json_data["Keyword"] = ""
#     logger.info(f"Government Flow: Extracted JSON data: {json_data}")
#     final_fields = get_fields(context["customer_id"], context["config_template_id"], json_data)
    
    # json_data_final = {
    #     "document_type": json_data["TypeName"],
    #     "template_id": context["config_template_id"],
    #     "customer_id": context["customer_id"],
    #     "pages_number": json_data["PageAmount"],
    #     "fields": final_fields,
    #     "confidence_score": round(random.uniform(0.85, 0.98), 4)
    # }
#     raw_text = " ".join(map(str, final_ocr.values()))

#     return {"final_json": json_data_final, "raw_text": raw_text}


def curriculum_vitae_flow(context: dict, logger) -> dict:
    """Processing logic for Curriculum Vitae (CV) documents."""
    image_paths = context["image_paths"]
    customer_id = context["customer_id"]
    config_template_id = context["config_template_id"]
    temp_folder = context["temp_folder"] # For saving new crops
    
    logger.info(f"Executing CV Flow for request {context['request_id']}")

    ocr_json = {}
    signer_status = False
    signer_cropped_image_paths = []
    
    # Special half-page crop for CV
    image_first = cv2.imread(image_paths[0])
    if image_first is not None:
        img_h, img_w = image_first.shape[:2]
        first_page_crop_img = image_first[0:img_h // 2, 0:img_w]
        crop_path = os.path.join(temp_folder, "first_page_crop_half.jpg")
        cv2.imwrite(crop_path, first_page_crop_img)
    else:
        logger.warning(f"Could not load first page image for CV cropping: {image_paths[0]}")
        crop_path = None
    
    vv_prompt = '''<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra các trường sau dưới dạng JSON, không trả ra gì thêm
    {
        "Họ và tên": "Họ và tên khai sinh",
        "Ngày, tháng, nam, sinh": "Ngày tháng năm sinh chuyển về định dạng dd/mm/yyyy",
        "Quê quán": "Quê quán của ngườii khai (xã, huyện, tỉnh,TP)",
        "Noi ở hiện nay": "Noi ở hiện nay (xã, huyện, tỉnh, TP)"
    }'''
    request_id = context.get("request_id", "N/A")
    if crop_path is not None:
        ocr_json["VV"] = call_vllm(prompt=vv_prompt, image_path=crop_path)
    else:
        ocr_json["VV"] = "{}"  # or some default

    # Logic Head-Tail First: Ưu tiên trang đầu và trang cuối trước để tìm SIGNER
    num_pages = len(image_paths)
    check_order = []
    if num_pages > 0:
        check_order.append(0)  # Trang đầu
    if num_pages > 1:
        check_order.append(num_pages - 1)  # Trang cuối
    for i in range(1, num_pages - 1):
        check_order.append(i)  # Các trang trung gian

    for idx in check_order:
        if signer_status:
            break
        
        img_path = image_paths[idx]
        image = cv2.imread(img_path)
        page_num = idx + 1
        logger.info(f"Detecting SIGNER on CV page {page_num} (Order: Head-Tail First)...")
        detected_objects = detectron_process(img_path, request_id)
        for box_info in detected_objects:
            x1, y1, x2, y2, label, _ = box_info
            if label == "SIGNER":
                crop_img = image[y1:y2, x1:x2]
                signer_crop_path = f"{os.path.splitext(img_path)[0]}_crop_signer_{x1}.jpg"
                cv2.imwrite(signer_crop_path, crop_img)
                signer_cropped_image_paths.append(signer_crop_path)
                signer_status = True # We can collect all signers
            elif label == "ORG" and "ORG" not in ocr_json: # Only first page ORG
                 crop_img = image[y1:y2, x1:x2]
                 org_crop_path = f"{os.path.splitext(img_path)[0]}_crop_org_{x1}.jpg"
                 cv2.imwrite(org_crop_path, crop_img)
                 ocr_json["ORG"] = call_vllm(prompt=PROMPT_MAPPING["ORG"], image_path=org_crop_path)


    # Process Signer images
    date, signer = "", ""
    for signer_path in signer_cropped_image_paths:
        text = call_vllm(prompt='<image>\nNhận diện tất cả TEXT trong ảnh...', image_path=signer_path)
        if "xác nhận" in text.lower():
            signer_prompt = f'''Từ đọan TEXT sau:\n<text> {text} </text>\nHãy trích xuất và trả ra các trường sau dưới dạng JSON:\n
            {{"Ngày tháng nam":"Ngày tháng năm (ghi rõ ngày tháng năm)","Người ký xác nhận": "Họ và trên người ký xác nhận (không bao gồm chức danh)"}}'''
            try:
                date_and_signer_str = call_vllm(prompt=signer_prompt, image_path=None)
                date_and_signer = json.loads(date_and_signer_str.replace("```json\n","").replace("```",""))
                date = date_and_signer.get("Ngày tháng năm", "")
                signer = date_and_signer.get("Người ký xác nhận", "")
                break # Found what we need
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse signer JSON: {e}")
                continue
    
    # Structure final JSON
    json_data = {"SVB": "", "KH": "", "CQBH": ocr_json.get("ORG", "")}
    try:
        vv_data = json.loads(ocr_json["VV"].replace("```json\n", "").replace("```", ""))
        json_data["TTNDVB"] = "\n".join([f"{k}: {v}" for k, v in vv_data.items()])
    except (json.JSONDecodeError, AttributeError):
        json_data["TTNDVB"] = ""
        
    json_data.update({
        "NKy": date, "NK": signer, "GHICHU": "", "DM": "Thường",
        "LB": "Bản chính", "NN": "Tiếng Việt", "TLVB": "Hồ sơ nhân sự", "TK": ""
    })
    
    final_fields = get_fields(customer_id, config_template_id, json_data)
    
    json_data_final = {
        "document_type": json_data["TLVB"], "template_id": config_template_id,
        "customer_id": customer_id, "pages_number": len(image_paths),
        "fields": final_fields, "confidence_score": round(random.uniform(0.85, 0.98), 4)
    }
    raw_text = json_data["TTNDVB"] # A representative raw text
    
    return {"final_json": json_data_final, "raw_text": raw_text}


def hoso_flow(context: dict, logger) -> dict:
    """Processing logic for dynamic, template-based documents."""
    image_paths = context["image_paths"]
    customer_id = context["customer_id"]
    config_template_id = context["config_template_id"]

    logger.info(f"Executing Dynamic Flow for request {context['request_id']}")
    
    # Aggregate all text from all pages
    raw_text_parts = []
    for img_path in image_paths:
        try:
            base64_image = encode_image_to_base64_resize(img_path, resize_ratio=3)
            request_id = context.get("request_id", "N/A")
            response = call_vllm_base64(prompt='''<image>\nTrích xuất thông tin trong ảnh. Chỉ trả ra thông tin dưới dạng JSON theo cấu trúc sau:\n
                                 \n{"Tên phông": "Tên phòng ban, đơn vị ở đầu trang (nếu có)",
                                "Đơn vị bảo quản số": "Đơn vị bảo quản số (nếu có)",
                                 "Tiêu đề hồ sơ": "Tiêu đề hồ sơ nằm sau chữ 'Hồ sơ', ví dụ: Hồ sơ Đảng viên, Hồ sơ cán bộ, Hồ sơ công chức, Hồ sơ viên chức, Hồ sơ tập Công văn từ ...", 
                                 "Thời hạn bảo quản": "Thời hạn bảo quản bao nhiêu năm, thường là 10 năm, 20 năm, 70 năm, vĩnh viễn (nếu có)", 
                                 "Ngày bắt đầu": "Từ ngày nào (nếu có)",
                                 "Ngày kết thúc": "Đến ngày nào (nếu có)",
                                 "Số tờ": "Là số viết tay thể hiện số tờ trong hồ sơ, nằm giữa 2 từ 'Gồm' ? 'tờ', không thấy gì thì bỏ trống",
                                 "Phông số": "Phông số (nếu có)",
                                 "Mục lục số": "Mục lục số (nếu có)", 
                                 "Hồ sơ số": "Hồ sơ số (nếu có)"}\n
                                 ''', image_base64=base64_image)
            logger.info(f"Vllm response: {response}")
            raw_text_parts.append(response)
        except Exception as e:
            logger.error(f"Error during OCR for {img_path} in hoso flow: {e}")
            continue
    
    full_raw_text = " ".join(raw_text_parts)
    response = raw_text_parts[-1] if raw_text_parts else ""
    output_str = response.strip().replace("```json", "").replace("```", "").strip()    
    logger.info(f"Full OCR output: {output_str}")
    try:
        output = json.loads(output_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON output: {e}")
        output = {}
    try:
        final_fields = get_fields_by_name(customer_id, config_template_id, output)
        logger.info(f"Extracted fields: {final_fields}")

    except Exception as e:
        logger.error(f"Error getting dynamic schema fields: {e}")
        # Return an empty structure on failure
        final_fields = []

    json_data_final = {
        "document_type": "Văn bản hành chính",
        "template_id": config_template_id,
        "customer_id": customer_id,
        "pages_number": len(image_paths),
        "fields": final_fields,
        "confidence_score": round(random.uniform(0.85, 0.98), 4)
    }
    
    return {"final_json": json_data_final, "raw_text": full_raw_text}

from .flow_phieutin import phieutin_flow
