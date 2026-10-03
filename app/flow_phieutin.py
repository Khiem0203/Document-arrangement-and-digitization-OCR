import os
import re
import cv2
import asyncio
import httpx
import base64
from celery.utils.log import get_task_logger
from .vision_service import call_vllm_async

celery_logger = get_task_logger(__name__)

DETECTRON2_PHIEUTIN_API = os.getenv("DETECTRON2_PHIEUTIN_API", "http://localhost:8010/detectron2_phieutin/detect")

PROMPT_MAPPING_PHIEUTIN = {
    "key":            '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
    "hopso":          '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT số nhận diện được, giữ nguyên số 0 ở đầu nếu có, không trả ra gì thêm.',
    "hososo":         '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT số nhận diện được, giữ nguyên số 0 ở đầu nếu có, không trả ra gì thêm.',
    "tieudehoso":     '<image>\nNhận diện toàn bộ TEXT trong ảnh. Trả ra TEXT nhận diện được, giữ nguyên xuống dòng, không trả ra gì thêm.',
    "thoigiantailieu": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
    "soto":           '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra số, không trả ra gì thêm.',
    "thoigianbaoquan": '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
    "quanlybienban":  '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
    "DEFAULT":        '<image>\nNhận diện TEXT trong ảnh. Chỉ trả ra TEXT nhận diện được, không trả ra gì thêm.',
}

_PHIEUTIN_PREFIX_MAP = {
    "key":             ["Key:", "Mã phiếu tin :", "Mã phiếu tin:", r"Mã phiếu tin\s*:", r"mã phiếu tin\s*:", r"phiếu tin\s*:", r"tin\s*:"],
    "hopso":           ["b. Hộp số:", "Hộp số", "b.Hộp số:", "b. Hộp số", "b.Hộp số", r"b\.\s*Hộp số\s*:", r"Hộp số\s*:", r"hộp số\s*:", r"ộp số\s*:", r"p số\s*:", r"số\s*:"],
    "hososo":          ["c. Hồ sơ số:", "c. Hồ sơ số", "c.Hồ sơ số", "Hồ sơ số", "c.Hồ sơ số:", r"c\.\s*Hồ sơ số\s*:", r"Hồ sơ số\s*:", r"hồ sơ số\s*:", r"ồ sơ số\s*:", r"sơ số\s*:", r"số\s*:"],
    "tieudehoso":      ["4. Tiêu đề hồ sơ:", "4.Tiêu đề hồ sơ:", "4. Tiêu đề hồ sơ", "4.Tiêu đề hồ sơ", "Tiêu đề hồ sơ", r"4\.\s*Tiêu đề hồ sơ\s*:", r"Tiêu đề hồ sơ\s*:"],
    "thoigiantailieu": ["5. Thời gian của tài liệu:", "Thời gian của tài liệu", "5. Thời gian của tài liệu", "5.Thời gian của tài liệu:", "5.Thời gian của tài liệu", r"5\.\s*Thời gian của tài liệu\s*:", r"Thời gian của tài liệu\s*:", r"gian của tài liệu\s*:", r"của tài liệu\s*:", r"tài liệu\s*:", r"liệu\s*:"],
    "soto":            ["6. Số lượng tờ:", "Số lượng tờ", "6.Số lượng tờ", "6. Số lượng tờ", "6:Số lượng tờ:", r"6\.\s*Số lượng tờ\s*:", r"Số lượng tờ\s*:", r"lượng tờ\s*:", r"tờ\s*:"],
    "thoigianbaoquan": ["7. Thời hạn bảo quản:", "Thời hạn bảo quản", "7.Thời hạn bảo quản:", "7. Thời hạn bảo quản", r"7\.\s*Thời hạn bảo quản\s*:", r"Thời hạn bảo quản\s*:", r"hạn bảo quản\s*:", r"bảo quản\s*:", r"quản\s*:"],
    "quanlybienban":   ["8. Quản lý biên bản:", "8.Quản lý biên bản:", "8. Quản lý biên bản", "9. Quản lý biên bản:", "Quản lý biên bản", "9.Quản lý biên bản:", "9. Quản lý biên bản", r"[89]\.\s*Quản lý biên bản\s*:", r"Quản lý biên bản\s*:", r"lý biên bản\s*:", r"biên bản\s*:", r"bản\s*:"],
}


def detectron_process_phieutin(image_path: str, request_id: str = "N/A") -> list:
    """
    Call Detectron2 phieutin microservice API (port 8010).
    Classes: key, tenkho, tenphong, soluutru, tenhoso, thoigian, soluong, thoihan, tinhtrang
    """
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        response = httpx.post(
            DETECTRON2_PHIEUTIN_API,
            json={
                "image_base64": img_base64,
                "image_path": image_path
            },
            timeout=300.0
        )
        response.raise_for_status()
        result = response.json()
        return result.get("detections", [])
    except Exception as e:
        celery_logger.error(f"Detectron2 Phieutin API call failed for {image_path}: {e}")
        return []


def _strip_phieutin_prefix(text: str, patterns) -> str:
    """Strip label prefix khỏi đầu chuỗi OCR. patterns là list, thử từng pattern theo thứ tự."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        stripped = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE).strip()
        if stripped != text.strip():
            return stripped
    return text.strip()


def _strip_phieutin_residual_junk(text: str) -> str:
    """Xóa số thứ tự + dấu chấm còn sót lại ở đầu chuỗi sau khi đã strip label
    (vd: '2.BC336' -> 'BC336'), sau đó xóa sạch MỌI dấu hai chấm (':' hoặc full-width '：')
    ở bất kỳ vị trí nào trong chuỗi — vì giá trị thật của các trường này không bao giờ chứa dấu hai chấm.
    Xóa không neo vị trí đầu chuỗi (không dùng ^) để tránh bị ký tự ẩn (vd: khoảng trắng/ký tự rác
    do OCR chèn trước dấu ':') làm regex anchor không khớp."""
    if not text:
        return text
    text = re.sub(r"^\d{0,2}[\.．]+\s*", "", text)
    text = text.replace(":", "").replace("：", "")
    return text.strip()


def _normalize_phieutin_thoigianbaoquan(text: str) -> str:
    """Chuẩn hóa hạn bảo quản: nếu có số + 'năm' thì giữ dạng 'N năm', ngược lại
    (kể cả lỗi chính tả như 'vinh vien', 'vĩnh viên'...) chuẩn hóa thành 'Vĩnh viễn'."""
    if not text:
        return text
    match = re.search(r"(\d+)\s*n[ăâa]m", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} năm"
    return "Vĩnh viễn"


def _run_detection_and_ocr_phieutin(image_paths, logger, request_id: str = "N/A"):
    """Detect và OCR phiếu tin — chỉ xử lý trang 1, không tìm SIGNER đa trang."""
    img_path = image_paths[0]
    temp_folder_base = os.path.dirname(os.path.dirname(img_path))
    crops_to_ocr = []

    image = cv2.imread(img_path)
    if image is not None:
        detected_objects = detectron_process_phieutin(img_path, request_id)
        # Giữ 1 detection per label (confidence cao nhất) trước khi crop và ghi file
        best_detections = {}
        for box_info in detected_objects:
            x1, y1, x2, y2, label, confidence_score = box_info
            if label not in best_detections or confidence_score > best_detections[label][5]:
                best_detections[label] = (x1, y1, x2, y2, label, confidence_score)

        for x1, y1, x2, y2, label, confidence_score in best_detections.values():
            crop_img = image[y1:y2, x1:x2]
            crop_path = os.path.join(temp_folder_base, f"page_1_crop_{label}_{x1}.jpg")
            cv2.imwrite(crop_path, crop_img)
            crops_to_ocr.append({"label": label, "crop_path": crop_path, "confidence": confidence_score})
    else:
        logger.error(f"Phieutin: Cannot read image {img_path}")

    async def run_async_ocr():
        async with httpx.AsyncClient() as client:
            tasks = [
                call_vllm_async(client, PROMPT_MAPPING_PHIEUTIN.get(c["label"], PROMPT_MAPPING_PHIEUTIN["DEFAULT"]), c["crop_path"])
                for c in crops_to_ocr
            ]
            return await asyncio.gather(*tasks)

    ocr_texts = asyncio.run(run_async_ocr())

    raw_results = {}
    for crop, text in zip(crops_to_ocr, ocr_texts):
        raw_results[crop["label"]] = text.strip() if text else ""

    logger.info(f"Phieutin raw OCR results: {raw_results}")

    for label in _PHIEUTIN_PREFIX_MAP:
        if raw_results.get(label) and _PHIEUTIN_PREFIX_MAP.get(label):
            raw_results[label] = _strip_phieutin_prefix(raw_results[label], _PHIEUTIN_PREFIX_MAP[label])

    # Xóa số thứ tự + dấu chấm/hai chấm còn sót lại ở đầu (vd: "2.:BC336" -> "BC336")
    # Áp dụng cho các trường mã/số/ngày tháng, trừ tiêu đề hồ sơ (văn bản tự do)
    # "hososo" KHÔNG nằm trong set này: format hồ sơ số rất biến thiên (có thể là "01.01"),
    # nếu áp dụng strip sẽ cắt nhầm phần dữ liệu thật (vd "01.01" -> "01")
    _RESIDUAL_JUNK_FIELDS = {"hopso", "soto", "quanlybienban", "thoigiantailieu", "thoigianbaoquan"}
    for label in _RESIDUAL_JUNK_FIELDS:
        if raw_results.get(label):
            raw_results[label] = _strip_phieutin_residual_junk(raw_results[label])

    # Riêng "key": xóa sạch mọi dấu hai chấm (':' / full-width '：') ở bất kỳ vị trí
    if raw_results.get("key"):
        raw_results["key"] = raw_results["key"].replace(":", "").replace("：", "").strip()

    # Chuẩn hóa hạn bảo quản
    if raw_results.get("thoigianbaoquan"):
        raw_results["thoigianbaoquan"] = _normalize_phieutin_thoigianbaoquan(raw_results["thoigianbaoquan"])

    # Xóa toàn bộ dấu cách trong các trường số/mã sau khi strip prefix
    _NO_SPACE_FIELDS = {"hopso", "hososo", "soto", "quanlybienban", "thoigiantailieu"}
    for label in _NO_SPACE_FIELDS:
        if raw_results.get(label):
            raw_results[label] = raw_results[label].replace(" ", "")

    return {
        "key":             raw_results.get("key", ""),
        "hopso":           raw_results.get("hopso", ""),
        "hososo":          raw_results.get("hososo", ""),
        "tieudehoso":      raw_results.get("tieudehoso", ""),
        "thoigiantailieu": raw_results.get("thoigiantailieu", ""),
        "soto":            raw_results.get("soto", ""),
        "thoigianbaoquan": raw_results.get("thoigianbaoquan", ""),
        "quanlybienban":   raw_results.get("quanlybienban", ""),
    }


def phieutin_flow(context: dict, logger) -> dict:
    """Processing logic for phiếu tin documents."""
    logger.info(f"Executing Phieutin Flow for request {context['request_id']}")
    request_id = context.get("request_id", "N/A")

    ocr_result = _run_detection_and_ocr_phieutin(context["image_paths"], logger, request_id)

    dataslips_data = {
        "key":             ocr_result.get("key", ""),
        "hopso":           ocr_result.get("hopso", ""),
        "hososo":          ocr_result.get("hososo", ""),
        "tieudehoso":      ocr_result.get("tieudehoso", ""),
        "thoigiantailieu": ocr_result.get("thoigiantailieu", ""),
        "soto":            ocr_result.get("soto", ""),
        "thoigianbaoquan": ocr_result.get("thoigianbaoquan", ""),
        "quanlybienban":   ocr_result.get("quanlybienban", ""),
        "path":            context.get("s3_path", "").replace("s3://", "").split("/", 1)[-1],
    }

    logger.info(f"Phieutin Flow: mapped dataslips_data = {dataslips_data}")
    return {"dataslips_data": dataslips_data}
