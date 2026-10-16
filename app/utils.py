# app/utils.py
import os
import boto3
import pdf2image
import requests
import time
import logging
from urllib.parse import urlparse

import re
import psycopg2
from PyPDF2 import PdfReader
from PIL import Image
from typing import Optional
import cv2
import numpy as np
import multiprocessing


logger = logging.getLogger(__name__)

ENABLE_CAPITALIZATION_RULES = os.getenv("ENABLE_CAPITALIZATION_RULES", "true").strip().lower() in ("1", "true", "yes")
ENABLE_SPELL_FIX = os.getenv("ENABLE_SPELL_FIX", "true").strip().lower() in ("1", "true", "yes")
# Sửa chính tả bằng model ProtonX (dịch vụ app/spellfix_api.py), chạy trước các quy tắc sửa hiện có
ENABLE_ML_SPELL_FIX = os.getenv("ENABLE_ML_SPELL_FIX", "true").strip().lower() in ("1", "true", "yes")
SPELLFIX_API_URL = os.getenv("SPELLFIX_API_URL", "http://localhost:8006/spellfix/correct")
SPELLFIX_TIMEOUT = float(os.getenv("SPELLFIX_TIMEOUT", "60"))
SPELLFIX_LABELS = {l.strip().upper() for l in os.getenv("SPELLFIX_LABELS", "ORG,VV,THELOAI,SENDTO").split(",") if l.strip()}

import unicodedata
def clean_text(text):
    BANG_XOA_DAU = str.maketrans(
        "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
        "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 + "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5
    )
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    return text.translate(BANG_XOA_DAU)

def get_template_name(CustomerID, ConfigTemplateID, logger):
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PW")
        )
        cursor = conn.cursor()
        query = """
        SELECT s.name FROM sampleconfigdocuments s
        WHERE s.id = %s AND s.companyid = %s AND s.status != 99
        LIMIT 1;
        """
        cursor.execute(query, (ConfigTemplateID, CustomerID))
        row = cursor.fetchone()
        
        template_name = row[0] if row else ""
        logger.info(f"DB query for ({CustomerID}, {ConfigTemplateID}) found template name: '{template_name}'")

        cursor.close()
        conn.close()
        return template_name

    except Exception as e:
        logger.error(f"Error loading template name from database: {e}")
        # Fallback to a default name to prevent total failure
        return ""



# --- UPDATED: S3 Download function with better endpoint parsing ---
def download_from_s3(s3_url: str, local_path: str, access_key: str, secret_key: str, endpoint: str) -> tuple[str, str]:
    """
    Download a file from S3/Minio to a local path using provided credentials.
    """
    # pattern = r"^(https?://[\d.]+:\d+)"
    # base_endpoint = ""
    # match = re.match(pattern, endpoint)
    # if match:
    #     base_endpoint = match.group(0)
    # else:
    #     logger.warning(f"Endpoint URL '{endpoint}' does not match expected pattern. Using as-is.")
    #     base_endpoint = endpoint

    try:
        s3_config = {
            'endpoint_url': endpoint,
            'aws_access_key_id': access_key,
            'aws_secret_access_key': secret_key,
            'region_name': os.getenv("S3_REGION", "us-east-1"),
            'use_ssl': os.getenv("S3_SSL", "False").lower() in ('true', '1', 't'),
        }
        
        s3_client = boto3.client('s3', **s3_config)
        
        parsed_url = urlparse(s3_url)
        if parsed_url.scheme != 's3':
            raise ValueError(f"Not an S3 URL: {s3_url}")
        
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        
        logger.info(f"Downloading from bucket: {bucket}, key: {key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content_type = response.get('ContentType', 'application/octet-stream')
        
        os.makedirs(local_path, exist_ok=True)
        
        file_name = os.path.basename(key)
        file_path = os.path.join(local_path, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(response['Body'].read())
            
        logger.info(f"File downloaded successfully to {file_path}")
        return file_path, content_type
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}", exc_info=True)
        raise

# --- OTHER UTILS (UNCHANGED) ---
def convertPDFImage(pdf_file_path: str, output_folder: str) -> list[str]:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        thread_count = multiprocessing.cpu_count()
        print(f"Số logical CPU cores: {thread_count}")
        images = pdf2image.convert_from_path(pdf_file_path,dpi=100, thread_count=thread_count)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
            image.save(image_path, "JPEG", quality=85, optimize=True)
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        raise


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF file.
    """
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        logger.info(f"Extracted metadata from {pdf_path}: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from PDF {pdf_path}: {e}")
        return {}


def find_skew(gray, delta=0.2, limit=5):
    """
    Tìm góc nghiêng của ảnh bằng projection profile (rất chính xác với nghiêng nhẹ 1-5°)
    gray: ảnh grayscale
    delta: bước quét (độ)
    limit: giới hạn quét +/- độ
    """
    angles = np.arange(-limit, limit + delta, delta)
    scores = []

    h, w = gray.shape
    center = (w // 2, h // 2)

    for angle in angles:
        # xoay ảnh thử
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # tính projection theo hàng
        projection = np.sum(rotated, axis=1)

        # score = độ biến thiên (variance) của histogram
        # khi ảnh thẳng → histogram sắc nét → variance cao
        score = np.var(projection)
        scores.append(score)

    # lấy góc có score lớn nhất
    best_angle = angles[np.argmax(scores)]
    return best_angle


def deskew(image):
    """
    Deskew đơn giản bằng find_skew()
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = find_skew(gray)

    print("Detected skew angle:", angle)

    # xoay ngược lại
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def send_webhook_sync(url: str, data: dict):
    # ... (same as before) ...
    logger.info(f"Sending webhook to {url}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, json=data, headers={"Content-Type": "application/json"}, timeout=30.0
            )
            response.raise_for_status()
            logger.info(f"Webhook sent successfully to {url}, status: {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed sending webhook to {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"Failed to send webhook to {url} after {max_retries} attempts.")
    return False



def variants(name: str):
    """Sinh ra các biến thể để kiểm tra."""
    return {
        "origin": name,
        "lower": name.lower(),
        "clean": clean_text(name)
    }

def process_text(text: str, list_co_quan: list) -> Optional[str]:
    text_clean = clean_text(text)
    text_lower = text.lower()

    for cq in list_co_quan:
        # 1) Kiểm tra trùng khớp hoàn toàn
        if text == cq["origin"] or text_lower == cq["lower"] or text_clean == cq["clean"]:
            return text

        # 2) Kiểm tra bắt đầu bằng tên cơ quan
        if (
            text.startswith(cq["origin"]) or
            text_lower.startswith(cq["lower"]) or
            text_clean.startswith(cq["clean"])
        ):
            text = text[len(cq["origin"]):]
            return text.replace("\n", " ").upper() + " " + cq["origin"]

    # 3) Mặc định
    text = text.replace("\n", " ")
    if text.startswith(" "):
        return text[1:].upper()
    return text.upper()


def is_box_in_top_left(x1, y1, x2, y2, image_width, image_height, ratio=0.5):
    region_w = image_width * ratio
    region_h = image_height * ratio
    return x2 <= region_w and y2 <= region_h



def detect_and_warp(image_input):
    """
    image_input: bytes OR PIL.Image OR OpenCV image (numpy array BGR)
    Returns: PIL.Image (RGB) của ảnh đã warp
    """
    from io import BytesIO
    import numpy as np
    import cv2
    from PIL import Image

    # Nếu là bytes -> decode
    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        # PIL -> OpenCV BGR
        img_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_cv = image_input
    else:
        raise ValueError("image_input phải là bytes, PIL.Image, hoặc numpy.ndarray")

    # --- (Tiền xử lý + tìm contour + warp) ---
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy contour!")

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        # Nếu không tìm được 4 điểm, ta có thể fallback: dùng boundingRect
        x, y, w, h = cv2.boundingRect(largest)
        warped_cv = img_cv[y:y+h, x:x+w]
    else:
        pts = approx.reshape(4, 2).astype("float32")
        # order points
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped_cv = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight))

    # Convert BGR -> RGB PIL.Image
    warped_pil = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
    return warped_pil


def is_full_caps_word(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def insert_newline_after_title(text: str) -> str:
    words = text.split()
    current = []
    title = None
    title_end_idx = 0

    for i, w in enumerate(words):
        if is_full_caps_word(w):
            current.append(w)
            title_end_idx = i
        else:
            break

    if not current:
        # không có loại văn bản
        return text

    title = " ".join(current)

    # vị trí kết thúc title trong text gốc
    pos = text.find(title)
    if pos == -1:
        return text

    after_title = pos + len(title)

    # nếu sau title đã là newline → bỏ qua
    if after_title < len(text) and text[after_title] == "\n":
        return text

    # nếu sau title là space rồi newline → bỏ qua
    if text[after_title:after_title+2] == " \n":
        return text

    # chèn newline
    return text[:after_title] + "\n" + text[after_title:].lstrip()


# ── Từ điển sửa lỗi chính tả + viết hoa ────────────────────────────────────
import json
import re as _re
from pathlib import Path as _Path
import openpyxl as _openpyxl

_SPELLCHECK_DIR = _Path(__file__).parent / 'spellcheck_data'
_SPELL_DICT_PATH = _SPELLCHECK_DIR / 'spell_dict.xlsx'
_DVHC_PATH = _SPELLCHECK_DIR / 'dvhc_2021.json'
_spell_cache: list | None = None
_spell_mtime: float = 0.0


def _match_word_case(src: str, word: str) -> str:
    letters = [c for c in src if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return word.upper()
    if src and src[0].isupper():
        return word[:1].upper() + word[1:].lower()
    return word.lower()


def _match_case(matched: str, correct: str) -> str:
    """Giữ nguyên kiểu hoa/thường của đoạn khớp (dùng khi tắt luật viết hoa)."""
    src_words, dst_words = matched.split(), correct.split()
    if len(src_words) == len(dst_words):
        return ' '.join(_match_word_case(s, d) for s, d in zip(src_words, dst_words))
    return _match_word_case(matched, correct)


def _make_replacer(correct: str):
    """Nếu correct có chữ hoa → exact; nếu toàn thường → preserve hoa/thường chữ đầu.
    Khi tắt luật viết hoa → chỉ sửa dấu, giữ nguyên hoa/thường của đoạn khớp."""
    has_upper = any(c.isupper() and c.isalpha() for c in correct)

    def _rep(m: _re.Match) -> str:
        if not ENABLE_CAPITALIZATION_RULES:
            return _match_case(m.group(0), correct)
        if has_upper:
            return correct
        matched = m.group(0)
        if matched and matched[0].isupper() and correct:
            return correct[0].upper() + correct[1:]
        return correct

    return _rep


def _load_spell_dict() -> list:
    global _spell_cache, _spell_mtime
    try:
        mtime = _SPELL_DICT_PATH.stat().st_mtime
    except FileNotFoundError:
        return []
    if _spell_cache is not None and mtime == _spell_mtime:
        return _spell_cache
    entries = []
    wb = _openpyxl.load_workbook(_SPELL_DICT_PATH, read_only=True, data_only=True)
    ws = wb.active
    first = True
    for row in ws.iter_rows(values_only=True):
        if first:
            first = False
            continue
        if not row or len(row) < 2:
            continue
        wrong = str(row[0]).strip() if row[0] is not None else ''
        correct = str(row[1]).strip() if row[1] is not None else ''
        if not wrong or not correct or wrong.startswith('#'):
            continue
        pat = _re.compile(r'(?<!\w)' + _re.escape(wrong) + r'(?!\w)',
                          _re.IGNORECASE | _re.UNICODE)
        entries.append((pat, _make_replacer(correct)))
    wb.close()
    _spell_cache = entries
    _spell_mtime = mtime
    return entries


# Trạng thái ProtonX của tiến trình hiện tại: None = chưa kiểm tra, True/False = đã chốt.
# Kiểm tra một lần lúc khởi động (check_spellfix); đã False thì mọi request đi thẳng luồng quy tắc,
# không kiểm tra lại. Muốn bật lại sau khi dịch vụ chạy trở lại: khởi động lại worker / API.
_spellfix_available = None


def spellfix_health_url() -> str:
    from urllib.parse import urlparse
    p = urlparse(SPELLFIX_API_URL)
    return os.getenv("SPELLFIX_HEALTH_URL", f"{p.scheme}://{p.netloc}/health")


def check_spellfix(force: bool = False) -> bool:
    """Kiểm tra dịch vụ ProtonX một lần và ghi nhớ kết quả cho tiến trình này."""
    global _spellfix_available
    if not ENABLE_ML_SPELL_FIX:
        _spellfix_available = False
        return False
    if _spellfix_available is not None and not force:
        return _spellfix_available
    url = spellfix_health_url()
    try:
        response = requests.get(url, timeout=float(os.getenv("SPELLFIX_HEALTH_TIMEOUT", "3")))
        response.raise_for_status()
        _spellfix_available = True
        logger.info(f"ProtonX spellfix [OK] {url} -> flow: OCR -> ProtonX -> rules")
    except Exception as e:
        _spellfix_available = False
        logger.warning(f"ProtonX spellfix [FAIL] {url} ({e}) -> ProtonX bypassed, flow: OCR -> rules")
    return _spellfix_available


def ml_spell_fix(texts: list) -> list:
    """Gửi danh sách text tới dịch vụ ProtonX; dịch vụ không chạy hoặc lỗi thì trả nguyên văn bản."""
    global _spellfix_available
    if not texts or not check_spellfix():
        return texts
    try:
        response = requests.post(SPELLFIX_API_URL, json={"texts": texts}, timeout=SPELLFIX_TIMEOUT)
        response.raise_for_status()
        fixed = response.json().get("texts", [])
        if len(fixed) != len(texts):
            raise ValueError(f"expected {len(texts)} texts, got {len(fixed)}")
        return fixed
    except Exception as e:
        _spellfix_available = False
        logger.error(f"ProtonX spellfix failed ({e}) -> ProtonX bypassed for this process until restart, using rules only")
        return texts


def fix_spelling(text: str) -> str:
    if not text or not ENABLE_SPELL_FIX:
        return text
    for pat, replacer in _load_spell_dict():
        text = pat.sub(replacer, text)
    return fix_place_names(text)


# ── Sửa chính tả địa danh (tỉnh / huyện / xã) theo danh mục ĐVHC ───────────
# So khớp theo "khung" (bỏ dấu, đ→d) sau từ khóa đơn vị hành chính, nên sửa được
# lỗi OCR về dấu thanh, e/ê, o/ô/ơ, a/ă/â, u/ư, d/đ; l/n được thử ở bước dự phòng.
# Khi nhiều địa danh trùng khung → ưu tiên địa danh thuộc tỉnh/huyện đã nhận diện
# trong cùng văn bản, sau đó chọn địa danh gần nhất (edit distance); hòa thì bỏ qua.

_UNIT_LEVELS = {
    'tỉnh': ('P',), 'thành phố': ('P', 'D'), 'tp': ('P', 'D'),
    'quận': ('D',), 'huyện': ('D',), 'thị xã': ('D',),
    'xã': ('W',), 'phường': ('W',), 'thị trấn': ('W',),
}
_UNIT_RE = _re.compile(
    r'(?<!\w)(thành phố|thị xã|thị trấn|tỉnh|tp\.?|quận|huyện|xã|phường)(?!\w)',
    _re.IGNORECASE | _re.UNICODE,
)
_NAME_WORD_RE = _re.compile(r'[ \t]+(?:[-–][ \t]+)?([^\W_]+)', _re.UNICODE)
_MAX_NAME_WORDS = 5

_OLD_TONE_RE = _re.compile(r'([oOuU])([àáảãạèéẻẽẹỳýỷỹỵ])(?!\w)', _re.UNICODE)
_TONE_SHIFT = dict(zip('àáảãạèéẻẽẹỳýỷỹỵ', 'aaaaaeeeeeyyyyy'))
_TONE_ON_O = dict(zip('àáảãạèéẻẽẹ', 'òóỏõọòóỏõọ'))
_TONE_ON_U = dict(zip('ỳýỷỹỵ', 'ùúủũụ'))

_dvhc_index: dict | None = None


def _tone_key(s: str) -> str:
    """Chuẩn hóa NFC + lowercase + bỏ dấu kiểu cũ (hoà → hòa) để so sánh."""
    s = unicodedata.normalize('NFC', s).lower()

    def _shift(m: _re.Match) -> str:
        v, t = m.group(1).lower(), m.group(2)
        if v == 'o' and t in _TONE_ON_O:
            return _TONE_ON_O[t] + _TONE_SHIFT[t]
        if v == 'u' and t in _TONE_ON_U:
            return _TONE_ON_U[t] + _TONE_SHIFT[t]
        return m.group(0)

    return _OLD_TONE_RE.sub(_shift, s)


def _skeleton(s: str, fold_ln: bool = False) -> str:
    s = unicodedata.normalize('NFD', s.lower().replace('đ', 'd'))
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = ' '.join(_re.sub(r'[^\w\s]', ' ', s).split())
    return s.replace('l', 'n') if fold_ln else s


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _load_dvhc_index() -> dict:
    """{(level, skeleton): [(name, province, district, level)]} + bản fold l/n."""
    global _dvhc_index
    if _dvhc_index is not None:
        return _dvhc_index
    exact, folded = {}, {}

    def _add(level, name, prov, dist):
        if len(name.split()) < 2:  # tên 1 âm tiết quá dễ trùng từ thường (vd. "xã hội")
            return
        entry = (name, prov, dist, level)
        exact.setdefault((level, _skeleton(name)), []).append(entry)
        folded.setdefault((level, _skeleton(name, True)), []).append(entry)

    try:
        with open(_DVHC_PATH, encoding='utf-8') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy danh mục ĐVHC: {_DVHC_PATH}")
        data = []
    for p in data:
        _add('P', p['name'], p['name'], None)
        for d in p.get('districts', []):
            _add('D', d['name'], p['name'], d['name'])
            for w in d.get('wards', []):
                _add('W', w['name'], p['name'], d['name'])
    _dvhc_index = {'exact': exact, 'folded': folded}
    return _dvhc_index


def _place_candidates(levels, observed: str) -> list:
    idx = _load_dvhc_index()
    for table, fold in (('exact', False), ('folded', True)):
        sk = _skeleton(observed, fold)
        found = [e for lv in levels for e in idx[table].get((lv, sk), [])]
        if found:
            return found
    return []


def _pick_place(observed: str, cands: list, provinces: set, districts: set):
    obs = _tone_key(observed)
    in_ctx = [c for c in cands if c[2] in districts] or [c for c in cands if c[1] in provinces]
    pool = in_ctx or cands
    for c in pool:
        if _tone_key(c[0]) == obs:
            return c
    ranked = sorted({_tone_key(c[0]): c for c in pool}.values(),
                    key=lambda c: _edit_distance(obs, _tone_key(c[0])))
    if len(ranked) > 1 and (_edit_distance(obs, _tone_key(ranked[0][0]))
                            == _edit_distance(obs, _tone_key(ranked[1][0]))):
        return None
    return ranked[0] if ranked else None


def _place_case(observed: str, name: str) -> str:
    letters = [c for c in observed if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return name.upper()
    if ENABLE_CAPITALIZATION_RULES:
        return name
    return _match_case(observed, name)


def fix_place_names(text: str) -> str:
    """Sửa lỗi OCR cho tên tỉnh/huyện/xã đứng sau từ khóa đơn vị hành chính."""
    if not text or not ENABLE_SPELL_FIX:
        return text
    found = []  # (level, start, end, observed, candidates)
    for m in _UNIT_RE.finditer(text):
        levels = _UNIT_LEVELS[m.group(1).lower().rstrip('.')]
        spans, pos = [], m.end()
        while len(spans) < _MAX_NAME_WORDS:
            w = _NAME_WORD_RE.match(text, pos)
            if not w:
                break
            spans.append(w.span(1))
            pos = w.end()
        for k in range(len(spans), 1, -1):
            start, end = spans[0][0], spans[k - 1][1]
            cands = _place_candidates(levels, text[start:end])
            if cands:
                found.append((start, end, text[start:end], cands))
                break

    # Giải quyết tỉnh trước, rồi huyện, rồi xã — dùng cấp trên làm ngữ cảnh.
    provinces, districts, picks = set(), set(), {}
    for rank in ('P', 'D', 'W'):
        for item in found:
            start, end, observed, cands = item
            if start in picks or not any(c[3] == rank for c in cands):
                continue
            pick = _pick_place(observed, [c for c in cands if c[3] == rank],
                               provinces, districts)
            if pick:
                picks[start] = (end, observed, pick)
                provinces.add(pick[1])
                if pick[2]:
                    districts.add(pick[2])

    for start in sorted(picks, reverse=True):
        end, observed, (name, *_) = picks[start]
        text = text[:start] + _place_case(observed, name) + text[end:]
    return text


# ── Chuẩn hóa viết tắt ──────────────────────────────────────────────────────

_ABBREV_PATTERNS = [
    (_re.compile(r'Hội\s+đồng\s+nhân\s+dân', _re.IGNORECASE | _re.UNICODE), 'HĐND'),
    (_re.compile(r'HỘI\s+ĐỒNG\s+NHÂN\s+DÂN', _re.UNICODE), 'HĐND'),
    (_re.compile(r'Ủy\s+ban\s+nhân\s+dân', _re.IGNORECASE | _re.UNICODE), 'UBND'),
    (_re.compile(r'ỦY\s+BAN\s+NHÂN\s+DÂN', _re.UNICODE), 'UBND'),
    (_re.compile(r'\bubnd\b', _re.IGNORECASE | _re.UNICODE), 'UBND'),
    (_re.compile(r'\bhđnd\b', _re.IGNORECASE | _re.UNICODE), 'HĐND'),
    (_re.compile(r'về\s+việc', _re.IGNORECASE | _re.UNICODE), 'v/v'),
]


def normalize_abbrevs(text: str) -> str:
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sentence_case(text: str) -> str:
    """Lowercase toàn bộ rồi viết hoa chữ cái đầu tiên."""
    if not text or not ENABLE_CAPITALIZATION_RULES:
        return text
    s = text.lower()
    return s[0].upper() + s[1:]


_ADMIN_UNIT_RE = _re.compile(
    r'\b(thôn|xóm|xã|phường|quận|huyện|tỉnh|thành phố|thị xã|thị trấn)\s+([^\n,;.(]+)',
    _re.IGNORECASE | _re.UNICODE,
)


def _title_vn(s: str) -> str:
    words = s.split()
    return ' '.join(w[0].upper() + w[1:] if w else w for w in words)


def capitalize_after_admin_units(text: str) -> str:
    if not text or not ENABLE_CAPITALIZATION_RULES:
        return text
    def _cap(m: _re.Match) -> str:
        unit = m.group(1)
        name = m.group(2).strip()
        return unit + ' ' + _title_vn(name)
    return _ADMIN_UNIT_RE.sub(_cap, text)


# ── chuan_hoa_tac_gia — port từ SQL ─────────────────────────────────────────
_GEO   = {'tỉnh','thành phố','tp','quận','huyện','thị xã','thị trấn','xã','phường','thôn'}
_PHR   = {'sở','phòng','ban','chi cục','cục','tổng cục','vụ','viện','trung tâm',
          'văn phòng','trường','bệnh viện','công ty','tập đoàn','chi nhánh','đội',
          'hội đồng','ủy ban','ban quản lý'}
_ORG   = {'nhà nước','chính phủ','quốc hội','bộ','thứ','cơ quan'}
_LEGAL = {'tnhh','cổ phần','hợp danh','tư nhân','một thành viên'}
_ABBR  = {
    'ubnd', 'hđnd', 'tnhh', 'cp', 'tnhh-mtv', 'bql', 'ban ql',
    # Cổng/hệ thống thông tin
    'ttđt', 'ttcntt', 'cntt',
    # Tổ chức / cơ quan
    'tctk', 'tcthn', 'tct', 'bca', 'bqp', 'bnn', 'btnmt', 'bgddt', 'bgtvt',
    'byt', 'bld', 'bkhdt', 'btc', 'btp', 'bvhttdl', 'btttt', 'bkhcn',
    'bxd', 'bcvt', 'bng', 'bct', 'bts',
    # Các tổ chức khác
    'vpcp', 'vp', 'pvn', 'evn', 'vnpt', 'viettel',
    'hđqt', 'hđtv', 'ban bql',
}
_GUARD = {'và','or','&'}
_CONJ  = {'và','hoặc','or','&','-','–'}

_COMPOUNDS = [
    ('chi cục', 'chi~cục'), ('tổng cục', 'tổng~cục'), ('thành phố', 'thành~phố'),
    ('thị xã', 'thị~xã'), ('thị trấn', 'thị~trấn'), ('ban quản lý', 'ban~quản~lý'),
    ('trung tâm', 'trung~tâm'), ('văn phòng', 'văn~phòng'), ('bệnh viện', 'bệnh~viện'),
    ('hội đồng', 'hội~đồng'), ('ủy ban', 'ủy~ban'), ('chi nhánh', 'chi~nhánh'),
    ('một thành viên', 'một~thành~viên'), ('tập đoàn', 'tập~đoàn'),
    ('chính phủ', 'chính~phủ'), ('quốc hội', 'quốc~hội'), ('nhà nước', 'nhà~nước'),
    ('cơ quan', 'cơ~quan'),
]

_ALREADY_UPPER_RE = _re.compile(r'^[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]{2,}', _re.UNICODE)
_STARTS_UPPER_RE  = _re.compile(r'^[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]', _re.UNICODE)


def _vn_hoa_dau(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]


_VN_TEN_RIENG = [
    (_re.compile(r'(?<!\w)ha noi(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Hà Nội'),
    (_re.compile(r'(?<!\w)ho chi minh(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Hồ Chí Minh'),
    (_re.compile(r'(?<!\w)da nang(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Đà Nẵng'),
    (_re.compile(r'(?<!\w)hai phong(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Hải Phòng'),
    (_re.compile(r'(?<!\w)can tho(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Cần Thơ'),
    (_re.compile(r'(?<!\w)đak lak(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Đắk Lắk'),
    (_re.compile(r'(?<!\w)đăk lăk(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Đắk Lắk'),
    (_re.compile(r'(?<!\w)lang son(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Lạng Sơn'),
    (_re.compile(r'(?<!\w)ninh thuan(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Ninh Thuận'),
    (_re.compile(r'(?<!\w)binh duong(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Bình Dương'),
    (_re.compile(r'(?<!\w)khanh hoa(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Khánh Hòa'),
    (_re.compile(r'(?<!\w)quang nam(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Quảng Nam'),
    (_re.compile(r'(?<!\w)thua thien hue(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Thừa Thiên Huế'),
    (_re.compile(r'(?<!\w)lam dong(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Lâm Đồng'),
    (_re.compile(r'(?<!\w)dong nai(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Đồng Nai'),
    (_re.compile(r'(?<!\w)dong thap(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Đồng Tháp'),
    (_re.compile(r'(?<!\w)tien giang(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Tiền Giang'),
    (_re.compile(r'(?<!\w)vung tau(?!\w)', _re.IGNORECASE | _re.UNICODE), 'Vũng Tàu'),
]


def _vn_ten_rieng(s: str) -> str:
    for pat, repl in _VN_TEN_RIENG:
        s = pat.sub(repl, s)
    return s


def chuan_hoa_tac_gia(text: str) -> str:
    """Chuẩn hóa tên tác giả / cơ quan ban hành — port từ SQL chuan_hoa_tac_gia."""
    if not text or not ENABLE_CAPITALIZATION_RULES:
        return text
    s = _re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    if not s:
        return s

    s = s.lower()
    s = _re.sub(r'(ủy|uỷ)\s+ban\s+nhân\s+dân', 'UBND', s)
    s = _re.sub(r'hội\s+đồng\s+nhân\s+dân',     'HĐND', s)

    for src, dst in _COMPOUNDS:
        s = s.replace(src, dst)

    parts = [w for w in s.split(' ') if w]
    n     = len(parts)
    outp  = []
    mode  = 'low'
    cap   = False
    tcnt  = 0
    prev_k = ''

    for w in parts:
        k = w.strip('.,;:()[]"\'-')
        k_raw = k.replace('~', ' ')

        if _ALREADY_UPPER_RE.match(w):
            outp.append(w)
            mode = 'low'

        elif k_raw in _ABBR:
            outp.append(w.upper())
            if mode != 'phrase':
                mode = 'low'; cap = False

        elif k_raw == 'tp':
            outp.append('TP')
            mode = 'title'; cap = False; tcnt = 0

        elif k_raw in _LEGAL:
            outp.append(w)
            mode = 'phrase'; cap = True

        elif k_raw in _GEO:
            outp.append(_vn_hoa_dau(w))
            mode = 'title'; cap = False; tcnt = 0

        elif k_raw in _ORG:
            outp.append(_vn_hoa_dau(w) if (mode == 'phrase' and cap) else w)
            mode = 'low'; cap = False

        elif k_raw in _PHR:
            outp.append(w if prev_k in _GUARD else _vn_hoa_dau(w))
            mode = 'phrase'; cap = True

        elif k_raw in _CONJ or k_raw == ',':
            outp.append(w)
            if mode == 'phrase':
                cap = True

        elif mode == 'title' and tcnt < 4:
            outp.append(_vn_hoa_dau(w))
            tcnt += 1

        elif mode == 'phrase' and cap:
            outp.append(_vn_hoa_dau(w))
            cap = False

        else:
            if mode == 'title':
                mode = 'low'
            outp.append(w)

        if mode == 'phrase' and n <= 12 and w and w[-1] in ',-–/':
            cap = True

        prev_k = k

    s = ' '.join(outp).replace('~', ' ')
    if s and not _STARTS_UPPER_RE.match(s):
        s = _vn_hoa_dau(s)
    return _vn_ten_rieng(s)
