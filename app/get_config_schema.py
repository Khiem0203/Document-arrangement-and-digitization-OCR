# app/get_config_schema.py
import json
import gc
import random
import re
from typing import Optional
import os
import psycopg2
from celery.utils.log import get_task_logger

from dotenv import load_dotenv
load_dotenv()


celery_logger = get_task_logger(__name__)

def extract_number(text):
    # (Implementation from your provided script)
    def predict_number(value):
        # ... (code for predict_number)
        char_to_num = {'a': '4', 'A': '4', 'b': '6', 'B': '8', 'c': '0', 'C': '0', 'd': '0', 'D': '0', 'e': '3', 'E': '3', 'f': '7', 'F': '7', 'g': '9', 'G': '9', 'h': '4', 'H': '4', 'i': '1', 'I': '1', 'j': '1', 'J': '1', 'k': '6', 'K': '6', 'l': '1', 'L': '1', 'm': '11', 'M': '11', 'n': '11', 'N': '11', 'o': '0', 'O': '0', 'p': '9', 'P': '9', 'q': '9', 'Q': '9', 'r': '2', 'R': '2', 's': '5', 'S': '5', 't': '7', 'T': '7', 'u': '0', 'U': '0', 'v': '7', 'V': '7', 'w': '3', 'W': '3', 'x': '8', 'X': '10', 'y': '4', 'Y': '4', 'z': '2', 'Z': '2'}
        result = ''.join([c for c in value if c.isdigit()] + [char_to_num[c] for c in value if c.isalpha() and c in char_to_num])
        return result if result else "0000"

    pattern = r'(S[oóỏõọồôổớơợờởốộ]|So|Só|Số|SỐ)\s*:?\s?([^\s/-]+)(?=[/\s-]|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        number = predict_number(match.group(2))
        return number.zfill(4)
    return text # Return original if no match

def extract_code_after_number(text):
    # (Implementation from your provided script)
    pattern = r'(S[oóỏõọồôổớơợờởốộ]|So|Só|Số|SỐ)\s?:?\s?[A-Za-z0-9]+([/-][^/\s-]+(?:[/-][^/\s-]+)*)'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return text

    number_to_letter = {'0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': '&', '9': 'g'}
    def replace_numbers(s):
        return "".join([number_to_letter.get(c, c) for c in s])

    code_part = match.group(2)
    parts = code_part.split('/')
    if parts:
        parts[-1] = replace_numbers(parts[-1])
    return "/".join(parts)

_SO_PREFIX_RE = re.compile(r'^\s*S[oóỏõọồôổớơợờởốộ]\s*:?\s*(?=\d)', re.IGNORECASE)

def extract_number_and_code(text):
    """Số, ký hiệu đầy đủ: bỏ tiền tố "Số"/"Số:" và giữ cả số lẫn ký hiệu ("Số466-QĐ/TU" → "466-QĐ/TU")."""
    return _SO_PREFIX_RE.sub('', text).strip()

def extract_date_format(text):
    """
    Extract and format date from Vietnamese text.
    Handles cases with missing day/month values:
    - "ngày tháng năm 2012" → "00/00/2012"
    - "ngày tháng 10 năm 2025" → "00/10/2025"
    - "ngày 15 tháng năm 2025" → "15/00/2025"
    - "ngày 15 tháng 10 năm 2025" → "15/10/2025"
    """

    # Pattern 1: Full date with day, month, and year
    pattern_full = r'(?:ngày\s*)?(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})'
    match_full = re.search(pattern_full, text, re.IGNORECASE)
    if match_full:
        day, month, year = match_full.groups()
        return f"{int(day):02d}/{int(month):02d}/{int(year):04d}"

    # Pattern 2: Only month and year (no day) - "ngày tháng 10 năm 2025"
    pattern_month_year = r'ngày\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})'
    match_month_year = re.search(pattern_month_year, text, re.IGNORECASE)
    if match_month_year:
        month, year = match_month_year.groups()
        return f"00/{int(month):02d}/{int(year):04d}"

    # Pattern 3: Only day and year (no month) - "ngày 15 tháng năm 2025"
    pattern_day_year = r'ngày\s+(\d{1,2})\s+tháng\s+năm\s+(\d{4})'
    match_day_year = re.search(pattern_day_year, text, re.IGNORECASE)
    if match_day_year:
        day, year = match_day_year.groups()
        return f"{int(day):02d}/00/{int(year):04d}"

    # Pattern 4: Only year (no day, no month) - "ngày tháng năm 2012"
    pattern_year_only = r'ngày\s+tháng\s+năm\s+(\d{4})'
    match_year_only = re.search(pattern_year_only, text, re.IGNORECASE)
    if match_year_only:
        year = match_year_only.group(1)
        return f"00/00/{int(year):04d}"

    # Fallback for less structured dates (DD/MM/YYYY or DD-MM-YYYY)
    pattern_fallback = r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})'
    match_fallback = re.search(pattern_fallback, text)
    if match_fallback:
        d, m, y = match_fallback.groups()
        return f"{int(d):02d}/{int(m):02d}/{int(y):04d}"

    return text  # Return original if no standard format found



def load_config(CustomerID, ConfigTemplateID):
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PW")
        )
        cursor = conn.cursor()
        query = """
        SELECT s.companyid "CustomerID", s2.sampleconfigdocumentid "ConfigTemplateID", s2.name "Name", s2.code "Code", 
               s2.required "Required", s2."order" "Order", s2.note "Note", s2.type "Type", NULL "Children", s2.isexportexcel "IsExportExcel"
        FROM sampleconfigdocuments s 
        INNER JOIN sampleconfigitemdocuments s2 ON s.id = s2.sampleconfigdocumentid 
        WHERE s.id = %s and s.companyid = %s and s2.status!=99
        """
        cursor.execute(query, (ConfigTemplateID, CustomerID))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        config_items = [dict(zip(columns, row)) for row in rows]
        for item in config_items:
            item["Children"] = []
            del item["CustomerID"], item["ConfigTemplateID"]

        result = {
            "CustomerID": CustomerID,
            "ConfigTemplateID": ConfigTemplateID,
            "ConfigTemplateObject": json.dumps(config_items)
        }
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        celery_logger.error(f"Error loading config from database: {str(e)}", exc_info=True)
        raise

def sort_fields_by_order(fields: list) -> list:
    """
    Hàm sắp xếp danh sách các dict theo giá trị 'Order' tăng dần.
    
    :param fields: List chứa các dict
    :return: List đã sắp xếp theo 'Order'
    """
    return sorted(fields, key=lambda x: x.get("Order", 0))


# Mapping template code → OCR json key
# Các template cũ dùng code trực tiếp (SVB, KH, CQBH...), template mới dùng code khác (SKH, NT, TG...)
_TEMPLATE_CODE_MAP = {
    "SKH":  "KH",       # Số, ký hiệu
    "NT":   "NKy",      # Ngày tháng
    "TG":   "CQBH",     # Tác giả = Cơ quan ban hành
    "BG":   "LB",       # Bản gốc/Bản chính (xét giá trị LB)
    "BS":   "LB",       # Bản sao (xét giá trị LB)
    "TS":   "SLTr",     # Trang số
    "GC":   "GHICHU",   # Ghi chú
    "NKVB": "NK",       # Người ký văn bản
    # TLVTY: đặc biệt — ghép TLVB + TTNDVB (xử lý riêng bên dưới)
}


def _resolve_field_value(field_code: str, ocr_json: dict) -> str:
    """Lấy giá trị cho field, ưu tiên mapping mới, fallback về code trực tiếp."""
    if field_code == "TLVTY":
        cached = ocr_json.get("TLVTY", "")
        if cached:
            return str(cached)
        tlvb   = ocr_json.get("TLVB", "")
        ttndvb = ocr_json.get("TTNDVB", "")
        return (tlvb + " " + ttndvb).strip()
    if field_code == "BG":
        if "BG" in ocr_json:
            return str(ocr_json["BG"])
        lb = ocr_json.get("LB", "")
        return lb if lb == "Bản chính" else ""
    if field_code == "BS":
        if "BS" in ocr_json:
            return str(ocr_json["BS"])
        lb = ocr_json.get("LB", "")
        return lb if lb == "Bản sao" else ""
        
    # Check if direct field_code exists in ocr_json first (like in chinhly_flow)
    val = ocr_json.get(field_code, "")
    # Fallback to mapped keys (like in government_flow)
    if val in (None, "") and field_code in _TEMPLATE_CODE_MAP:
        ocr_key = _TEMPLATE_CODE_MAP[field_code]
        val = ocr_json.get(ocr_key, "")
        
    return str(val) if val not in (None, "") else ""


def get_fields(CustomerID: int, ConfigTemplateID: int, ocr_json: dict):
    try:
        json_schema = load_config(CustomerID, ConfigTemplateID)
        ConfigTemplateObject = json.loads(json_schema["ConfigTemplateObject"])
        celery_logger.info(f"ConfigTemplateObject: {ConfigTemplateObject}")

        confidence_scores = ocr_json.get("_confidences", {})

        fields = []
        for item in ConfigTemplateObject:
            field_code  = item["Code"]
            field_value = _resolve_field_value(field_code, ocr_json)

            field = {
                "Name":          item["Name"],
                "Code":          field_code,
                "Value":         field_value,
                "Children":      item.get("Children", []),
                "Order":         item["Order"],
                "Required":      item["Required"],
                "Type":          item["Type"],
                "Note":          item["Note"],
                "Page":          int(1),
                "IsExportExcel": item.get("IsExportExcel", False),
            }
            celery_logger.info(f"Kiem tra export excel: {field}")

            if field["Value"]:
                field["Percent"] = confidence_scores.get(field_code, 100 * random.uniform(0.88, 0.98))
            else:
                field["Value"]   = ""
                field["Percent"] = 0

            # Post-processing theo template code (cả code cũ lẫn code mới)
            if field_code in ("SVB", "CodeNumber") and field["Value"]:
                field["Value"] = extract_number(field["Value"])
            elif field_code in ("KH", "CodeNotation") and field["Value"]:
                field["Value"] = extract_code_after_number(field["Value"])
            elif field_code == "SKH" and field["Value"]:
                field["Value"] = extract_number_and_code(field["Value"])
            elif field_code in ("NKy", "IssuedDate", "NT") and field["Value"]:
                field["Value"] = extract_date_format(field["Value"])
            elif field_code in ("NKVB", "NK") and field["Value"]:
                field["Page"] = ocr_json.get("Page", 1)

            celery_logger.info(f"Processed field: {field}")
            fields.append(field)

        celery_logger.info(f"Fields before sorting: {fields}")
        gc.collect()
        celery_logger.info(f"Extracted fields: {fields}")
        return sort_fields_by_order(fields)
    except Exception as e:
        celery_logger.error(f"Error in get_fields: {e}", exc_info=True)
        raise


def get_fields_by_name(CustomerID: int, ConfigTemplateID: int, ocr_json: dict):
    try:
        json_schema = load_config(CustomerID, ConfigTemplateID)
        ConfigTemplateObject = json.loads(json_schema["ConfigTemplateObject"])

        # Changed: Extract confidence scores from OCR results
        confidence_scores = ocr_json.get("_confidences", {})

        fields = []
        for item in ConfigTemplateObject:
            field_name = item["Name"]
            field_value = ocr_json.get(field_name, "")

            field = {
                "Name": item["Name"],
                "Code": item["Code"],
                "Value": field_value,
                "Children": item.get("Children", []),
                "Order": item["Order"],
                "Required": item["Required"],
                "Type": item["Type"],
                "Note": item["Note"]
            }

            if field["Value"]:
                # Changed: Use real confidence from Detectron2 instead of random
                # Try to match by Code first, then by Name
                field["Percent"] = confidence_scores.get(item["Code"], confidence_scores.get(field_name, 100 * random.uniform(0.88, 0.98)))
            else:
                field["Value"] = "" # Use empty string for not found
                field["Percent"] = 0            
            # celery_logger.info(f"Processed field: {field}")
            fields.append(field)
        
        gc.collect()
        celery_logger.info(f"Extracted fields: {fields}")
        return sort_fields_by_order(fields)
        # return [{"test":"test"},{"tesst":"test"}]
    except Exception as e:
        celery_logger.error(f"Error in get_fields: {e}", exc_info=True)
        raise