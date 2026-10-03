import cv2
import numpy as np
import json
import os

def detect_colors_in_image(image_path):
    """
    Kiểm tra sự hiện diện của các màu xanh da trời, đỏ, và đen trong ảnh

    Args:
        image_path (str): Đường dẫn đến file ảnh

    Returns:
        dict: Dictionary chứa kết quả kiểm tra màu sắc
    """

    # Kiểm tra file có tồn tại không
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc file ảnh: {image_path}")

    # Chuyển đổi từ BGR sang HSV để detect màu tốt hơn
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Định nghĩa phạm vi HSV cho từng màu

    # Xanh da trời (Blue)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Đỏ (Red) - có 2 phạm vi vì red nằm ở 2 đầu của color wheel
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Đen (Black)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Tạo mask cho từng màu
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2  # Kết hợp 2 phạm vi của màu đỏ
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Kiểm tra sự hiện diện của từng màu
    # Nếu có ít nhất 100 pixel của màu đó thì coi như màu đó có trong ảnh
    min_pixels = 100

    blue_detected = bool(np.sum(blue_mask) > min_pixels)
    red_detected = bool(np.sum(red_mask) > min_pixels)
    black_detected = bool(np.sum(black_mask) > min_pixels)

    # Tạo kết quả JSON
    result = {
        "Blue": blue_detected,
        "Red": red_detected,
        "Black": black_detected
    }

    return result
