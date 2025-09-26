import sys
import os
import cv2
import math
import numpy as np
from PIL import Image
import pyocr
import pyocr.builders

def gamma(img, gam, count):
    """ガンマ変換"""
    Y = np.ones((256, 1), dtype = 'uint8') * 0
    for i in range(256):
        Y[i][0] = 255 * pow(float(i) / 255, 1.0 / gam)
    img_dst = cv2.LUT(img, Y)
    count += 1
    return img_dst, count

def gray(img, count):
    """グレイスケール化"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count += 1
    return img_gray, count

def senei(img, k, count):
    """鮮鋭化"""
    a = np.array([[-k, -k, -k],
        [-k, 1+8*k, -k],
        [-k, -k, -k]])
    img_tmp = cv2.filter2D(img, -1, a)
    img_dst = cv2.convertScaleAbs(img_tmp)
    count += 1
    return img_dst, count

def localbinary(img, count):
    """局所的二値化"""
    img_dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 51, 30)
    count += 1
    return img_dst, count

def bilateral(img, d, color, space, count):
    """バイラテラルフィルタ（エッジ保持平滑化）"""
    img_dst = cv2.bilateralFilter(img, d, color, space)
    count += 1
    return img_dst, count

def resize(img, n, count):
    """画像のサイズ変更"""
    height, width = img.shape[:2]
    mul = n #倍率
    img_dst = cv2.resize(img, dsize=None, fx=mul, fy=mul)
    count += 1
    return img_dst, count

def adjust(img, a, b, count):
    """コントラスト強調"""
    img_dst = np.zeros(img.shape, img.dtype)
    img_dst = cv2.convertScaleAbs(img, alpha = a, beta = b)
    img_dst = cv2.convertScaleAbs(img_dst)
    count += 1
    return img_dst, count

def process_image(file_src):
    """画像を処理してOCRを実行"""
    img_src = cv2.imread(file_src, 1)
    
    if img_src is None:
        print(f"エラー: 画像ファイル '{file_src}' を読み込めません")
        return None
    
    count = 0
    img = img_src
    
    # 画像処理パイプライン
    img, count = gamma(img, 1.5, count)
    img, count = resize(img, 4.0, count)
    img, count = gray(img, count)
    img, count = adjust(img, 1.0, -80.0, count)
    img, count = senei(img, 2, count)
    img, count = bilateral(img, 15, 20, 6, count)
    img, count = adjust(img, 1.1, -20.0, count)
    img, count = resize(img, 0.25, count)
    img, count = localbinary(img, count)
    
    # 最終画像を保存
    cv2.imwrite("processed_image.jpg", img)
    
    # Tesseract-OCRの設定
    path_tesseract = "C:\\Program Files\\Tesseract-OCR"
    if path_tesseract not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + path_tesseract
    
    tools = pyocr.get_available_tools()
    if not tools:
        print("エラー: OCRツールが見つかりません")
        return None
    
    tool = tools[0]
    builder = pyocr.builders.TextBuilder()
    
    try:
        result = tool.image_to_string(Image.fromarray(img), lang="jpn", builder=builder)

        if result.strip():
            return result
        else:
            print("結果が空です")
            return None
    except Exception as e:
        print(f"OCRエラー: {e}")
        print("英語で再試行...")
        try:
            result = tool.image_to_string(Image.fromarray(img), lang="eng", builder=builder)
            print("英語OCR完了")
            return result
        except Exception as e2:
            print(f"英語OCRエラー: {e2}")
            return None

def main():
    if len(sys.argv) != 2:
        print("使用方法: python ocr_recognition.py <画像ファイル>")
        sys.exit(1)
    
    file_src = sys.argv[1]
    result = process_image(file_src)
    
    if result:
        print(result)
        
        # 結果をファイルに保存
        with open("ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(result)

    else:
        print("OCRの実行に失敗しました")

if __name__ == "__main__":
    main()
