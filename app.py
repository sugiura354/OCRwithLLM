from flask import Flask, request, render_template, jsonify, send_file
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import pyocr
import pyocr.builders
import requests
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# アップロードフォルダの設定
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# フォルダを作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gamma(img, gam):
    """ガンマ変換"""
    Y = np.ones((256, 1), dtype = 'uint8') * 0
    for i in range(256):
        Y[i][0] = 255 * pow(float(i) / 255, 1.0 / gam)
    return cv2.LUT(img, Y)

def gray(img):
    """グレイスケール化"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def senei(img, k):
    """鮮鋭化"""
    a = np.array([[-k, -k, -k],
        [-k, 1+8*k, -k],
        [-k, -k, -k]])
    img_tmp = cv2.filter2D(img, -1, a)
    return cv2.convertScaleAbs(img_tmp)

def localbinary(img):
    """局所的二値化"""
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 30)

def bilateral(img, d, color, space):
    """バイラテラルフィルタ（エッジ保持平滑化）"""
    return cv2.bilateralFilter(img, d, color, space)

def resize(img, n):
    """画像のサイズ変更"""
    return cv2.resize(img, dsize=None, fx=n, fy=n)

def adjust(img, a, b):
    """コントラスト強調"""
    img_dst = np.zeros(img.shape, img.dtype)
    img_dst = cv2.convertScaleAbs(img, alpha = a, beta = b)
    return cv2.convertScaleAbs(img_dst)

def process_image_for_ocr(image_path):
    """画像をOCR用に処理"""
    img_src = cv2.imread(image_path, 1)
    
    if img_src is None:
        return None
    
    # 画像サイズを確認
    height, width = img_src.shape[:2]
    print(f"元画像サイズ: {width}x{height}")
    
    # 画像が小さい場合は拡大
    if width < 300 or height < 300:
        scale_factor = max(300/width, 300/height)
        img_src = resize(img_src, scale_factor)
        print(f"拡大後サイズ: {img_src.shape[1]}x{img_src.shape[0]}")
    
    # 画像処理パイプライン（パラメータを調整）
    img = gamma(img_src, 1.2)  # ガンマ値を下げる
    img = resize(img, 4.0)      # 拡大率を下げる
    img = gray(img)
    img = adjust(img, 1.2, -50.0)  # コントラスト調整を緩和
    img = senei(img, 1.5)      # 鮮鋭化を緩和
    img = bilateral(img, 9, 75, 75)  # バイラテラルフィルタを調整
    img = adjust(img, 1.1, -10.0)  # 最終調整を緩和
    img = resize(img, 0.5)      # 縮小率を調整
    img = localbinary(img)
    

    cv2.imwrite("processed_image.jpg", img)
    return img

def perform_ocr(image_path):
    """OCRを実行"""
    # Tesseract-OCRの設定
    path_tesseract = "C:\\Program Files\\Tesseract-OCR"
    if path_tesseract not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + path_tesseract
    
    tools = pyocr.get_available_tools()
    if not tools:
        return None, "OCRツールが見つかりません"
    
    tool = tools[0]
    builder = pyocr.builders.TextBuilder()
    
    # 画像を処理
    processed_img = process_image_for_ocr(image_path)
    if processed_img is None:
        return None, "画像の処理に失敗しました"
    
    # デバッグ用：処理された画像を保存
    debug_path = os.path.join(PROCESSED_FOLDER, f"debug_{os.path.basename(image_path)}")
    cv2.imwrite(debug_path, processed_img)
    
    try:
        result = tool.image_to_string(Image.fromarray(processed_img), lang="jpn", builder=builder)
        if result.strip():
            return result, None
        else:
            # 英語で再試行
            result = tool.image_to_string(Image.fromarray(processed_img), lang="eng", builder=builder)
            return result if result.strip() else None, "OCRの結果が空です"
    except Exception as e:
        return None, f"OCRエラー: {e}"

def call_llama3_api(text, api_key):
    """Llama3のAPIを呼び出してテキストを改善する"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""以下のOCRで読み取ったテキストを、より読みやすく正確な日本語に修正してください。
OCRの結果には誤認識や文字化け、不要な記号やスペースが含まれている可能性があります。
元のテキスト: {text}

修正されたテキストのみを出力してください。説明は不要です。"""
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            return text, f"APIエラー: {response.status_code}"
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        return text, f"API呼び出しエラー: {e}"

def load_api_key():
    """APIキーを読み込み"""
    try:
        with open("api.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    if file and allowed_file(file.filename):
        # ユニークなファイル名を生成
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        try:
            # OCRを実行
            ocr_result, ocr_error = perform_ocr(file_path)
            
            if ocr_error:
                return jsonify({
                    'error': f'OCRエラー: {ocr_error}',
                    'ocr_result': ocr_result or ''
                }), 400
            
            # LLMで改善
            api_key = load_api_key()
            if api_key:
                improved_result, llm_error = call_llama3_api(ocr_result, api_key)
                if llm_error:
                    improved_result = ocr_result  # エラーの場合は元のテキストを使用
            else:
                improved_result = ocr_result
                llm_error = "APIキーが見つかりません"
            
            # 処理された画像を保存
            processed_img = process_image_for_ocr(file_path)
            if processed_img is not None:
                processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{unique_filename}")
                cv2.imwrite(processed_path, processed_img)
            
            return jsonify({
                'success': True,
                'ocr_result': ocr_result,
                'improved_result': improved_result,
                'llm_error': llm_error,
                'processed_image': f"processed_{unique_filename}" if processed_img is not None else None
            })
            
        except Exception as e:
            return jsonify({'error': f'処理エラー: {str(e)}'}), 500
        
        finally:
            # アップロードファイルを削除
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({'error': '無効なファイル形式です'}), 400

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
