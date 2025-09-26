import requests
import json
import sys
import os

def call_llama3_api(text, api_key):
    """Llama3のAPIを呼び出してテキストを改善する"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # OCRの結果を改善するためのプロンプト
    prompt = f"""以下のOCRで読み取ったテキストを、より読みやすく正確な日本語に修正してください。
OCRの結果には誤認識や文字化けが含まれている可能性があります。
元のテキスト: {text}

修正されたテキストのみを出力してください。説明は不要です。"""
    
    data = {
        "model": "llama-3.1-8b-instant",  # 利用可能なモデルに変更
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
        # print(f"API呼び出し中... テキスト長: {len(text)}文字")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        # print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code != 200:
            # print(f"エラーレスポンス: {response.text}")
            return text
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Llama3 API エラー: {e}")
        return text  # エラーの場合は元のテキストを返す

def load_api_key():
    """APIキーを読み込み"""
    try:
        with open("api.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("api.txtファイルが見つかりません")
        return None

def improve_text_from_file(input_file="ocr_result.txt"):
    """ファイルからテキストを読み込んで改善"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_file}' が見つかりません")
        return None
    
    if not text:
        print("エラー: ファイルが空です")
        return None
    
    api_key = load_api_key()
    if not api_key:
        return None
    
    # print("=== 元のテキスト ===")
    # print(text)
    # print("\n=== Llama3で改善された結果 ===")
    
    improved_result = call_llama3_api(text, api_key)
    print(improved_result)
    
    # 改善された結果をファイルに保存
    with open("improved_result.txt", "w", encoding="utf-8") as f:
        f.write(improved_result)
    # print(f"\n改善された結果を 'improved_result.txt' に保存しました")
    
    return improved_result

def improve_text_direct(text):
    """直接テキストを改善"""
    api_key = load_api_key()
    if not api_key:
        return None
    
    print("=== 元のテキスト ===")
    print(text)
    print("\n=== Llama3で改善された結果 ===")
    
    improved_result = call_llama3_api(text, api_key)
    print(improved_result)
    
    return improved_result

def main():
    if len(sys.argv) == 1:
        # ファイルから読み込み
        improve_text_from_file()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("使用方法:")
            print("  python llm_text_improvement.py                    # ocr_result.txtから読み込み")
            print("  python llm_text_improvement.py <ファイル名>        # 指定ファイルから読み込み")
            print("  python llm_text_improvement.py --text 'テキスト'  # 直接テキストを指定")
        elif sys.argv[1].startswith("--text "):
            # 直接テキストを指定
            text = sys.argv[1][7:]  # "--text " を除去
            improve_text_direct(text)
        else:
            # ファイル名を指定
            improve_text_from_file(sys.argv[1])
    else:
        print("使用方法: python llm_text_improvement.py [ファイル名|--text 'テキスト']")
        print("詳細: python llm_text_improvement.py --help")

if __name__ == "__main__":
    main()
