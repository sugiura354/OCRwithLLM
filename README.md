# OCR + LLM テキスト改善システム

このシステムは、画像から文字を認識し、LLM（Llama3）を使用して認識結果を改善するツールです。

## ファイル構成

- `ocr_recognition.py`: 画像から文字認識を行う
- `llm_text_improvement.py`: LLMを使用してテキストを改善する
- `app.py`: 上2つをweb上で実行するためのバックエンド
- `templetes/index.html` : web表示用htmlファイル
- `api.txt`: Groq APIキーを保存するファイル, 各自で作成してください
- `requirements.txt`: 必要なPythonライブラリ

## セットアップ

1. 必要なライブラリをインストール:
```bash
pip install -r requirements.txt
```

2. Tesseract-OCRをインストール:
   - Windows: [Tesseract-OCR公式サイト](https://github.com/UB-Mannheim/tesseract/wiki)からダウンロード
   - 日本語データファイルも必要

3. Groq APIキーを取得して`api.txt`に保存

## 使用方法

### 1. 画像から文字認識のみ実行
```bash
python ocr_recognition.py <画像ファイル>
```
- 結果は`ocr_result.txt`に保存されます

### 2. OCR結果をLLMで改善
```bash
python llm_text_improvement.py
```
- `ocr_result.txt`から読み込んで改善

### 3. 指定ファイルから改善
```bash
python llm_text_improvement.py <ファイル名>
```

### 4. 直接テキストを改善
```bash
python llm_text_improvement.py --text "改善したいテキスト"
```

## 一連の処理を実行

1. 画像から文字認識:
```bash
python ocr_recognition.py dst.jpg
```

2. 認識結果を改善:
```bash
python llm_text_improvement.py
```

## web上で実行:
```bash
python app.py
```
- `(http://localhost:5000/)`にアクセス
## 出力ファイル

- `processed_image.jpg`: 処理された画像
- `ocr_result.txt`: OCRの生の結果
- `improved_result.txt`: LLMで改善された結果

## 注意事項

- Groq APIキーが必要
- インターネット接続が必要
- 大量のテキスト処理には時間がかかる場合がある
