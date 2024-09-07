from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import os
import ttslearn
from ttslearn.dnntts import DNNTTS
import wave

app = FastAPI()

# テンプレートと静的ファイルの設定
templates = Jinja2Templates(directory="templates")

# 継続長モデルの設定ファイル名
duration_config_name = "duration_dnn"
# 音響モデルの設定ファイル名
acoustic_config_name = "acoustic_dnn_sr16k"

# パッケージングしたモデルのパスを指定します
model_dir = f"C:/kikagaku/tts_models/jsut_sr16000_{duration_config_name}_{acoustic_config_name}"
engine = DNNTTS(model_dir)

# ファイルを保存するディレクトリ
upload_dir = "./uploads"
os.makedirs(upload_dir, exist_ok=True)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/synthesize/")
async def synthesize_text(request: Request, text: str = Form(...)):
    if text:
        wav, sr = engine.tts(text)

        # 音声ファイルを一時的に保存
        output_path = os.path.join(upload_dir, "output.wav")
        with wave.open(output_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(wav.tobytes())

        # 保存した音声ファイルをストリーム化してレスポンスとして返す
        return StreamingResponse(open(output_path, "rb"), media_type="audio/wav")

    raise HTTPException(status_code=400, detail="テキストが提供されていません")
