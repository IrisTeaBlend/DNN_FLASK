from flask import Flask, render_template, request, send_file, abort
import os
import ttslearn
from ttslearn.dnntts import DNNTTS
import wave
import io

app = Flask(__name__)

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

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/synthesize/", methods=["POST"])
def synthesize_text():
    text = request.form.get("text")
    if text:
        # テキストを音声に変換
        wav, sr = engine.tts(text)

        # 音声データをバイトストリームに保存
        output_io = io.BytesIO()
        with wave.open(output_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sr)
            wf.writeframes(wav.tobytes())

        # バイトストリームのポインタを先頭に戻す
        output_io.seek(0)

        # ストリームレスポンスを返す
        return send_file(output_io, as_attachment=True, download_name="output.wav", mimetype="audio/wav")

    abort(400, description="テキストが提供されていません")

if __name__ == "__main__":
    app.run(debug=True)
