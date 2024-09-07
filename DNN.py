from ttslearn.dnntts import DNNTTS
from IPython.display import Audio #追加
import librosa.display #追加
import matplotlib.pyplot as plt #追加
import numpy as np #追加

# 継続長モデルの設定ファイル名
duration_config_name="duration_dnn"
# 音響モデルの設定ファイル名
acoustic_config_name="acoustic_dnn_sr16k"

# パッケージングしたモデルのパスを指定します
# model_dir = f"./tts_models/jsut_sr16000_{duration_config_name}_{acoustic_config_name}"
model_dir = f"C:/kikagaku/tts_models/jsut_sr16000_{duration_config_name}_{acoustic_config_name}" #修正
engine = DNNTTS(model_dir)
wav, sr = engine.tts("ここまでお読みいただき、ありがとうございました。")

# fig, ax = plt.subplots(figsize=(8,2))
# librosa.display.waveshow(wav.astype(np.float32), sr, ax=ax)
# librosa.display.waveshow(wav.astype(np.float32), sr=sr, ax=ax) # 修正
# ax.set_xlabel("Time [sec]")
# ax.set_ylabel("Amplitude")
# plt.tight_layout()

Audio(wav, rate=sr)