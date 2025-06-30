import sys

import numpy as np
import soundfile as sf

from core.speech_to_text.moonshine import MoonshineOnnx, MoonshineSynap


def transcribe(audio: np.ndarray, runner: MoonshineOnnx | MoonshineSynap):
    result = runner.transcribe(audio)
    print(f"Transcribed ({runner.last_infer_time * 1000:.2f} ms): \"{result}\"\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_moonshine.py <audio_file>")
        sys.exit(1)
    audio_path = sys.argv[1]
    data, sr = sf.read(audio_path, dtype="float32")
    data = data[:80000]
    print(f"Input: {data.shape[-1]} samples, {sr} Hz")
    
    stt_onnx = MoonshineOnnx()
    print(f"Transcribing with ONNX backend...")
    transcribe(data, stt_onnx)

    stt_synap = MoonshineSynap()
    print(f"Transcribing with SyNAP backend...")
    transcribe(data, stt_synap)
