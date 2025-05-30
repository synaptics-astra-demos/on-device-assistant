import json
from pathlib import Path
from typing import Final

from tqdm import tqdm

from core.text_to_speech import TextToSpeechAgent
from core.utils.download import download_from_url, download_from_hf

DATA_DIR: Final = Path("./data")
MODELS_DIR: Final = Path("./models")


if __name__ == "__main__":
    YELLOW: Final = "\033[93m"
    GREEN: Final = "\033[32m"
    CYAN: Final = "\033[36m"
    RESET: Final = "\033[0m"

    print(CYAN + "Downloading models..." + RESET)
    # download MiniLM models
    download_from_url(
        url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/all-MiniLM-L6-v2-Q8_0.gguf",
        filename=MODELS_DIR / f"gguf/all-MiniLM-L6-v2-Q8_0.gguf"
    )
    download_from_url(
        url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/all-MiniLM-L6-v2.synap",
        filename=MODELS_DIR / f"synap/all-MiniLM-L6-v2.synap"
    )
    # download Moonshine models
    download_from_hf(
        repo_id="UsefulSensors/moonshine",
        filename=f"onnx/merged/base/quantized/encoder_model.onnx",
    )
    download_from_hf(
        repo_id="UsefulSensors/moonshine",
        filename=f"onnx/merged/base/quantized/decoder_model_merged.onnx",
    )
    download_from_hf(
        repo_id="UsefulSensors/moonshine-base", 
        filename="config.json"
    )
    download_from_hf(
        repo_id="UsefulSensors/moonshine-base", 
        filename="tokenizer.json"
    )
    # download piper-tts models
    download_from_hf(
        repo_id="rhasspy/piper-voices",
        filename="en/en_US/lessac/low/en_US-lessac-low.onnx"
    )
    download_from_hf(
        repo_id="rhasspy/piper-voices",
        filename="en/en_US/lessac/low/en_US-lessac-low.onnx.json"
    )
    print(GREEN + "Downloads complete." + RESET)

    print(CYAN + "Generating TTS cache..." + RESET)
    tts = TextToSpeechAgent()
    for qa_file in DATA_DIR.glob("qa*.json"):
        with open(qa_file, "r") as f:
            answers = [pair["answer"] for pair in json.load(f)]
            for answer in tqdm(answers, desc=qa_file.name):
                tts.synthesize(answer)
    print(GREEN + "TTS cache generation complete." + RESET)
