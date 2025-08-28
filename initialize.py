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
    # download MiniLM
    download_from_url(
        url="https://github.com/synaptics-astra-demos/on-device-assistant/releases/download/models-v1/all-MiniLM-L6-v2-quantized.synap",
        filename=MODELS_DIR / f"synap/all-MiniLM-L6-v2/model_quantized.synap"
    )
    # download Moonshine
    download_from_hf(
        repo_id="UsefulSensors/moonshine",
        filename=f"onnx/merged/base/float/encoder_model.onnx",
    )
    download_from_hf(
        repo_id="UsefulSensors/moonshine",
        filename=f"onnx/merged/base/float/decoder_model_merged.onnx",
    )
    download_from_hf(
        repo_id="UsefulSensors/moonshine-tiny", 
        filename="config.json"
    )
    download_from_hf(
        repo_id="UsefulSensors/moonshine-tiny", 
        filename="tokenizer.json"
    )
    # download piper-tts
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
    qa_file = DATA_DIR / "qa_dishwasher.json"
    with open(qa_file, "r") as f:
        answers = [pair["answer"] for pair in json.load(f)]
        for answer in tqdm(answers, desc=qa_file.name):
            tts.synthesize(answer)
    print(GREEN + "TTS cache generation complete." + RESET)
