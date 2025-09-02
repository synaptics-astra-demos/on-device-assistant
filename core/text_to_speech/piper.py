import argparse
import os
import wave

from piper import PiperVoice

from .base import BaseTextToSpeechModel
from ..utils.download import download_from_hf


class PiperOnnx(BaseTextToSpeechModel):

    def __init__(
        self,
        voice: str | None = "en_US-lessac-low",
        model: str | os.PathLike | None = None
    ):
        super().__init__(voice)

        if not model:
            lang = self._voice.split("_")[0]
            model_dir = lang + "/" + "/".join(voice.split("-"))
            model_file = model_dir + "/" + f"{self._voice}.onnx"
            model = download_from_hf(
                repo_id="rhasspy/piper-voices",
                filename=model_file
            )
            download_from_hf(
                repo_id="rhasspy/piper-voices",
                filename=f"{model_file}.json"
            )
        self._sess = PiperVoice.load(model)

    def synthesize_to_wav(self, text: str, output_filename: str | os.PathLike):
        with wave.open(output_filename, "wb") as wav_file:
            self._sess.synthesize_wav(text, wav_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Piper Text-to-Speech command line tool."
    )
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default="en_US-lessac-low",
        help="Voice in format LOCALE-VOICENAME-STYLE (default: %(default)s)",
    )
    args = parser.parse_args()

    tts = PiperOnnx(args.voice)
    tts.synthesize_to_wav(args.text, args.output)
    print(f"Audio written to: {args.output}")
