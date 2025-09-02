import os
import hashlib
import logging
from typing import Literal

from ..utils.audio import AudioManager

logger = logging.getLogger(__name__)


def tts_factory(model: Literal["piper", "kitten"], voice: str):
    from .piper import PiperOnnx

    if model == "piper":
        return PiperOnnx(voice)
    else:
        raise NotImplementedError()


class TextToSpeechAgent:

    def __init__(
        self,
        tts_model: Literal["piper", "kitten"] = "piper",
        tts_voice: str = "en_US-lessac-low",
        output_dir: str = "output",
        audio_manager: AudioManager | None = None
    ):
        self.tts = tts_factory(tts_model, tts_voice)
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_manager = audio_manager or AudioManager()

    @staticmethod
    def file_checksum(content: str, hash_length: int = 16) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:hash_length]

    def synthesize(self, text: str, output_filename: str = None, play_audio: bool = False) -> str:
        if output_filename is None:
            chk = self.file_checksum(self.tts.voice + text)
            output_filename = os.path.join(self.output_dir, f"speech-output-{chk}.wav")

        if os.path.exists(output_filename):
            logger.debug("Found TTS cache at '%s'", output_filename)
        else:
            self.tts.synthesize_to_wav(text, output_filename)
            logger.debug("Cached TTS to '%s'", output_filename)

        if play_audio:
            if self.audio_manager.device:
                self.audio_manager.play(output_filename)
            else:
                logger.warning("Skipping audio playback, no valid playback device in audio manager")
        return output_filename
