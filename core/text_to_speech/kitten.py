import os
from typing import Final

import soundfile as sf
from kittentts import KittenTTS

from .base import BaseTextToSpeechModel

DEFAULT_KITTEN_VOICE: Final[str] = "expr-voice-2-f"


class KittenTextToSpeech(BaseTextToSpeechModel):

    def __init__(
        self,
        voice: str = "expr-voice-2-f",
        model: str = "KittenML/kitten-tts-nano-0.2"
    ):
        super().__init__(voice)

        self._sess = KittenTTS(model)

    def synthesize_to_wav(self, text: str, output_filename: str | os.PathLike):
        audio = self._sess.generate(text, self._voice)
        sf.write(output_filename, audio, 24_000)
