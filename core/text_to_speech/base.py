import os
from abc import ABC, abstractmethod


class BaseTextToSpeechModel(ABC):

    def __init__(
        self,
        voice: str
    ):
        self._voice = voice

    @property
    def voice(self):
        return self._voice

    @abstractmethod
    def synthesize_to_wav(self, text: str, output_filename: str | os.PathLike): ...
