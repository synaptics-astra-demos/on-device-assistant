import logging
import time
from typing import Any, Callable

import numpy as np

from silero_vad import VADIterator, load_silero_vad

from .moonshine import MoonshineOnnx, MoonshineSynap
from ..utils.audio import AudioManager

SAMPLING_RATE = 16000
CHUNK_SIZE = 512
LOOKBACK_CHUNKS = 7
MAX_LINE_LENGTH = 80
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.5

logger = logging.getLogger(__name__)


def moonshine_factory(model_size: str, sampling_rate: int, cpu_only: bool = False, n_threads: int | None = None) -> MoonshineOnnx:
    if cpu_only:
        return MoonshineOnnx(model_size=model_size, rate=sampling_rate, n_threads=n_threads)
    return MoonshineSynap(model_size=model_size, rate=sampling_rate)


class SpeechToTextAgent:
    def __init__(
        self, 
        model_size: str, 
        handler: Callable[[str], Any], 
        cpu_only: bool = False, 
        n_threads: int | None = None, 
        threshold: float = 0.3,
        min_silence_duration_ms: int = 300,
    ):
        self.handler = handler

        self.speech_to_text = moonshine_factory(model_size, SAMPLING_RATE, cpu_only, n_threads)
        self.vad_model = load_silero_vad(onnx=True)
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        self.audio_manager = AudioManager()
        self.caption_cache = []
        self.lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
        self.speech = np.empty(0, dtype=np.float32)
        self.recording = False

    @property
    def last_infer_time(self) -> float | None:
        return self.speech_to_text.last_infer_time

    def print_captions(self, text):
        if len(text) < MAX_LINE_LENGTH:
            for cap in self.caption_cache[::-1]:
                text = cap + " " + text
                if len(text) > MAX_LINE_LENGTH:
                    break
        if len(text) > MAX_LINE_LENGTH:
            text = text[-MAX_LINE_LENGTH:]
        print("\r" + (" " * MAX_LINE_LENGTH) + "\r" + text, end="", flush=True)

    def soft_reset(self):
        self.vad_iterator.triggered = False
        self.vad_iterator.temp_end = 0
        self.vad_iterator.current_sample = 0

    def end_recording(self):
        text = self.speech_to_text.transcribe(self.speech)

        if text.strip():
            self.handler(text)

        self.speech *= 0.0

    def run(self):
        try:
            self.audio_manager.start_record(chunk_size=CHUNK_SIZE)
            print("Press Ctrl+C to quit speech-to-text.\n")
            start_time = time.time()
            call_end_recording = False

            for chunk in self.audio_manager.read(chunk_size=CHUNK_SIZE):
                if call_end_recording:
                    self.end_recording()
                    call_end_recording = False

                self.speech = np.concatenate((self.speech, chunk))
                if not self.recording:
                    self.speech = self.speech[-self.lookback_size :]
                speech_dict = self.vad_iterator(chunk)

                if speech_dict:
                    if "start" in speech_dict and not self.recording:
                        self.recording = True
                        start_time = time.time()
                    if "end" in speech_dict and self.recording:
                        call_end_recording = True
                        self.recording = False
                elif self.recording:
                    if (len(self.speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        call_end_recording = True
                        self.recording = False
                        self.soft_reset()
                    elif (time.time() - start_time) > MIN_REFRESH_SECS:
                        start_time = time.time()

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            self.audio_manager.stop_record()


if __name__ == "__main__":
    # Example usage:
    def handle_results(text, inference_time):
        if text:
            print(f"\033[93mSTT: {text} \033[92m({inference_time*1000:.0f}ms)\033[0m")

    pipe = SpeechToTextAgent(
        model="base",  # Set to "tiny" for faster but less accurate model
        handler=handle_results
    )

    pipe.run()