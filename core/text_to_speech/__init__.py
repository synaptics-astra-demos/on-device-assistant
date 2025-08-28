import os
import argparse
import hashlib
import logging
import signal
import subprocess
import sys

from ..utils.audio import AudioManager
from ..utils.download import download_from_hf

logger = logging.getLogger(__name__)


def handle_sigint(signum, frame):
    logger.info("Ctrl+C detected, exiting...")
    sys.exit(130)


class TextToSpeechAgent:
    def __init__(self, voice="en_US-lessac-low", output_dir="output", audio_manager: AudioManager | None = None):
        # Voice must be in the format LOCALE-VOICENAME-STYLE, e.g. en_US-lessac-low
        parts = voice.split("-")
        if len(parts) != 3:
            raise ValueError(
                "Voice must be in format LOCALE-VOICENAME-STYLE, e.g. en_US-lessac-low"
            )
        locale, voice_name, style = parts
        short_lang = locale.split("_")[0] if "_" in locale else locale[:2].lower()
        onnx_file_path = f"{short_lang}/{locale}/{voice_name}/{style}/{voice}.onnx"
        json_file_path = f"{short_lang}/{locale}/{voice_name}/{style}/{voice}.onnx.json"
        self.onnx_file = download_from_hf(
            repo_id="rhasspy/piper-voices", filename=onnx_file_path
        )
        self.json_file = download_from_hf(
            repo_id="rhasspy/piper-voices", filename=json_file_path
        )
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_manager = audio_manager or AudioManager()

    @staticmethod
    def file_checksum(content: str, hash_length: int = 16) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:hash_length]

    def synthesize(self, text: str, output_filename: str = None, play_audio: bool = False) -> str:
        if output_filename is None:
            chk = self.file_checksum(self.onnx_file + text)
            output_filename = os.path.join(self.output_dir, f"speech-output-{chk}.wav")

        if os.path.exists(output_filename):
            logger.debug(f"Found TTS cache at '{output_filename}'")
            return output_filename

        old_hdlr = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handle_sigint)
        echo_proc = subprocess.Popen(
            ['echo', text],
            stdout=subprocess.PIPE
        )
        piper_proc = subprocess.Popen(
            ['piper', '--model', self.onnx_file, '--output_file', output_filename],
            stdin=echo_proc.stdout
        )
        echo_proc.stdout.close()  # Allow echo_proc to receive SIGPIPE if piper exits
        piper_proc.wait()
        echo_proc.wait()

        logger.debug(f"Caching TTS to '{output_filename}'")
        signal.signal(signal.SIGINT, old_hdlr)

        if play_audio:
            if self.audio_manager.device:
                self.audio_manager.play(output_filename)
            else:
                logger.warning("Skipping audio playback, no valid playback device in audio manager")

        return output_filename


def main():
    parser = argparse.ArgumentParser(
        description="Piper Text-to-Speech command line tool."
    )
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default="en_US-lessac-low",
        help="Voice in format LOCALE-VOICENAME-STYLE (default: en_US-lessac-low)",
    )
    args = parser.parse_args()

    tts = TextToSpeechAgent(voice=args.voice)
    wav_path = tts.synthesize(args.text)
    print(f"Audio written to: {wav_path}")


if __name__ == "__main__":
    main()