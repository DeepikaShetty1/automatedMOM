"""
This module provides utility functions for audio processing,
primarily for converting audio files to a format suitable for NeMo.
It uses FFmpeg for audio format conversion.
"""
import os
import subprocess
import logging
from typing import Tuple, Dict, Any, Optional

# Get a logger for this module
logger = logging.getLogger(__name__)

def convert_audio_to_wav(
    input_audio_path: str,
    output_dir: str,
    original_filename: str
) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[int]]:
    """
    Converts an audio file to WAV format using FFmpeg.
    Ensures the WAV is 16kHz sample rate, single channel.

    Args:
        input_audio_path (str): The full path to the input audio file.
        output_dir (str): The directory where the converted WAV file should be saved.
        original_filename (str): The original filename, used for naming the output WAV.

    Returns:
        Tuple[Optional[str], Optional[Dict[str, str]], Optional[int]]:
        - Full path to the converted WAV file if successful.
        - Error dictionary if conversion fails.
        - HTTP status code for the error.
    """
    # Create a unique name for the output WAV file
    output_wav_filename = f"{os.path.splitext(original_filename)[0]}.wav"
    output_wav_path = os.path.join(output_dir, output_wav_filename)

    # FFmpeg command to convert to 16kHz WAV, mono channel
    # -i: input file
    # -ar 16000: set audio sample rate to 16kHz
    # -ac 1: set audio channels to 1 (mono)
    # -y: overwrite output file without asking
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_wav_path
    ]

    logger.info(f"Attempting to convert '{input_audio_path}' to WAV at '{output_wav_path}' using FFmpeg.")
    try:
        # Run FFmpeg as a subprocess
        result = subprocess.run(
            ffmpeg_command,
            check=True,  # Raise CalledProcessError for non-zero exit codes
            capture_output=True,
            text=True
        )
        logger.info(f"FFmpeg conversion successful for '{original_filename}'. Output:\n{result.stdout}")
        return output_wav_path, None, None # Success: return path, no error
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"FFmpeg conversion failed for '{original_filename}'. "
            f"Command: {' '.join(e.cmd)}\n"
            f"Stdout: {e.stdout}\n"
            f"Stderr: {e.stderr}"
        )
        logger.error(error_msg)
        return None, {"error": f"Audio conversion failed for '{original_filename}'. Details: {e.stderr.strip()}. Please check if the audio file is valid or if FFmpeg is correctly installed and in PATH."}, 500
    except FileNotFoundError:
        error_msg = (
            "FFmpeg not found. Please ensure FFmpeg is installed on your system "
            "and its executable directory is added to your system's PATH environment variable."
        )
        logger.error(error_msg)
        return None, {"error": "FFmpeg is not installed or not found in system PATH. Please install FFmpeg to process audio files."}, 500
    except Exception as e:
        error_msg = f"An unexpected error occurred during audio conversion: {str(e)}"
        logger.exception(error_msg)
        return None, {"error": f"An unexpected internal error occurred during audio conversion for '{original_filename}'. Details: {str(e)}"}, 500

