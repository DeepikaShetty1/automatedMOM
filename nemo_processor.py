"""
This module handles the core NeMo processing logic for the speaker diarization API.
It includes functions for loading NeMo models at application startup and
processing individual audio files for diarization and ASR.
"""
import os
import json
import uuid
import shutil
import logging
from typing import Optional, Tuple, Dict, Any, List, cast
import wave
import torch
from torch import nn


# --- NeMo Imports ---
# Using type: ignore to suppress Pylance/Pylint warnings if NeMo is not installed
# in the analysis environment. These imports must be at the top of the module.
from omegaconf import DictConfig, OmegaConf # type: ignore
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import \
    ASRDecoderTimeStamps # type: ignore
from nemo.collections.asr.parts.utils.diarization_utils import \
    OfflineDiarWithASR, rttm_to_labels # type: ignore # <--- THIS LINE IS CRUCIAL: rttm_to_labels is now imported

# Import audio_utils at the top level for Pylint C0415 (import-outside-toplevel)
from audio_utils import convert_audio_to_wav

# Get a logger for this module
logger = logging.getLogger(__name__)

# Global variables for the NeMo configuration object and initialized components
# These are initialized once at application startup.
nemo_cfg: Optional[DictConfig] = None
asr_decoder_ts_instance: Optional[ASRDecoderTimeStamps] = None
asr_model_instance: Optional[nn.Module] = None
asr_diar_offline_instance: Optional[OfflineDiarWithASR] = None

# Define the path to your NeMo configuration YAML file
YAML_CONFIG_PATH = "diar_infer_meeting.yaml"

# Global variable to store the temporary directory created during initialization
_init_temp_dir: Optional[str] = None

# Define model paths and parameters (can be overridden by config)
DEFAULT_SPEAKER_EMBEDDINGS_MODEL = 'SpeakerVerification_titanet_large'
DEFAULT_VAD_MODEL = 'vad_multilingual_marblenet'
DEFAULT_ASR_MODEL = 'QuartzNet15x5Base-En'
DEFAULT_WORD_TS_ANCHOR_OFFSET = 0.03


def _create_dummy_audio_files(temp_dir: str) -> Tuple[str, str, str]:
    """
    Helper to create dummy audio and manifest files for NeMo initialization.
    These are minimal files to satisfy NeMo's internal checks during model loading.
    """
    dummy_audio_filepath = os.path.join(temp_dir, "init_dummy_audio.wav")
    dummy_manifest_filepath = os.path.join(temp_dir, "init_dummy_manifest.json")
    dummy_out_dir = os.path.join(temp_dir, "init_dummy_output_dir")
    os.makedirs(dummy_out_dir, exist_ok=True)

    # Create a minimal, empty WAV file.
    # This creates a valid, empty WAV header.
    # Pylint E1101 (no-member) is disabled here as wave.open in 'wb' mode
    # dynamically adds these methods, which Pylint might not infer.
    with wave.open(dummy_audio_filepath, 'wb') as wav_file: # pylint: disable=no-member
        wav_file.setnchannels(1)  # Mono # pylint: disable=no-member
        wav_file.setsampwidth(2)  # 16-bit # pylint: disable=no-member
        wav_file.setframerate(16000) # 16 kHz # pylint: disable=no-member
        # No frames written, so it's a 0-duration WAV

    logger.info("Created empty dummy WAV file at: %s", dummy_audio_filepath)

    # Create a minimal, valid JSON manifest file referencing the dummy audio.
    minimal_manifest_data = [
        {
            "audio_filepath": dummy_audio_filepath,
            "offset": 0,
            "duration": 0.0,
            "label": "init_dummy",
            "text": "-",
            "num_speakers": 1
        }
    ]
    with open(dummy_manifest_filepath, 'w', encoding='utf-8') as f:
        for line in minimal_manifest_data:
            f.write(json.dumps(line) + '\n')
    logger.info("Created dummy manifest file at: %s", dummy_manifest_filepath)

    return dummy_audio_filepath, dummy_manifest_filepath, dummy_out_dir


def _load_nemo_config_and_set_defaults(
    yaml_config_path: str,
    dummy_manifest_filepath: str,
    dummy_out_dir: str
) -> DictConfig:
    """Helper to load NeMo config and apply default model paths/parameters."""
    if not os.path.exists(yaml_config_path):
        raise FileNotFoundError(
            f"Required NeMo config file not found: {yaml_config_path}. "
            "Please ensure it is in the same directory as app.py."
        )

    logger.info("Loading configuration from: %s", yaml_config_path)
    temp_cfg = OmegaConf.load(yaml_config_path)

    if not isinstance(temp_cfg, DictConfig) or 'diarizer' not in temp_cfg:
        raise ValueError(
            f"Loaded config from {yaml_config_path} is not a valid DictConfig "
            "or missing the 'diarizer' key. Check your YAML structure."
        )

    logger.info("Applying hardcoded model path overrides and parameters.")
    temp_cfg.diarizer.speaker_embeddings.model_path = DEFAULT_SPEAKER_EMBEDDINGS_MODEL
    temp_cfg.diarizer.vad.model_path = DEFAULT_VAD_MODEL
    temp_cfg.diarizer.asr.model_path = DEFAULT_ASR_MODEL

    if 'parameters' not in temp_cfg.diarizer.asr:
        temp_cfg.diarizer.asr.parameters = OmegaConf.create({})
    temp_cfg.diarizer.asr.parameters.asr_based_vad = True

    if 'parameters' not in temp_cfg.diarizer.speaker_embeddings:
        temp_cfg.diarizer.speaker_embeddings.parameters = OmegaConf.create({})
    temp_cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False

    # Assign dummy paths for initialization
    logger.info(
        "Assigning temporary manifest_filepath and out_dir for NeMo component "
        "initialization."
    )
    temp_cfg.diarizer.manifest_filepath = dummy_manifest_filepath
    temp_cfg.diarizer.out_dir = dummy_out_dir
    
    return temp_cfg


def _instantiate_nemo_components(
    cfg: DictConfig
) -> Tuple[ASRDecoderTimeStamps, nn.Module, OfflineDiarWithASR]:
    """Helper to instantiate ASRDecoderTimeStamps, ASR model, and OfflineDiarWithASR."""
    asr_decoder_ts = ASRDecoderTimeStamps(cfg_diarizer=cfg.diarizer) # type: ignore
    assert asr_decoder_ts is not None, "Failed to instantiate ASRDecoderTimeStamps."

    # Explicitly cast the result to nn.Module for type checking
    asr_model = cast(nn.Module, asr_decoder_ts.set_asr_model())

    asr_diar_offline = OfflineDiarWithASR(cfg_diarizer=cfg.diarizer) # type: ignore
    assert asr_diar_offline is not None, "Failed to instantiate OfflineDiarWithASR."

    asr_diar_offline.word_ts_anchor_offset = (
        asr_decoder_ts.word_ts_anchor_offset
        if asr_decoder_ts is not None else 0.0
    )
    logger.info("NeMo components instantiated successfully.")
    return asr_decoder_ts, asr_model, asr_diar_offline


def load_nemo_components():
    """
    Loads the NeMo components and models. This function is called once
    when the Flask application starts to pre-load heavy ML models.
    It loads configuration directly from 'diar_infer_meeting.yaml'
    and provides genuinely existing, minimal temporary files for initialization
    to satisfy NeMo's strict requirements.
    """
    # W0603: Using the global statement - necessary for singleton pattern in Flask app context.
    global nemo_cfg, asr_decoder_ts_instance, asr_model_instance, \
        asr_diar_offline_instance, _init_temp_dir

    logger.info("Attempting to load NeMo components...")

    _init_temp_dir = None
    try:
        _init_temp_dir = os.path.join("/tmp", f"nemo_init_temp_{uuid.uuid4()}")
        os.makedirs(_init_temp_dir, exist_ok=True)
        logger.info("Created temporary directory for initialization: %s", _init_temp_dir)

        dummy_audio_filepath, dummy_manifest_filepath, dummy_out_dir = \
            _create_dummy_audio_files(_init_temp_dir)

        temp_cfg = _load_nemo_config_and_set_defaults(
            YAML_CONFIG_PATH, dummy_manifest_filepath, dummy_out_dir
        )
        nemo_cfg = temp_cfg
        logger.info("Successfully loaded configuration from YAML.")

        (asr_decoder_ts_instance, asr_model_instance, asr_diar_offline_instance) = \
            _instantiate_nemo_components(nemo_cfg)

        logger.info("NeMo components loaded successfully.")
    except Exception as e: # W0718: Catching too general exception Exception (broad-exception-caught)
        # Catching broad Exception here is acceptable for top-level initialization
        # to prevent application startup failure due to model loading issues.
        logger.exception(
            "Failed to load NeMo components during initialization. Details: %s",
            e # pylint: disable=W0612 # W0612: Unused variable 'e'
        )
        nemo_cfg = None
        asr_decoder_ts_instance = None
        asr_model_instance = None
        asr_diar_offline_instance = None
    finally:
        if _init_temp_dir and os.path.exists(_init_temp_dir):
            shutil.rmtree(_init_temp_dir)
            logger.info(
                "Cleaned up temporary initialization directory: %s",
                _init_temp_dir
            )
            _init_temp_dir = None


def _save_and_validate_audio(
    audio_file_stream: Any, original_filename: str, temp_session_dir: str, request_id: str
) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[int]]:
    """Helper to save and validate the uploaded audio file."""
    original_audio_path = os.path.join(temp_session_dir, original_filename)
    audio_file_stream.save(original_audio_path)
    logger.info("[%s] Saved uploaded audio to: %s", request_id, original_audio_path)

    if os.path.getsize(original_audio_path) == 0:
        logger.warning(
            "[%s] Uploaded audio file '%s' is empty.",
            request_id, original_filename
        )
        # Fix: Return types match the function signature: (None, dict, int)
        return None, {"error": "Uploaded audio file is empty. "
                      "Please upload a valid audio file with content."}, 400

    return original_audio_path, None, None


def _convert_audio(
    original_audio_path: str, temp_session_dir: str, original_filename: str, request_id: str
) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[int]]:
    """Helper to convert audio to WAV format using audio_utils."""
    # The `audio_utils.convert_audio_to_wav` function does NOT take `request_id`.
    # We keep `request_id` in this function's signature for logging purposes only.
    converted_path, error_response, status_code_conversion = convert_audio_to_wav(
        original_audio_path, temp_session_dir, original_filename # <--- Pass request_id here
    )

    if error_response:
        logger.error(
            "[%s] Audio conversion failed for '%s': %s",
            request_id, original_filename,
            error_response.get('error', 'Unknown conversion error')
        )
        return None, error_response, status_code_conversion

    if converted_path is None:
        logger.critical(
            "[%s] Audio conversion returned None path unexpectedly for '%s'. "
            "This indicates a severe internal logic error.",
            request_id, original_filename
        )
        return None, {"error": "Internal error: Audio conversion path is unexpectedly None."}, 500

    return converted_path, None, None


def _prepare_nemo_inputs(
    audio_path_for_nemo: str, temp_session_dir: str, request_id: str
) -> Tuple[str, str]:
    """Helper to prepare manifest and output directories for NeMo."""
    manifest_filepath = os.path.join(temp_session_dir, "audio_manifest.json")
    output_diar_dir = os.path.join(temp_session_dir, "diarization_output")
    os.makedirs(output_diar_dir, exist_ok=True)
    logger.info("[%s] Created output directory for NeMo: %s", request_id, output_diar_dir)

    manifest_data_content = [
        {
            "audio_filepath": audio_path_for_nemo,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None
        }
    ]
    with open(manifest_filepath, 'w', encoding='utf-8') as f:
        for line in manifest_data_content:
            f.write(json.dumps(line) + '\n')
    logger.info("[%s] Created manifest at: %s", request_id, manifest_filepath)
    return manifest_filepath, output_diar_dir


def _run_nemo_pipeline(
    manifest_filepath: str, output_diar_dir: str, request_id: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Helper to run NeMo ASR and Diarization pipeline."""
    # Assertions for runtime safety
    assert nemo_cfg is not None and isinstance(nemo_cfg, DictConfig), \
        "NeMo config is unexpectedly None or not a DictConfig."
    assert asr_decoder_ts_instance is not None, \
        "ASRDecoderTimeStamps instance is unexpectedly None."
    assert asr_diar_offline_instance is not None, \
        "OfflineDiarWithASR instance is unexpectedly None."
    assert asr_model_instance is not None, \
        "ASR model instance is unexpectedly None."

    # Create a COPY of the global config for this request to avoid modifying it
    request_cfg = nemo_cfg.copy()

    # Update the manifest_filepath and out_dir in the *request-specific* config
    logger.info(
        "[%s] Updated request-specific NeMo config with manifest: %s and out_dir: %s",
        request_id, manifest_filepath, output_diar_dir
    )
    request_cfg.diarizer.manifest_filepath = manifest_filepath
    request_cfg.diarizer.out_dir = output_diar_dir
    
    logger.info("[%s] Starting ASR inference...", request_id)
    with torch.no_grad():
        word_hyp, word_ts_hyp = asr_decoder_ts_instance.run_ASR(asr_model_instance) # type: ignore
    logger.info("[%s] ASR inference finished. Obtained %d words.", request_id, len(word_hyp))

    logger.info("[%s] Starting Diarization inference...", request_id)
    rttm_filepath, _ = asr_diar_offline_instance.run_diarization(request_cfg, word_ts_hyp) # type: ignore
    logger.info("[%s] Diarization inference finished.", request_id)

    # ADDED: Parse the RTTM file into the format expected by get_transcript_with_speaker_labels
    # rttm_to_labels returns a dictionary mapping audio_id to a list of labels (strings).
    # This import is now global at the top of the file, so it's not needed here.
    # Explicitly cast the output of rttm_to_labels to help Pylance with type inference.
    # The rttm_to_labels function returns a dictionary mapping audio_id to a list of labels (strings).
    parsed_diar_hyp = cast(Dict[str, List[str]], rttm_to_labels(rttm_filepath))
    logger.info("[%s] Combining transcript with speaker labels.", request_id)
    trans_info_dict = asr_diar_offline_instance.get_transcript_with_speaker_labels(parsed_diar_hyp, word_hyp, word_ts_hyp)
    logger.info("[%s] Transcript combined.", request_id)
    return trans_info_dict, output_diar_dir


def _process_word_data(
    word_data: List[Dict[str, Any]], request_id: str
) -> List[str]:
    """Helper to process and format word data into conversation lines."""
    formatted_conversation_lines: List[str] = []
    current_speaker: Optional[str] = None
    current_utterance_words: List[Dict[str, Any]] = []

    for entry in word_data:
        speaker = entry.get('speaker')
        word = entry.get('word')
        start_time = entry.get('start_time')
        end_time = entry.get('end_time')

        # Robust validation for entry data
        if not all(isinstance(val, (str, float, int)) for val in
                   [speaker, word, start_time, end_time]) or \
           speaker is None or word is None or start_time is None or \
           end_time is None:
            logger.warning(
                "[%s] Skipping malformed word entry (type mismatch or missing keys): %s",
                request_id, entry
            )
            continue

        if current_speaker is None or speaker != current_speaker:
            if current_utterance_words:
                full_text = " ".join([w['word'] for w in current_utterance_words])
                first_start = current_utterance_words[0]['start_time']
                last_end = current_utterance_words[-1]['end_time']
                formatted_conversation_lines.append(
                    f"{current_speaker}: ({first_start:.2f}-{last_end:.2f}s) "
                    f"{full_text.strip()}"
                )
            current_speaker = speaker
            current_utterance_words = [{'word': word, 'start_time': start_time,
                                        'end_time': end_time}]
        else:
            current_utterance_words.append({'word': word, 'start_time': start_time,
                                            'end_time': end_time})

    if current_utterance_words:
        full_text = " ".join([w['word'] for w in current_utterance_words])
        first_start = current_utterance_words[0]['start_time']
        last_end = current_utterance_words[-1]['end_time']
        formatted_conversation_lines.append(
            f"{current_speaker}: ({first_start:.2f}-{last_end:.2f}s) "
            f"{full_text.strip()}"
        )

    return formatted_conversation_lines


def _format_conversation_output(
    trans_info_dict: Dict[str, Any],
    audio_path_for_nemo: str,
    output_diar_dir: str,
    request_id: str
) -> List[str]:
    """Helper to format the NeMo output into readable conversation lines."""
    word_data_from_nemo: List[Dict[str, Any]] = []

    processed_audio_base_name = os.path.splitext(os.path.basename(audio_path_for_nemo))[0]

    # Attempt to get word data from trans_info_dict
    if trans_info_dict and isinstance(trans_info_dict, dict):
        if processed_audio_base_name in trans_info_dict:
            if 'words' in trans_info_dict[processed_audio_base_name] and \
               isinstance(trans_info_dict[processed_audio_base_name]['words'], list):
                word_data_from_nemo = cast(List[Dict[str, Any]],
                                           trans_info_dict[processed_audio_base_name]['words'])
        elif len(trans_info_dict) == 1 and list(trans_info_dict.values())[0] and \
             'words' in list(trans_info_dict.values())[0] and \
             isinstance(list(trans_info_dict.values())[0]['words'], list):
            word_data_from_nemo = cast(List[Dict[str, Any]],
                                       list(trans_info_dict.values())[0]['words'])

    if not word_data_from_nemo:
        logger.warning(
            "[%s] get_transcript_with_speaker_labels did not return expected word data. "
            "Attempting to read direct JSON output generated by run_diarization as fallback.",
            request_id
        )
        audio_base_name_for_json = os.path.splitext(os.path.basename(audio_path_for_nemo))[0]
        output_json_path = os.path.join(output_diar_dir, "pred_rttms",
                                        f"{audio_base_name_for_json}.json")

        if not os.path.exists(output_json_path):
            logger.error("[%s] Expected direct output JSON not found at %s.",
                         request_id, output_json_path)
            # Return an empty list to indicate no conversation could be formatted
            return ["Error: Diarization output file not found. This might indicate "
                    "an issue with NeMo processing."]

        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                diarization_results = json.load(f)
            logger.info("[%s] Successfully loaded fallback JSON from %s.",
                        request_id, output_json_path)
        except json.JSONDecodeError as jde:
            logger.error(
                "[%s] Failed to decode fallback JSON at %s: %s.",
                request_id, output_json_path, jde, exc_info=True
            )
            return ["Error: Failed to read diarization output. Invalid JSON format."]
        except Exception as e:
            # Catching broad Exception here is acceptable as a final fallback for file I/O
            logger.error(
                "[%s] Error reading fallback JSON at %s: %s.",
                 request_id, output_json_path, e,
                 exc_info=True
            )
            return ["Error: Error accessing diarization output."]

        if 'words' in diarization_results and isinstance(diarization_results['words'], list):
            word_data_from_nemo = cast(List[Dict[str, Any]], diarization_results['words'])
        elif isinstance(diarization_results, list):
            word_data_from_nemo = cast(List[Dict[str, Any]], diarization_results)
        else:
            logger.error(
                "[%s] Diarization output JSON has an unexpected structure or 'words' "
                "list is missing/maldformed. Check file: %s",
                request_id, output_json_path
            )
            return ["Error: Diarization output JSON has an unexpected structure. "
                    "'words' list is missing or malformed."]

    formatted_conversation_lines = _process_word_data(word_data_from_nemo, request_id)

    if not formatted_conversation_lines:
        logger.info(
            "[%s] No distinct speech detected or processed for the provided audio, "
            "resulting in empty formatted conversation.",
            request_id
        )
        formatted_conversation_lines.append("No distinct speech detected or processed for the "
                                            "provided audio.")

    return formatted_conversation_lines


def process_audio(
    audio_file_stream: Any, original_filename: str, temp_session_dir: str, request_id: str
) -> Tuple[Dict[str, Any], int]:
    """
    Processes the uploaded audio file using the NeMo diarization pipeline.
    This function mirrors the core logic of offline_diar_with_asr_infer.py.

    Args:
        audio_file_stream: The Flask FileStorage object containing the audio.
        original_filename (str): The original name of the uploaded file.
        temp_session_dir (str): The temporary directory for this specific request.
        request_id (str): A unique identifier for the current request, for logging.

    Returns:
        tuple: (dict_response, http_status_code)
    """
    # Step 1: Save and validate the uploaded audio
    audio_path_result, error_resp, status_code = _save_and_validate_audio(
        audio_file_stream, original_filename, temp_session_dir, request_id
    )
    if error_resp:
        return error_resp, status_code # type: ignore

    # Step 2: Convert audio to WAV if necessary
    converted_path_result, error_resp, status_code = _convert_audio(
        cast(str, audio_path_result), temp_session_dir, original_filename, request_id
    )
    if error_resp:
        return error_resp, status_code # type: ignore
    audio_path_for_nemo: str = cast(str, converted_path_result)
    # Step 3: Prepare NeMo inputs (manifest and output directories)
    manifest_filepath, output_diar_dir = _prepare_nemo_inputs(
        audio_path_for_nemo, temp_session_dir, request_id
    )

    try:
        # Step 4: Run NeMo ASR and Diarization pipeline
        trans_info_dict, _ = _run_nemo_pipeline(
            manifest_filepath, output_diar_dir, request_id
        )

        # Step 5: Format the conversation output
        formatted_conversation_lines = _format_conversation_output(
            trans_info_dict, audio_path_for_nemo, output_diar_dir, request_id
        )

        logger.info("[%s] Successfully processed audio. Returning conversation.", request_id)
        return {"conversation": formatted_conversation_lines}, 200

    except Exception as e:
        # Catching broad Exception here is acceptable for top-level API endpoint
        # to ensure all unhandled errors are caught and a 500 response is returned.
        logger.exception(
            "[%s] An unexpected error occurred during NeMo processing for '%s'.",
            request_id, original_filename
        )
        return {"error": f"An unexpected internal server error occurred while "
                          f"processing '{original_filename}'. Details: {str(e)}"}, 500

