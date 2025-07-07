import os
import uuid
import shutil
import logging
from flask import Flask, request, jsonify
from flasgger import Swagger
from typing import Optional # Import Optional for type hinting
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()

# --- Logging Configuration ---
# Use an environment variable for the log directory, with a fallback default
LOG_DIR = os.getenv('LOG_DIR', './logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, 'diarization_api.log')

# Configure logging to write to a file and stream to console
logging.basicConfig(
    level=logging.INFO, # Default level, can be overridden by env var
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set logging level from environment variable if provided, default to INFO
LOG_LEVEL = os.getenv('APP_LOG_LEVEL', 'INFO').upper()
logger.setLevel(LOG_LEVEL)
logging.getLogger('nemo_processor').setLevel(LOG_LEVEL) # Also set level for nemo_processor
logging.getLogger('audio_utils').setLevel(LOG_LEVEL) # Also set level for audio_utils
logging.getLogger('werkzeug').setLevel(LOG_LEVEL) # Flask's internal logger

logger.info(f"Application logging configured. Log level set to: {LOG_LEVEL}")
logger.info(f"Log file path: {LOG_FILE_PATH}")
# --- End Logging Configuration ---


# Flask app setup
app = Flask(__name__)

# Initialize Flasgger for Swagger UI
swagger = Swagger(app)

# Base directory for storing temporary audio files and diarization outputs
# Read from environment variable, with a sensible default
OUTPUT_BASE_DIR = os.getenv('OUTPUT_BASE_DIR', '/tmp/diarization_api_results')
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True) # Ensure base dir exists at startup
logger.info(f"Temporary output base directory set to: {OUTPUT_BASE_DIR}")


# Load NeMo components when the Flask app starts.
# This ensures models are loaded only once.
with app.app_context():
    logger.info("Starting NeMo AI model loading process... This may take a few minutes.")

    # Import the core processing logic and utility functions from our new modules
    import nemo_processor # Import the entire module

    nemo_processor.load_nemo_components()
    if nemo_processor.nemo_cfg and nemo_processor.asr_model_instance:
        logger.info("NeMo AI models loaded successfully! The API is ready.")
        print("NeMo AI models loaded successfully! The API is ready.") # Keep print for immediate console feedback
    else:
        logger.critical("ERROR: Failed to load NeMo AI models at startup. The API will not function.")
        print("ERROR: Failed to load NeMo AI models. The API will not function.") # Keep print for immediate console feedback


@app.route('/diarize', methods=['POST'])
def diarize_audio():
    """
    Submit Audio for Speaker Diarization and Get Results.
    This endpoint processes an uploaded audio file to perform speaker diarization and ASR.
    It returns the formatted conversation directly upon completion.
    ---
    parameters:
      - name: audio_file
        in: formData
        type: file
        required: true
        description: The audio file to be processed (WAV, FLAC, or MP3).
    responses:
      200:
        description: Diarization successful, returns the formatted conversation.
        schema:
          type: object
          properties:
            conversation:
              type: array
              items:
                type: string
              example:
                - "speaker_0: (0.48-41.20s) good morning everyone today I would like to start their conversation..."
                - "speaker_1: (41.22-77.30s) said tangeuver giving this opportunity iwould like do said eu best point..."
      400:
        description: Bad Request - Missing file, empty file, or unsupported format.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "No audio file selected. Please select a file to upload."
      500:
        description: Internal Server Error - Issues with FFmpeg, NeMo processing, or other server errors.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Audio conversion failed for 'your_audio.mp3'. Details: ffmpeg output. Please check if the audio file is valid."
      503:
        description: Service Unavailable - NeMo models failed to load at server startup.
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Diarization service not ready. Models failed to load at startup. Please try again later or contact support."
    tags:
      - Audio Diarization
    """
    request_id = str(uuid.uuid4()) # Generate a unique ID for each request for better traceability
    logger.info(f"[{request_id}] Received new diarization request.")

    # Validate API service readiness by checking if NeMo core components are loaded.
    # asr_decoder_ts_instance and asr_diar_offline_instance are no longer global attributes.
    if nemo_processor.nemo_cfg is None or nemo_processor.asr_model_instance is None:
        logger.error(f"[{request_id}] Request received but NeMo core components are not loaded. Returning 503.")
        print("API is not ready. NeMo models failed to load at startup. Please check server logs.")
        return jsonify({"error": "Diarization service not ready. NeMo core components failed to load at startup. Please try again later or contact support."}), 503

    if 'audio_file' not in request.files:
        logger.warning(f"[{request_id}] Received request with no 'audio_file' part. Returning 400.")
        print("Error: No audio file provided in the request.")
        return jsonify({"error": "No 'audio_file' part in the request. Please ensure you are uploading a file with the field name 'audio_file'."}), 400

    audio_file = request.files['audio_file']
    raw_filename: Optional[str] = audio_file.filename

    if raw_filename is None or raw_filename == '':
        logger.warning(f"[{request_id}] Received request with empty or None filename for 'audio_file'. Returning 400.")
        print("Error: No audio file selected or file has no name. Please select a valid file to upload.")
        return jsonify({"error": "No audio file selected or file has no name. Please select a valid file to upload."}), 400

    filename: str = raw_filename
    logger.info(f"[{request_id}] Processing audio file: {filename}")
    print(f"Processing audio: {filename}...")

    temp_session_dir = os.path.join(OUTPUT_BASE_DIR, request_id) # Use request_id for the session dir
    os.makedirs(temp_session_dir, exist_ok=True)
    logger.info(f"[{request_id}] Created temporary session directory: {temp_session_dir}")

    try:
        # Call process_audio from the nemo_processor module, now correctly passing request_id
        result, status_code = nemo_processor.process_audio(audio_file, filename, temp_session_dir, request_id)
        if status_code != 200:
            logger.error(f"[{request_id}] Processing failed for '{filename}': {result.get('error', 'Unknown error')}")
            print(f"Processing failed for {filename}. Error: {result.get('error', 'Please check server logs.')}")
        else:
            logger.info(f"[{request_id}] Successfully processed '{filename}'.")
            print(f"Successfully processed {filename}. Results returned.")
        return jsonify(result), status_code

    except Exception as e:
        logger.exception(f"[{request_id}] An unhandled exception occurred during processing of '{filename}'.")
        print(f"An unexpected error occurred while processing {filename}. Please check server logs.")
        return jsonify({"error": f"An unexpected internal server error occurred while processing '{filename}'. Please try again or contact support if the issue persists. Details: {str(e)}"}), 500

    finally:
        if os.path.exists(temp_session_dir):
            shutil.rmtree(temp_session_dir)
            logger.info(f"[{request_id}] Cleaned up temporary directory: {temp_session_dir}")

if __name__ == '__main__':
    # Use environment variable for Flask debug mode, default to False for production readiness
    DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))

    logger.info(f"Starting Flask application server on {FLASK_HOST}:{FLASK_PORT} (Debug: {DEBUG_MODE})...")
    app.run(debug=DEBUG_MODE, host=FLASK_HOST, port=FLASK_PORT)

