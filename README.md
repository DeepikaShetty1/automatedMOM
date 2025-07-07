Speaker Diarization API with NeMo and Flask
This project provides a RESTful API built with Flask that leverages NVIDIA NeMo for advanced speaker diarization and Automatic Speech Recognition (ASR). It allows you to upload audio files and receive a transcript with identified speakers and their respective timestamps.

✨ Features
Speaker Diarization: Identifies and separates different speakers in an audio recording.

Automatic Speech Recognition (ASR): Transcribes speech to text.

Combined Output: Provides a single output with speaker labels, start/end times, and transcribed text for each segment.

RESTful API: Easy integration with other applications via HTTP requests.

Configurable: Uses environment variables for flexible deployment and settings.

Robust Error Handling: Provides informative error messages for common issues.

Logging: Comprehensive logging for monitoring and debugging.

🚀 Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.8+: The programming language for this application.

Git: For cloning the repository.

FFmpeg: An essential tool for audio format conversion.

What it means: Your audio_utils.py script uses FFmpeg behind the scenes to convert various audio types into a standard WAV format that NeMo can process. Without FFmpeg, audio conversion will fail.

Installation:

Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg

macOS (using Homebrew): brew install ffmpeg

Windows: Download from ffmpeg.org and add it to your system's PATH.

⚙️ Setup Instructions
Follow these steps to get your project up and running locally.

1. Clone the Repository
First, clone this repository to your local machine:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME # Navigate into your project directory

What it means: This downloads all the project files to your computer and moves you into the main project folder.

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

python -m venv venv

What it means: This creates an isolated Python environment named venv in your project directory. This prevents conflicts with other Python projects on your system.

3. Activate the Virtual Environment
Activate the virtual environment based on your operating system:

macOS/Linux:

source venv/bin/activate

Windows (Command Prompt):

venv\Scripts\activate.bat

Windows (PowerShell):

venv\Scripts\Activate.ps1

What it means: Activating the virtual environment ensures that any Python packages you install are specific to this project. Your terminal prompt might change to indicate the active environment (e.g., (venv) your_username@your_machine:~/your_project$).

4. Install Python Dependencies
Install all the required Python packages using pip:

pip install -r requirements.txt

What it means: This command reads the requirements.txt file and installs all the necessary libraries like Flask, NeMo, PyTorch, etc., into your active virtual environment.

Note on PyTorch (for GPU support):
If you have a CUDA-enabled GPU and want to leverage it for faster processing with NeMo, you MUST install PyTorch with the correct CUDA version. The requirements.txt file includes comments with examples. For instance, for CUDA 11.8:

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

What it means: This command tells pip to download PyTorch packages specifically compiled for a particular CUDA version, which is necessary for GPU acceleration. If you don't do this, PyTorch will default to CPU, which is slower.

5. Configure Environment Variables (.env file)
Create a file named .env in the root directory of your project (the same directory as app.py). This file will store your environment-specific configurations.

# .env file content
# --- Application Configuration (for app.py) ---
LOG_LEVEL=INFO
FLASK_PORT=5000
FLASK_DEBUG=False # Set to True for development, False for production

# --- NeMo Processor Configuration (for nemo_processor.py) ---
# Directory where NeMo model configuration YAMLs are located
MODEL_CONFIG_DIR=conf
# Logging level for NeMo specific messages (e.g., INFO, DEBUG, WARNING, ERROR)
NEMO_LOG_LEVEL=INFO

# --- Audio Utilities Configuration (for audio_utils.py) ---
# Temporary directory for audio conversions and NeMo outputs
TEMP_AUDIO_DIR=tmp_audio

What it means: This file allows you to customize settings like the server port, logging levels, and temporary directories without modifying the Python code directly. FLASK_DEBUG=False is recommended for production environments.

🚀 Usage
1. Run the Flask Application
Once all dependencies are installed and your .env file is set up, you can start the Flask server:

python app.py

What it means: This command executes app.py, which initializes the Flask web server and loads the NeMo models into memory. You will see log messages indicating the server starting and NeMo components loading.

Initial Loading: The first time you run the application, NeMo models will be downloaded (if not cached) and loaded. This can take a significant amount of time (several minutes) depending on your internet connection and system resources. Subsequent runs will be faster as models will be cached.

2. Check API Status
You can check if the NeMo components are loaded and the API is ready by accessing the /status endpoint:

curl http://127.0.0.1:5000/status

Expected Output (if ready):

{
  "errors": [],
  "nemo_components_loaded": true,
  "status": "ready"
}

What it means: This command sends a request to your API to verify that all the necessary NeMo models have been successfully loaded and the system is operational.

3. Diarize an Audio File
To diarize an audio file, send a POST request to the /diarize endpoint with your audio file.

curl -X POST -F "audio=@/path/to/your/audio.wav" http://127.0.0.1:5000/diarize

Replace /path/to/your/audio.wav with the actual path to your audio file (e.g., audio.mp3, meeting.ogg). The API supports various audio formats thanks to FFmpeg.

Expected Output:

{
  "conversation": [
    {
      "speaker": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 2.5,
      "text": "Hello, how are you doing today?"
    },
    {
      "speaker": "SPEAKER_01",
      "start_time": 3.0,
      "end_time": 5.8,
      "text": "I'm doing great, thanks for asking!"
    }
    // ... more conversation segments
  ]
}

What it means: This command sends your audio file to the API. The API then processes it using NeMo's ASR and diarization models, and returns a structured JSON response containing the transcribed text, speaker labels, and timestamps for each spoken segment.

🧠 Understanding the ML Pipeline
This API implements a Machine Learning (ML) pipeline for speaker diarization and ASR. Here's a simplified breakdown of how an audio file goes from input to a fully diarized transcript:

Audio Input & Pre-processing (app.py, audio_utils.py)

What it means: When you send an audio file to the /diarize endpoint, the app.py receives it. This raw audio can be in various formats (MP3, OGG, etc.).

How it works: The app.py then uses the audio_utils.py module to convert this audio into a standard format (16kHz sample rate, mono channel, 16-bit WAV). This step is crucial because NeMo models expect a very specific audio format.

NeMo Model Initialization (nemo_processor.py, diar_infer_meeting.yaml)

What it means: Before processing any audio, the application needs to load the "brains" of the operation – the NeMo models. This happens only once when the Flask application starts up.

How it works: The nemo_processor.py module reads the diar_infer_meeting.yaml file to understand which NeMo models to use (e.g., specific ASR models like stt_en_citrinet and speaker embedding models like titanet_large). It then loads these large models into your computer's memory, making them ready for fast processing.

Automatic Speech Recognition (ASR) (nemo_processor.py)

What it means: Once the audio is pre-processed, the first ML step is to convert the speech into text.

How it works: The nemo_processor.py feeds the standardized audio to the loaded ASR model. This model generates a transcript of what was said, along with precise timestamps for each word.

Speaker Diarization (nemo_processor.py)

What it means: At the same time, or shortly after ASR, the pipeline identifies who spoke when.

How it works: The nemo_processor.py uses the speaker diarization model. This model analyzes the audio and determines distinct speakers (e.g., "SPEAKER_00", "SPEAKER_01"). It outputs information about when each speaker started and stopped talking.

Result Combination & Formatting (nemo_processor.py)

What it means: The final step is to bring the text (from ASR) and the speaker information (from diarization) together into one easy-to-read output.

How it works: The nemo_processor.py takes the word timestamps from ASR and the speaker segments from diarization. It then intelligently combines them, assigning each piece of transcribed text to the correct speaker and providing the exact start and end times for their speech turn. This combined information is then formatted into the JSON response you receive.

📁 Project Structure
app.py: The main Flask application file, defining API routes and orchestrating calls to other modules.

audio_utils.py: Contains utility functions for audio file conversion (e.g., to WAV format using FFmpeg).

nemo_processor.py: Encapsulates the core NeMo logic for loading models, running ASR, performing diarization, and combining results.

diar_infer_meeting.yaml: The configuration file for the NeMo diarization pipeline, specifying model paths and parameters.

requirements.txt: Lists all Python dependencies required for the project.

.env: (Not committed to Git) Stores environment variables for configuration.

.gitignore: Specifies files and directories that Git should ignore (e.g., virtual environments, logs, temporary files, .env).

🪵 Logging
The application is configured to log messages to both the console and a file (./logs/diarization_api.log). You can control the logging level (e.g., INFO, DEBUG, WARNING, ERROR) via the LOG_LEVEL environment variable in your .env file. NeMo's internal logging level can also be adjusted via NEMO_LOG_LEVEL.

What it means: Logging helps you monitor the application's activity, track the progress of audio processing, and diagnose any issues that might arise.

🐛 Troubleshooting
FFmpeg not found error: Ensure FFmpeg is correctly installed on your system and its executable directory is added to your system's PATH environment variable.

NeMo model download issues: Check your internet connection. Large NeMo models are downloaded on first use. If downloads are interrupted, you might need to clear the NeMo cache (typically in ~/.cache/torch/NeMo or ~/.cache/huggingface/hub) and retry.

ModuleNotFoundError: Make sure you have activated your virtual environment and installed all dependencies from requirements.txt.

Argument of type "list[Unknown]" cannot be assigned... (Pylance error): This is a type-checking hint. Ensure your nemo_processor.py has the cast operation for rttm_to_labels as discussed in previous conversations to help the type checker. The code should still run correctly even if Pylance shows this warning without the cast.

