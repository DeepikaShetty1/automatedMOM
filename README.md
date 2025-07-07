# Speaker Diarization API with NeMo and Flask

This project provides a powerful RESTful API built with Flask, leveraging NVIDIA NeMo for advanced speaker diarization and Automatic Speech Recognition (ASR). Upload an audio file and receive a detailed transcript, identifying who spoke when.

---

## ✨ Key Features

* **Speaker Diarization:** Automatically identifies and separates distinct speakers in an audio recording.
* **Automatic Speech Recognition (ASR):** Converts spoken words into text.
* **Unified Output:** Provides a single, easy-to-read transcript with speaker labels, precise start/end times, and the corresponding text for each segment.
* **RESTful Interface:** Designed for easy integration with other applications or services via HTTP requests.
* **Flexible Configuration:** Customize settings using environment variables for different deployment needs.
* **Robust Error Handling:** Clear and informative error messages to aid in troubleshooting.
* **Comprehensive Logging:** Detailed logs for monitoring application activity and debugging.

---

## 🚀 Getting Started

### Prerequisites

Before you set up the project, ensure you have the following installed:

* **Python 3.8+**: The programming language for this application.
* **Git**: For cloning the repository.
* **FFmpeg**: An essential tool for audio format conversion. This application uses FFmpeg to convert various audio types into the standard WAV format required by NeMo.

    * **Installation Commands:**
        * **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
        * **macOS (Homebrew):** `brew install ffmpeg`
        * **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system's PATH.

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create & Activate Virtual Environment:**
    It's best practice to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows (Command Prompt):
    venv\Scripts\activate.bat
    # On Windows (PowerShell):
    venv\Scripts\Activate.ps1
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note on GPU Support (PyTorch):** If you have a CUDA-enabled GPU, install PyTorch with the correct CUDA version for faster processing. Refer to the comments in `requirements.txt` for specific commands (e.g., `pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118`).

4.  **Configure Environment Variables (`.env` file):**
    Create a file named `.env` in the root of your project (next to `app.py`) and add your configurations:
    ```ini
    # .env file content
    LOG_LEVEL=INFO
    FLASK_PORT=5000
    FLASK_DEBUG=False # Set to True for development, False for production

    MODEL_CONFIG_DIR=conf
    NEMO_LOG_LEVEL=INFO

    TEMP_AUDIO_DIR=tmp_audio
    ```

---

## 🚀 Usage

1.  **Run the Flask Application:**
    ```bash
    python app.py
    ```
    The server will start, and NeMo models will begin loading. This initial loading can take several minutes as large models are downloaded and cached. Subsequent runs will be faster.

2.  **Check API Status:**
    Verify the API is ready by checking the `/status` endpoint:
    ```bash
    curl http://127.0.0.1:5000/status
    ```
    Expected output (if ready):
    ```json
    {"errors": [], "nemo_components_loaded": true, "status": "ready"}
    ```

3.  **Diarize an Audio File:**
    Send a POST request with your audio file to the `/diarize` endpoint:
    ```bash
    curl -X POST -F "audio=@/path/to/your/audio.wav" http://127.0.0.1:5000/diarize
    ```
    Replace `/path/to/your/audio.wav` with the actual path to your audio file. The API supports various formats (MP3, OGG, etc.) thanks to FFmpeg.

    Expected output (JSON):
    ```json
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
    ```

---

## 🧠 Understanding the Machine Learning Pipeline

This API processes audio through a sophisticated Machine Learning pipeline to deliver accurate speaker diarization and transcription. Here's a step-by-step breakdown:

### 1. Audio Ingestion & Pre-processing

* **What happens:** Your raw audio file (e.g., MP3, OGG) is received by the Flask API.
* **ML relevance:** Before any ML models can work, the audio needs to be in a very specific, clean format. The `audio_utils.py` module uses FFmpeg to convert the input audio to a standardized 16kHz sample rate, mono channel, 16-bit WAV format. This ensures compatibility and optimal performance for the downstream NeMo models.

### 2. NeMo Model Initialization

* **What happens:** When the Flask application first starts, the necessary NeMo ML models are loaded into memory.
* **ML relevance:** This crucial step involves reading the configuration (`diar_infer_meeting.yaml`) to identify which ASR and speaker diarization models to use (e.g., a Conformer-based ASR model and a TitaNet speaker embedding model). Loading these large, pre-trained neural networks once at startup prevents delays for every new request, making the API responsive.

### 3. Automatic Speech Recognition (ASR)

* **What happens:** The pre-processed audio is fed into the ASR model.
* **ML relevance:** The ASR model (a deep learning model trained on vast amounts of speech data) analyzes the audio's acoustic patterns and converts them into text. It not only provides the words but also precise timestamps for each word, which is vital for later combining with speaker information.

### 4. Speaker Diarization

* **What happens:** Simultaneously or sequentially, the speaker diarization component processes the audio.
* **ML relevance:** This part of the pipeline uses various ML techniques (like voice activity detection, speaker embedding extraction, and clustering algorithms) to identify segments of speech and group them by speaker. It determines "who spoke when," assigning unique identifiers (e.g., SPEAKER_00, SPEAKER_01) to each detected voice.

### 5. Result Integration & Formatting

* **What happens:** The separate outputs from ASR (text and word timestamps) and Speaker Diarization (speaker turns) are merged.
* **ML relevance:** This final step intelligently combines the information. The system aligns the transcribed words with the speaker segments, ensuring that each piece of text is correctly attributed to the speaker who uttered it, along with their exact start and end times. The result is then formatted into a clear JSON structure for the API response.

---

## 📁 Project Structure

* `app.py`: The main Flask application, defining API routes and orchestrating the ML pipeline.
* `audio_utils.py`: Handles audio conversion using FFmpeg.
* `nemo_processor.py`: Contains the core logic for loading NeMo models, executing the ASR and Diarization steps, and combining their outputs.
* `diar_infer_meeting.yaml`: Configuration file for the NeMo ML models and pipeline parameters.
* `requirements.txt`: Lists all Python package dependencies.
* `.env`: (Not committed to Git) Stores environment-specific configurations.
* `.gitignore`: Specifies files and directories to be ignored by Git.

---

## 🪵 Logging

The application logs messages to both the console and a file (`./logs/diarization_api.log`). Adjust the `LOG_LEVEL` and `NEMO_LOG_LEVEL` in your `.env` file to control the verbosity (e.g., `INFO`, `DEBUG`, `WARNING`, `ERROR`). Logging helps monitor the application's activity and diagnose issues.

---


