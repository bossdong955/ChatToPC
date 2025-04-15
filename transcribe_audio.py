import os
import traceback
import sys
from flask import Flask, request, jsonify # Flask core components
import torch # Check CUDA availability

# --- FunASR Imports ---
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# --- Configuration ---
MODEL_IDENTIFIER = "models/SenseVoiceSmall" # Or your model path/name
VAD_MODEL = "fsmn-vad"
VAD_KWARGS = {"max_single_segment_time": 30000}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- FunASR Model Loading Function ---
def load_funasr_sensevoice_model():
    """Loads the FunASR SenseVoiceSmall model. Called once on app startup."""
    print(f"INFO: Loading FunASR AutoModel: {MODEL_IDENTIFIER}")
    print(f"INFO: Using device: {DEVICE}")
    try:
        model = AutoModel(
            model=MODEL_IDENTIFIER,
            vad_model=VAD_MODEL,
            vad_kwargs=VAD_KWARGS,
            device=DEVICE,
        )
        print(f"INFO: FunASR model '{MODEL_IDENTIFIER}' loaded successfully.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load FunASR model '{MODEL_IDENTIFIER}': {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- FunASR Transcription Function ---
def transcribe_with_funasr(model, audio_path):
    """
    Performs speech recognition using the loaded FunASR model.

    Args:
        model: The loaded FunASR AutoModel object.
        audio_path (str): The absolute path to the audio file on the server
                          (constructed relative to CWD in this version).

    Returns:
        str: The recognized text (potentially post-processed).
        None: If transcription fails.
    """
    print(f"INFO: Transcribing audio file with FunASR: {audio_path} ...")
    try:
        # Check existence before attempting transcription
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file does not exist at path passed to transcription: {audio_path}", file=sys.stderr)
            return None
        if not os.path.isfile(audio_path):
             print(f"ERROR: Path passed to transcription is not a file: {audio_path}", file=sys.stderr)
             return None

        res = model.generate(
            input=audio_path, # Use the server-side audio file path
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        if not res or not isinstance(res, list) or len(res) == 0 or "text" not in res[0]:
             print(f"ERROR: Unexpected or empty result format from model.generate: {res}", file=sys.stderr)
             return None

        raw_text = res[0].get("text", "")
        if not raw_text:
             print(f"WARNING: FunASR returned result but 'text' field is empty.", file=sys.stderr)
             return None # Treat empty text as potential issue in this context

        print(f"INFO: Raw transcription: '{raw_text}'")
        processed_text = rich_transcription_postprocess(raw_text)
        print(f"INFO: Post-processed transcription: '{processed_text}'")
        return processed_text

    except Exception as e:
        print(f"ERROR: Exception during FunASR transcription ({os.path.basename(audio_path)}): {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Flask Application Initialization ---
app = Flask(__name__)

# --- Load FunASR Model (once on startup) ---
print("INFO: Flask application starting, loading FunASR model...")
FUNASR_MODEL = load_funasr_sensevoice_model()

if FUNASR_MODEL is None:
    print("CRITICAL WARNING: Model loading failed, API will not be able to process requests.", file=sys.stderr)


# --- API Endpoint Definition ---
# WARNING: This endpoint relies on paths relative to the server's CWD, which is insecure.
@app.route('/transcribe', methods=['POST'])
def handle_transcription_by_relative_path_request():
    """
    Handles POST requests containing a JSON payload with an 'audio_path' key.
    The 'audio_path' value should be a relative path. The API will attempt
    to resolve this path relative to its Current Working Directory (CWD).

    *** WARNING: THIS IS AN INSECURE AND UNRELIABLE APPROACH. ***
    Use the version with ALLOWED_AUDIO_BASE_DIR for production.
    """
    # 1. Check if model is loaded
    if FUNASR_MODEL is None:
         print("ERROR: Transcription request received, but model is not loaded.", file=sys.stderr)
         return jsonify({"error": "Server model error, transcription service unavailable"}), 200

    # 2. Check if request body is JSON
    if not request.is_json:
        print("ERROR: Request content type is not application/json.", file=sys.stderr)
        return jsonify({"error": "Request body must be JSON"}), 200

    # 3. Get JSON data
    data = request.get_json()
    if not data:
        print("ERROR: Received empty JSON payload.", file=sys.stderr)
        return jsonify({"error": "Empty JSON payload received"}), 200

    # 4. Check for 'audio_path' key in JSON
    relative_path_from_client = data.get('audio_path')
    if not relative_path_from_client:
        print("ERROR: 'audio_path' key missing or empty in JSON payload.", file=sys.stderr)
        return jsonify({"error": "Missing 'audio_path' in JSON request body"}), 200

    if not isinstance(relative_path_from_client, str):
         print(f"ERROR: 'audio_path' value is not a string: {type(relative_path_from_client)}", file=sys.stderr)
         return jsonify({"error": "'audio_path' value must be a string"}), 200

    # --- 5. Path Construction and Basic Validation (INSECURE METHOD) ---
    server_audio_path = None
    try:
        # Check 1: Disallow absolute paths provided by client
        if os.path.isabs(relative_path_from_client):
            print(f"SECURITY REJECT: Absolute path provided by client: {relative_path_from_client}", file=sys.stderr)
            return jsonify({"error": "Absolute paths are not allowed"}), 200

        # Check 2: Disallow paths containing '..' to prevent basic traversal
        # WARNING: This might not catch all traversal tricks depending on OS/environment.
        if ".." in relative_path_from_client.split(os.path.sep):
            print(f"SECURITY REJECT: Path contains '..': {relative_path_from_client}", file=sys.stderr)
            return jsonify({"error": "Directory traversal ('..') is not allowed"}), 200

        # Construct path relative to the Current Working Directory
        # *** WARNING: CWD can be unpredictable and depends on how the server is run! ***
        current_cwd = os.getcwd()
        # Use os.path.abspath to resolve the path relative to CWD
        server_audio_path = os.path.abspath(os.path.normpath(os.path.join(current_cwd+"/audio/", relative_path_from_client)))

        # Log CWD and the resulting path for debugging (essential with this method)
        print(f"INFO: Current Working Directory: {current_cwd}")
        print(f"INFO: Attempting to access server-side path: {server_audio_path} (based on client input: '{relative_path_from_client}')")

    except Exception as e:
        print(f"ERROR: Error during path processing for '{relative_path_from_client}': {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": "Internal error during path processing"}), 200


    # 6. Check if the constructed file path exists and is a file
    if not server_audio_path:
         # Should not happen if code above runs correctly, but as a safeguard
         print(f"ERROR: Server audio path was not constructed.", file=sys.stderr)
         return jsonify({"error": "Internal error processing path"}), 200

    if not os.path.exists(server_audio_path):
        print(f"ERROR: Resolved audio file does not exist on server: {server_audio_path}", file=sys.stderr)
        return jsonify({"error": f"Audio file not found on server at resolved path based on: {relative_path_from_client}"}), 200 # Not Found

    if not os.path.isfile(server_audio_path):
        print(f"ERROR: Resolved path is not a file: {server_audio_path}", file=sys.stderr)
        return jsonify({"error": f"Resolved path is not a file based on: {relative_path_from_client}"}), 200 # Bad Request


    # --- 7. Perform Transcription ---
    try:
        transcription_result = transcribe_with_funasr(FUNASR_MODEL, server_audio_path)

        if transcription_result is not None:
            print("INFO: Transcription successful.")
            return jsonify({"transcription": transcription_result}), 200 # OK
        else:
            print("ERROR: Transcription failed (FunASR function returned None).", file=sys.stderr)
            return jsonify({"error": "Speech transcription processing failed on server"}), 200 # Internal Server Error

    except Exception as e:
        # Catch unexpected errors during transcription call
        print(f"ERROR: Unexpected exception calling transcription function: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": "Internal server error during transcription"}), 200


# --- Main Entry Point ---
if __name__ == '__main__':
    print("Starting FunASR Speech Recognition API (Relative Path Mode)...")
    print("\n *** WARNING: Running in insecure mode. Paths are resolved relative to CWD. ***")
    print(" *** This is NOT recommended for production environments. ***\n")
    if FUNASR_MODEL is None:
         print("\n *** WARNING: Model loading failed. Requests will fail. ***\n", file=sys.stderr)
    else:
        print(f"Model: {MODEL_IDENTIFIER}, Device: {DEVICE}")
        print(f"Current Working Directory at startup: {os.getcwd()}") # Log CWD at startup

    # Run Flask server
    # Use debug=False for production/security
    app.run(host='0.0.0.0', port=8001, debug=False)