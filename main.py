# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import keyboard
import soundfile as sf
import threading
import time
import sys
import traceback
import requests
import os
from dotenv import load_dotenv
import json
import pyttsx3
import tempfile
import tkinter as tk
from tkinter import scrolledtext, Label
import queue

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from openai import AuthenticationError, APIError

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file into environment

# --- Configuration Parameters (Loaded from .env) ---

# Audio Settings
DEFAULT_SAMPLERATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_DEVICE = None
DEFAULT_AUDIO_SAVE_DIR = "./audio/"
DEFAULT_FILENAME_BASE = "recorded_audio"
DEFAULT_RECORD_START_DELAY = 0.3
DEFAULT_SHOW_LLM_RESPONSE_POPUP = "True"
DEFAULT_POPUP_AUTO_CLOSE = "True"
DEFAULT_ENABLE_TTS = "True"
DEFAULT_SENSEVOICE_API_URL = "http://localhost:8001/transcribe"
DEFAULT_OPENAI_MODEL_NAME = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and friendly conversational assistant. Respond concisely and naturally to the user's transcribed speech."


try:
    SAMPLERATE = int(os.getenv("SAMPLERATE", DEFAULT_SAMPLERATE))
except (ValueError, TypeError):
    print(f"警告: .env 中的 SAMPLERATE 无效，使用默认值 {DEFAULT_SAMPLERATE}", file=sys.stderr)
    SAMPLERATE = DEFAULT_SAMPLERATE
try:
    CHANNELS = int(os.getenv("CHANNELS", DEFAULT_CHANNELS))
except (ValueError, TypeError):
    print(f"警告: .env 中的 CHANNELS 无效，使用默认值 {DEFAULT_CHANNELS}", file=sys.stderr)
    CHANNELS = DEFAULT_CHANNELS

# Handle Audio Device setting (None, int index, or string name)
device_setting = os.getenv("AUDIO_INPUT_DEVICE") # Returns None if not set
if not device_setting: # If empty string or None
    DEVICE = DEFAULT_DEVICE
    print(f"信息: 未在 .env 中指定 AUDIO_INPUT_DEVICE，使用系统默认设备 ({DEVICE})")
else:
    try:
        DEVICE = int(device_setting) # Try integer index first
        print(f"信息: 使用 .env 中指定的音频设备索引: {DEVICE}")
    except ValueError:
        DEVICE = device_setting # Use as string name otherwise
        print(f"信息: 使用 .env 中指定的音频设备名称: '{DEVICE}'")

# Recording Settings

AUDIO_SAVE_DIR = os.getenv("AUDIO_SAVE_DIR", DEFAULT_AUDIO_SAVE_DIR)
FILENAME_BASE = os.getenv("FILENAME_BASE", DEFAULT_FILENAME_BASE)
try:
    RECORD_START_DELAY = float(os.getenv("RECORD_START_DELAY", DEFAULT_RECORD_START_DELAY))
    if RECORD_START_DELAY < 0:
        print(f"警告: RECORD_START_DELAY 不能为负数，使用默认值 {DEFAULT_RECORD_START_DELAY}", file=sys.stderr)
        RECORD_START_DELAY = DEFAULT_RECORD_START_DELAY
except (ValueError, TypeError):
    print(f"警告: .env 中的 RECORD_START_DELAY 无效，使用默认值 {DEFAULT_RECORD_START_DELAY}", file=sys.stderr)
    RECORD_START_DELAY = DEFAULT_RECORD_START_DELAY

# Feature Flags (Boolean)

SHOW_LLM_RESPONSE_POPUP = os.getenv("SHOW_LLM_RESPONSE_POPUP", DEFAULT_SHOW_LLM_RESPONSE_POPUP).lower() == "true"
POPUP_AUTO_CLOSE = os.getenv("POPUP_AUTO_CLOSE", DEFAULT_POPUP_AUTO_CLOSE).lower() == "true"
ENABLE_TTS = os.getenv("ENABLE_TTS", DEFAULT_ENABLE_TTS).lower() == "true"

# API Configuration (Strings)

SENSEVOICE_API_URL = os.getenv("SENSEVOICE_API_URL", DEFAULT_SENSEVOICE_API_URL)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # No default, should be explicitly set
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") # Optional, None if not set
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", DEFAULT_OPENAI_MODEL_NAME)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT_CHAT", DEFAULT_SYSTEM_PROMPT)


# --- Global Variables and State ---
is_recording = False
audio_data = None
stream = None
recording_lock = threading.Lock()
recording_start_timer = None

# --- TTS State Management ---
tts_engine = None
tts_finished_event = threading.Event()
tts_finished_event.set()

# --- Status Pop-up State ---
status_popup_ref = {'window': None, 'root': None}
status_popup_lock = threading.Lock()

# --- TTS Engine Initialization ---
if ENABLE_TTS:
    try:
        tts_engine = pyttsx3.init()
        print("信息: TTS 引擎已初始化。")
    except Exception as e:
        print(f"错误: 初始化 TTS 引擎失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        tts_engine = None
        ENABLE_TTS = False # Disable TTS if init fails
        print("警告: TTS 功能因初始化失败已被禁用。")
else:
    print("信息: TTS 功能已通过配置禁用，跳过引擎初始化。")
    tts_engine = None

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(f"Audio Stream Status Error: {status}", file=sys.stderr)
    try:
        with recording_lock:
            if is_recording and isinstance(audio_data, list):
                 audio_data.append(indata.copy())
    except Exception as e:
        print(f"Error in audio_callback: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

# --- Start Recording Functions ---
def start_recording():
    global stream, is_recording, audio_data
    try:
        if stream is not None and not stream.closed:
            print("DEBUG: Closing leftover audio stream before starting new one.")
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except Exception as e_close:
                print(f"Error closing leftover stream: {e_close}", file=sys.stderr)
            finally:
                stream = None

        stream = sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            callback=audio_callback,
            device=DEVICE,
            dtype='float32'
        )
        stream.start()
        print("音频流已启动。")

    except sd.PortAudioError as pae:
        print(f"PortAudio 错误: 无法启动音频流。", file=sys.stderr)
        print(f"错误详情: {pae}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        with recording_lock:
             is_recording, audio_data, stream = False, None, None
        print("错误: 录音启动失败。")
        close_status_popup() # Close "Listening" popup if start fails
    except Exception as e:
        print(f"启动音频流时发生未知错误: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        with recording_lock:
             is_recording, audio_data, stream = False, None, None
        print("错误: 录音启动失败。")
        close_status_popup() # Close "Listening" popup if start fails

def _initiate_recording_after_delay():
    global recording_start_timer, is_recording, audio_data
    should_start = False
    with recording_lock:
        if recording_start_timer is None:
             print("DEBUG: _initiate_recording_after_delay called, but timer was cancelled.")
             return

        print(f"DEBUG: {RECORD_START_DELAY}秒计时器触发，准备启动录音...")
        recording_start_timer = None # Timer has fired

        if not is_recording:
             is_recording = True
             audio_data = [] # Reset audio data list
             should_start = True
        else:
             print("DEBUG: Timer fired, but recording flag was already true.", file=sys.stderr)

    if should_start:
        print(f"开始录音 (已等待 {RECORD_START_DELAY} 秒)...")
        display_status_popup("正在聆听中...")
        start_recording() # Now actually start the audio stream


# --- Status Pop-up Functions ---
def _run_status_popup_thread(text, q):
    """Target function for the status popup thread."""
    popup = None
    root = None
    try:
        root = tk.Tk()
        root.withdraw() # Hide the root window

        popup = tk.Toplevel(root)
        popup.title("状态")
        popup.geometry("200x80")
        popup.resizable(False, False)
        popup.attributes('-topmost', True)

        # Attempt to center the popup window
        root.update_idletasks() # Ensure screen size is available
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (200 // 2)
        y = (screen_height // 2) - (80 // 2)
        popup.geometry(f'+{x}+{y}')

        # Display the status message in a Label
        label = Label(popup, text=text, padx=20, pady=20)
        label.pack(expand=True)

        # Put references into the queue for the calling thread
        q.put({'window': popup, 'root': root})

        # Disable the close button's default action
        popup.protocol("WM_DELETE_WINDOW", lambda: None) # Ignore close button click

        # Start the event loop for this popup - runs until root is destroyed
        root.mainloop()

    except Exception as e:
        print(f"Error in status popup thread: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        q.put(None) # Signal error
    # <<<--- REMOVED finally block that caused the error ---<<<

    # Optional: Print message *after* mainloop finishes, indicating thread exit
    print("DEBUG: Status popup thread mainloop finished.")


def display_status_popup(text):
    """Creates and displays a short-lived status popup."""
    global status_popup_ref, status_popup_lock
    print(f"DEBUG: Displaying status popup: '{text}'")

    # Close any existing status popup first
    close_status_popup()

    q = queue.Queue()
    thread = threading.Thread(target=_run_status_popup_thread, args=(text, q), daemon=True)
    thread.start()

    try:
        # Wait for the thread to create the window and return references
        ref = q.get(timeout=3.0)
        if ref and ref.get('window') and ref.get('root'):
            # Store references safely
            with status_popup_lock:
                status_popup_ref = ref
            print(f"DEBUG: Status popup '{text}' displayed.")
        else:
            print("Error: Status popup thread failed to return valid window/root.", file=sys.stderr)
    except queue.Empty:
        print("Error: Timed out waiting for status popup thread.", file=sys.stderr)
    except Exception as e:
        print(f"Error displaying status popup: {e}", file=sys.stderr)

def close_status_popup():
    """Safely closes the current status popup using root.after."""
    global status_popup_ref, status_popup_lock
    print("DEBUG: Attempting to close status popup...")

    # Safely get the current references and clear the global ones
    with status_popup_lock:
        ref_to_close = status_popup_ref.copy()
        status_popup_ref = {'window': None, 'root': None} # Clear immediately

    root_to_close = ref_to_close.get('root')
    if root_to_close:
        try:
            # Schedule the destroy call on the Tkinter thread of the popup's root
            if root_to_close.winfo_exists(): # Check existence first
                 root_to_close.after(0, root_to_close.destroy)
                 print("DEBUG: Scheduled status popup closure.")
            else:
                 print("DEBUG: Status popup root already destroyed.")
        except tk.TclError as e:
            print(f"DEBUG: TclError closing status popup (likely race): {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error scheduling status popup closure: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    else:
        print("DEBUG: No active status popup root to close.")

# --- LLM Response Pop-up Functions ---
def _create_and_run_tk_window(text_to_display, q, title="LLM Response"):
    """Creates and manages the LLM Response tkinter Toplevel window."""
    popup_window = None
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        popup_window = tk.Toplevel(root)
        popup_window.title(title)
        popup_window.geometry("500x300")
        popup_window.attributes('-topmost', True)

        st = scrolledtext.ScrolledText(popup_window, wrap=tk.WORD, height=15, width=60)
        st.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        st.insert(tk.END, text_to_display)
        st.config(state=tk.DISABLED)

        q.put(popup_window)
        print("DEBUG: LLM Response Tkinter window object placed in queue.")

        def on_close():
            print("DEBUG: LLM Response Tkinter window closed by user (X button).")
            try:
                if root and root.winfo_exists():
                    root.destroy() # Destroy root to exit mainloop
            except tk.TclError:
                pass # Ignore if already destroyed

        popup_window.protocol("WM_DELETE_WINDOW", on_close)

        print("DEBUG: Starting LLM Response Tkinter mainloop in background thread.")
        root.mainloop() # Blocks until root is destroyed
        print("DEBUG: LLM Response Tkinter mainloop finished.")

    except Exception as e:
        print(f"Error in LLM Response Tkinter thread: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        q.put(None) # Signal error

def show_response_popup_tk(text_to_display, title="LLM Response"):
    """Starts the tkinter pop-up for LLM response in a thread."""
    print("DEBUG: Requesting LLM Response Tkinter pop-up window creation...")
    q = queue.Queue()
    thread = threading.Thread(target=_create_and_run_tk_window, args=(text_to_display, q, title), daemon=True)
    thread.start()
    try:
        popup_window = q.get(timeout=5.0)
        if popup_window:
            print("DEBUG: Received LLM Response Tkinter window object.")
            return popup_window
        else:
            print("Error: LLM Response Tkinter thread signaled error.", file=sys.stderr)
            return None
    except queue.Empty:
        print("Error: Timed out waiting for LLM Response popup.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error retrieving LLM Response popup: {e}", file=sys.stderr)
        return None

# --- Stop Recording, Save, Transcribe, Query LLM, Show/Speak Response ---
def stop_recording_and_save():
    """Stops recording, saves audio, transcribes, gets LLM response, shows/speaks it."""
    global is_recording, audio_data, stream, SHOW_LLM_RESPONSE_POPUP, POPUP_AUTO_CLOSE, ENABLE_TTS
    local_stream = None
    local_audio_data = None
    should_process = False
    llm_popup_window = None # For the final LLM response popup

    with recording_lock:
        if is_recording:
            print("DEBUG: Stopping recording process...")
            is_recording, should_process = False, True
            local_stream, local_audio_data = stream, audio_data
            stream, audio_data = None, None
        else:
            print("DEBUG: stop_recording_and_save called, but not currently recording.")
            return

    if not should_process:
        return

    close_status_popup() # Close "Listening" popup
    print("停止录音...")

    if local_stream:
        try:
            if local_stream.active:
                local_stream.stop()
            if not local_stream.closed:
                local_stream.close()
            print("DEBUG: Audio input stream stopped and closed.")
        except Exception as e:
            print(f"停止/关闭音频流时出错: {e}", file=sys.stderr)

    if not local_audio_data or len(local_audio_data) == 0:
        print("没有录制到有效音频数据。")
        error_message = "I didn't capture any audio."
        if ENABLE_TTS:
            speak_text(error_message)
        else:
            print(f"DEBUG: TTS Disabled. Info: {error_message}")
        print("-" * 20)
        print("按住空格开始新的录音。按 Ctrl+C 退出。")
        return

    filename = None
    server_relative_path = None
    try:
        os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
        if not local_audio_data:
            raise ValueError("Internal error: local_audio_data None after check")
        recording = np.concatenate(local_audio_data, axis=0)
        if recording.size == 0:
            raise ValueError("录音数据合并后为空")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_base = f"{FILENAME_BASE}_{timestamp}.wav"
        full_save_path = os.path.join(AUDIO_SAVE_DIR, filename_base)
        sf.write(full_save_path, recording, SAMPLERATE)
        print(f"录音已保存到: {full_save_path}")
        server_relative_path = filename_base

        transcribed_text = transcribe_audio_by_path(server_relative_path)

        if transcribed_text:
            print("*" * 100)
            print(f"\n转录结果: {transcribed_text}")

            display_status_popup("正在生成中...")
            llm_response = None
            try:
                 llm_response = get_llm_response_langchain(transcribed_text)
            finally:
                 close_status_popup() # Close "Generating" popup

            if llm_response:
                print(f"\nLLM 回复: {llm_response}")
                try:
                    if SHOW_LLM_RESPONSE_POPUP:
                        llm_popup_window = show_response_popup_tk(llm_response)
                        if not llm_popup_window:
                            print("警告: 无法创建LLM回复弹窗。", file=sys.stderr)

                    if ENABLE_TTS:
                        speak_text(llm_response)
                    else:
                        print("DEBUG: TTS reading disabled.")

                    if llm_popup_window and POPUP_AUTO_CLOSE and ENABLE_TTS:
                        print("DEBUG: Auto-closing LLM response popup...")
                        try:
                            llm_popup_window.destroy()
                            print("DEBUG: LLM response popup destroyed via auto-close.")
                        except tk.TclError as e_tk:
                             print(f"DEBUG: Info auto-closing LLM popup (likely closed): {e_tk}", file=sys.stderr)
                        except Exception as e_close:
                            print(f"Error auto-closing LLM popup: {e_close}", file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                    elif llm_popup_window:
                         if not POPUP_AUTO_CLOSE:
                             print("DEBUG: Auto-close disabled for LLM popup.")
                         elif not ENABLE_TTS:
                             print("DEBUG: TTS disabled, LLM popup requires manual close.")

                except Exception as e_popup_tts:
                     print(f"Error during LLM popup/TTS phase: {e_popup_tts}", file=sys.stderr)
                     traceback.print_exc(file=sys.stderr)

            else:
                print("\n未能获取 LLM 回复。")
                error_message = "Sorry, no LLM response."
                if ENABLE_TTS:
                    speak_text(error_message)
                else:
                    print(f"DEBUG: TTS Disabled. Error: {error_message}")
        else:
            print("\n未能转录音频。")
            error_message = "Sorry, couldn't transcribe."
            if ENABLE_TTS:
                speak_text(error_message)
            else:
                print(f"DEBUG: TTS Disabled. Error: {error_message}")

    except ValueError as ve:
        print(f"处理音频出错: {ve}", file=sys.stderr)
        error_message = "Audio processing error."
        if ENABLE_TTS:
            speak_text(error_message)
        else:
            print(f"DEBUG: TTS Disabled. Error: {error_message}")
    except Exception as e:
        print(f"处理/回复未知错误: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        error_message = "Unexpected error."
        if ENABLE_TTS:
            speak_text(error_message)
        else:
            print(f"DEBUG: TTS Disabled. Error: {error_message}")
    finally:
        print("-" * 20)
        try:
             llm_popup_open = llm_popup_window and llm_popup_window.winfo_exists()
             if llm_popup_open:
                  was_auto_close_intended = POPUP_AUTO_CLOSE and ENABLE_TTS
                  if not was_auto_close_intended:
                       if not ENABLE_TTS:
                           print("LLM回复弹窗仍然打开，可手动关闭或开始下一轮对话。")
                       else:
                           print("LLM回复弹窗仍然打开，需手动关闭。")
        except Exception as e_final_check:
             print(f"DEBUG: Error checking LLM popup status: {e_final_check}", file=sys.stderr)
        print("按住空格开始新的录音。")


# --- Transcription Function ---
def transcribe_audio_by_path(audio_path_relative_to_server_dir):
    print(f"请求 SenseVoice 转录: {audio_path_relative_to_server_dir} -> {SENSEVOICE_API_URL}")
    if not SENSEVOICE_API_URL:
        print("错误: SENSEVOICE_API_URL 未配置。", file=sys.stderr)
        return None
    payload = json.dumps({"audio_path": audio_path_relative_to_server_dir})
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(SENSEVOICE_API_URL, headers=headers, data=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if 'transcription' in result:
            print(f"转录成功: {audio_path_relative_to_server_dir}")
            return result['transcription']
        else:
            print(f"错误: API 响应缺少 'transcription': {response.text}", file=sys.stderr)
            return None
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 API ({SENSEVOICE_API_URL}).", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        print("错误: API 请求超时。", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"API 请求失败: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"错误: 无法解析 API JSON: {response.text if 'response' in locals() else 'N/A'}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"处理转录未知错误: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

# --- LLM Interaction ---
def get_llm_response_langchain(prompt_text):
    print(f"向 LLM 发送请求 (模型: {OPENAI_MODEL_NAME})...")
    if not OPENAI_API_KEY:
        print("错误: OPENAI_API_KEY 未配置。", file=sys.stderr)
        return None
    if not OPENAI_MODEL_NAME:
        print("错误: OPENAI_MODEL_NAME 未配置。", file=sys.stderr)
        return None
    try:
        openai_kwargs = {"openai_api_key": OPENAI_API_KEY, "model": OPENAI_MODEL_NAME, "temperature": 0.7}
        if OPENAI_BASE_URL:
            openai_kwargs["base_url"] = OPENAI_BASE_URL
        chat = ChatOpenAI(**openai_kwargs)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)]
        response = chat.invoke(messages)
        if response and hasattr(response, 'content') and response.content:
            print("LLM 回复接收成功。")
            return response.content.strip()
        else:
            print(f"错误: LLM 响应无效: {response}", file=sys.stderr)
            return None
    except AuthenticationError as e:
        print(f"OpenAI 认证失败: {e}", file=sys.stderr)
        return None
    except APIError as e:
        print(f"OpenAI API 错误: {e}", file=sys.stderr)
        return None
    except OutputParserException as e:
        print(f"LangChain 解析错误: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"LLM 交互未知错误: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

# --- Text-to-Speech Function ---
def speak_text(text_to_speak):
    global tts_finished_event, tts_engine, ENABLE_TTS
    if not ENABLE_TTS:
        print("DEBUG: speak_text called but TTS is disabled.")
        return
    if not text_to_speak:
        print("TTS: 无文本提供。")
        return
    if not tts_engine:
        print("TTS: 引擎未初始化。")
        return

    current_thread_id = threading.get_ident()
    print(f"DEBUG: speak_text - Preparing [Thread: {current_thread_id}]")
    temp_audio_file = None

    if not tts_finished_event.wait(timeout=10.0):
        print("警告: 等待上一个 TTS 操作超时。", file=sys.stderr)
        tts_finished_event.set()

    tts_finished_event.clear()
    print(f"DEBUG: speak_text - Cleared event [Thread: {current_thread_id}]")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_file = tmpfile.name

        print(f"DEBUG: Saving TTS to: {temp_audio_file} [Thread: {current_thread_id}]")
        tts_engine.save_to_file(text_to_speak, temp_audio_file)

        print(f"DEBUG: Before runAndWait (save) [Thread: {current_thread_id}]")
        tts_engine.runAndWait()
        print(f"DEBUG: After runAndWait (save) [Thread: {current_thread_id}]")

        if not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0:
            raise IOError(f"TTS engine failed to save audio to {temp_audio_file}")

        print(f"TTS: 正在播放... [Thread: {current_thread_id}]")
        try:
            audio_data_tts, file_samplerate = sf.read(temp_audio_file, dtype='float32')
            print(f"DEBUG: Playing {temp_audio_file} (Rate: {file_samplerate}) [Thread: {current_thread_id}]")
            sd.play(audio_data_tts, file_samplerate, blocking=True)
            sd.wait()
            print(f"DEBUG: sd.play/wait finished [Thread: {current_thread_id}]")
            print("TTS: 播放完毕。")
        except sf.SoundFileError as e_sf:
            print(f"Error reading TTS file {temp_audio_file}: {e_sf}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        except sd.PortAudioError as e_sd:
            print(f"Error playing TTS audio via sounddevice: {e_sd}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        except Exception as e_play:
            print(f"Unknown error during TTS playback: {e_play}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    except Exception as e:
        print(f"Error during TTS generation/setup [Thread: {current_thread_id}]: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        print(f"DEBUG: speak_text - finally [Thread: {current_thread_id}]")
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                print(f"DEBUG: Deleting temp TTS: {temp_audio_file}")
                os.remove(temp_audio_file)
            except Exception as e_del:
                print(f"Warning: Failed delete temp TTS {temp_audio_file}: {e_del}", file=sys.stderr)
        tts_finished_event.set()
        print(f"DEBUG: speak_text - Exiting finally (event set)")

# --- Keyboard Handlers ---
def handle_space_press(event):
    global tts_finished_event, recording_start_timer, is_recording, ENABLE_TTS
    if ENABLE_TTS and not tts_finished_event.is_set():
        print("TTS 正在播放，请稍候...")
        return
    with recording_lock:
        if is_recording or recording_start_timer is not None:
            return
        print(f"空格按下，将在 {RECORD_START_DELAY} 秒后开始录音...")
        recording_start_timer = threading.Timer(RECORD_START_DELAY, _initiate_recording_after_delay)
        recording_start_timer.daemon = True
        recording_start_timer.start()

def handle_space_release(event):
    global recording_start_timer, is_recording
    should_stop_and_save = False
    with recording_lock:
        if recording_start_timer is not None:
            print("空格释放，取消录音启动。")
            recording_start_timer.cancel()
            recording_start_timer = None
            print("-" * 20)
            print("按住空格录音。")
            return
        if is_recording:
            print("空格释放，停止录音...")
            should_stop_and_save = True
    if should_stop_and_save:
        processing_thread = threading.Thread(target=stop_recording_and_save, daemon=True)
        processing_thread.start()

# --- Main Program Entry Point ---
if __name__ == "__main__":
    print("程序启动。")
    print("-" * 30)
    print("依赖项: sounddevice, numpy, keyboard, soundfile, requests, python-dotenv, langchain, langchain-openai, openai, pyttsx3, tkinter")
    print("-" * 30)
    print("配置 (从 .env 加载):")
    print(f"  - 采样率: {SAMPLERATE} Hz")
    print(f"  - 通道数: {CHANNELS}")
    device_print = f"'{DEVICE}'" if isinstance(DEVICE, str) else DEVICE
    print(f"  - 音频设备: {device_print if DEVICE is not None else '系统默认'}")
    print(f"  - 录音延迟: {RECORD_START_DELAY} 秒")
    print(f"  - 保存目录: {os.path.abspath(AUDIO_SAVE_DIR)}")
    print(f"  - 文件名前缀: {FILENAME_BASE}")
    print(f"  - 显示LLM弹窗: {'启用' if SHOW_LLM_RESPONSE_POPUP else '禁用'}")
    print(f"  - 弹窗自动关闭 (TTS启用时): {'启用' if POPUP_AUTO_CLOSE else '禁用'}")
    print(f"  - 启用TTS阅读: {'是' if ENABLE_TTS else '否'}")
    print(f"  - SenseVoice API: {SENSEVOICE_API_URL or '未配置'}")
    print(f"  - OpenAI Key: {'已配置' if OPENAI_API_KEY else '未配置!'}")
    print(f"  - OpenAI Base URL: {OPENAI_BASE_URL or '默认 (OpenAI API)'}")
    print(f"  - OpenAI 模型: {OPENAI_MODEL_NAME}")
    print(f"  - 系统提示: '{SYSTEM_PROMPT[:50]}...'")
    tts_status = '已初始化' if tts_engine else ('初始化失败/禁用' if ENABLE_TTS else '已禁用')
    print(f"  - TTS 引擎状态: {tts_status}")
    print("-" * 30)
    print("操作指南:")
    print(f"  - 按住 [空格键] {RECORD_START_DELAY} 秒开始录音。")
    print("  - 松开 [空格键] 停止录音、处理并获取回复。")
    print("  - 在执行程序的终端按 [Ctrl+C] 键退出程序。")
    print("  - 当弹窗出现且TTS禁用时，可直接开始下一轮对话。")
    print("-" * 30)
    print("!! 重要提示:")
    print("  - 确保 `.env` 文件存在且包含所有需要的配置，特别是 OPENAI_API_KEY。")
    print("  - `keyboard` 库可能需要特殊权限。")
    print("  - 确保本地 API 服务 (若使用) 正在运行。")
    print("-" * 30)

    try:
        os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
    except OSError as e:
        print(f"错误: 无法创建目录 '{AUDIO_SAVE_DIR}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print("可用音频设备列表 (供 AUDIO_INPUT_DEVICE 参考):")
        print(sd.query_devices())
        print("-" * 30)
        default_input = sd.query_devices(kind='input')
        if default_input and isinstance(default_input, dict):
            print(f"系统默认输入: {default_input.get('name', 'N/A')} (Index: {default_input.get('index', 'N/A')})")
        elif isinstance(default_input, list) and len(default_input) > 0:
             print(f"系统默认输入 (列表0): {default_input[0].get('name', 'N/A')} (Index: {default_input[0].get('index', 'N/A')})")
        else:
            print("未找到系统默认输入设备。")
        print(f"当前使用设备: {device_print if DEVICE is not None else '系统默认'}")
        print("-" * 30)
    except Exception as e:
        print(f"查询音频设备出错: {e}", file=sys.stderr)
        print("-" * 30)

    try:
        keyboard.unhook_all()
        keyboard.on_press_key('space', handle_space_press, suppress=False)
        keyboard.on_release_key('space', handle_space_release)
        print("键盘监听器已注册。")
    except ImportError as e:
        print(f"\n错误：导入 keyboard 失败 - {e}。请运行 'pip install keyboard'。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n错误：无法注册键盘监听器！请检查权限。 {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print(f"准备就绪。按住空格键开始录音。")

    # --- Main Loop ---
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，开始退出...")
    except Exception as e:
        print(f"\n主循环发生错误: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # --- Cleanup Actions ---
        print("DEBUG: 开始最终清理...")
        close_status_popup() # Close status popup if open

        with recording_lock:
            if recording_start_timer is not None:
                print("DEBUG: 取消待处理的录音定时器。")
                recording_start_timer.cancel()

        if ENABLE_TTS and not tts_finished_event.is_set():
             print("DEBUG: TTS 仍在进行，尝试停止播放...")
             try:
                 sd.stop()
                 print("DEBUG: sd.stop() called.")
             except Exception as e_sd_stop:
                 print(f"sd.stop() 出错: {e_sd_stop}", file=sys.stderr)
             tts_finished_event.set() # Ensure event is set

        try:
            keyboard.unhook_all()
            print("DEBUG: 键盘监听已移除。")
        except Exception as e_unhook:
            print(f"移除监听出错: {e_unhook}", file=sys.stderr)

        final_check_stream = None
        with recording_lock:
            if is_recording and stream:
                final_check_stream = stream
                is_recording = False # Mark as stopped

        if final_check_stream:
            print("警告: 退出时录音流仍在运行。强制停止...", file=sys.stderr)
            try:
                if final_check_stream.active:
                    final_check_stream.stop()
                if not final_check_stream.closed:
                    final_check_stream.close()
                print("DEBUG: 成功关闭剩余音频流。")
            except Exception as e_final_stop:
                print(f"最终流关闭出错: {e_final_stop}", file=sys.stderr)
            with recording_lock: # Reset global stream var
                stream = None

        print("程序结束。")
        sys.exit(0)