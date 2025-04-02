#!/usr/bin/env python3
import numpy as np
import pyaudio
import argparse
import os
import signal
import sys
import threading
import time
import queue
from collections import deque
import subprocess

# Buffer to hold audio
audio_buffer = queue.Queue()
# Flag to control recording
recording = True
# For continuous output
last_text = ""

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global recording
    print("\n\nStopping transcription...")
    recording = False
    time.sleep(1)  # Give threads time to clean up
    sys.exit(0)

def audio_capture_thread(device_index, chunk_size=1024, sample_rate=16000, channels=1):
    """Thread to continuously capture audio"""
    global recording
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
        
        print("🎤 Audio capture started")
        
        while recording:
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_buffer.put(data)
            except Exception as e:
                print(f"\rAudio read error: {e}", end="")
                continue
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("🎤 Audio capture stopped")

def transcription_thread(model_size="tiny", language="en", use_cpu=False):
    """Thread to continuously transcribe audio from the buffer"""
    global recording, last_text
    
    # Import faster_whisper here to keep it isolated
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper not installed. Install with:")
        print("pip install faster-whisper")
        recording = False
        return
    
    print(f"⏳ Loading model {model_size}...")
    
    # Choose device
    compute_type = "int8" if use_cpu else "float16"
    device = "cpu" if use_cpu else "cuda"
    
    # Try to load the model
    try:
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=os.path.expanduser("~/.cache/whisper")
        )
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if not use_cpu and "CUDA" in str(e):
            print("Try running with --cpu flag if you don't have a compatible GPU")
        recording = False
        return
    
    # Audio processing settings
    sample_width = 2  # 16-bit audio
    sample_rate = 16000
    sliding_window_len = 3.0  # seconds
    sliding_window_size = int(sliding_window_len * sample_rate)
    update_interval = 0.2  # seconds
    update_samples = int(update_interval * sample_rate)
    
    # Buffer for continuous processing
    audio_ring = deque(maxlen=sliding_window_size)
    
    # Ensure buffer starts with silence
    audio_ring.extend([0] * sliding_window_size)
    
    # Transcription options
    transcribe_options = {
        "language": language,
        "beam_size": 1,
        "best_of": 1,
        "temperature": 0,
        "condition_on_previous_text": True,
        "initial_prompt": "Transcribing system audio:"
    }
    
    # For throttling output
    last_update_time = time.time()
    
    print("🔄 Transcription started")
    
    while recording:
        # Get all available audio data
        while not audio_buffer.empty() and recording:
            data = audio_buffer.get()
            # Convert to numpy array
            samples = np.frombuffer(data, dtype=np.int16)
            # Add to ring buffer
            audio_ring.extend(samples)
        
        # Wait until we have enough data
        if len(audio_ring) < sliding_window_size:
            time.sleep(0.1)
            continue
            
        # Throttle updates
        current_time = time.time()
        if current_time - last_update_time < 0.5:  # Update at most twice per second
            time.sleep(0.05)
            continue
            
        last_update_time = current_time
        
        # Create array from deque
        audio_array = np.array(list(audio_ring), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Skip if audio is mostly silence
        if np.abs(audio_array).mean() < 0.01:
            continue
        
        # Transcribe current audio window
        try:
            segments, _ = model.transcribe(
                audio_array, 
                **transcribe_options
            )
            
            # Process segments
            new_text = ""
            for segment in segments:
                new_text += segment.text
            
            # Only update if we have new text that's different
            if new_text and new_text.strip() != last_text:
                # Clear line and print new text
                print("\r" + " " * 100, end="\r")  # Clear line
                print(new_text.strip())
                last_text = new_text.strip()
            
            # Small sleep to give other threads CPU time
            time.sleep(0.01)
            
        except Exception as e:
            print(f"\r⚠️ Transcription error: {e}", end="")
    
    print("🔄 Transcription stopped")

def list_audio_devices():
    """List available audio devices"""
    p = pyaudio.PyAudio()
    
    print("\n📢 Available audio devices:")
    print("-" * 60)
    
    valid_inputs = []
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            max_input = int(info.get('maxInputChannels', 0))
            
            if max_input > 0:
                valid_inputs.append((i, info['name'], max_input))
                print(f"{i}: {info['name']} (Inputs: {max_input})")
        except Exception as e:
            print(f"⚠️  Error getting info for device {i}: {e}")
            
    p.terminate()
            
    if not valid_inputs:
        print("❌ No valid input devices found!")
        return None
        
    print("-" * 60)
    
    # Try to find pulse/pipewire
    pulse_index = None
    pipewire_index = None
    default_index = None
    
    for idx, name, _ in valid_inputs:
        lower_name = name.lower()
        if 'pulse' in lower_name:
            pulse_index = idx
        elif 'pipewire' in lower_name:
            pipewire_index = idx
        elif 'default' in lower_name:
            default_index = idx
    
    if pulse_index is not None:
        print(f"\n💡 Recommended device: {pulse_index} (pulse)")
        return pulse_index
    elif pipewire_index is not None:
        print(f"\n💡 Recommended device: {pipewire_index} (pipewire)")
        return pipewire_index
    elif default_index is not None:
        print(f"\n💡 Recommended device: {default_index} (default)")
        return default_index
    elif valid_inputs:
        print(f"\n💡 First available device: {valid_inputs[0][0]}")
        return valid_inputs[0][0]
    
    return None

def get_pulse_monitor_devices():
    """Get PulseAudio monitor devices"""
    try:
        output = subprocess.check_output(["pactl", "list", "sources"], universal_newlines=True)
        lines = output.split('\n')
        
        monitors = []
        current_name = None
        current_desc = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Name:'):
                current_name = line.split(':', 1)[1].strip()
            elif line.startswith('Description:'):
                current_desc = line.split(':', 1)[1].strip()
                if 'Monitor' in current_desc and current_name:
                    monitors.append((current_name, current_desc))
                    current_name = None
                    current_desc = None
                    
        if monitors:
            print("\n🎧 PulseAudio Monitor devices:")
            print("-" * 60)
            for i, (name, desc) in enumerate(monitors):
                print(f"  {i}: {desc}")
                print(f"     Source name: {name}")
            print("-" * 60)
            
            print("\n💡 TIP: For capturing system audio, open pavucontrol in another terminal:")
            print("   $ pavucontrol")
            print("   Then go to 'Recording' tab, find this Python process, and set source to 'Monitor of [your output device]'")
        
    except Exception as e:
        print(f"⚠️  Could not get PulseAudio monitor devices: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing.append("faster-whisper")
    
    if missing:
        print("❌ Missing dependencies: " + ", ".join(missing))
        print("\nInstall required packages with:")
        if "faster-whisper" in missing:
            print("pip install faster-whisper")
        if "pyaudio" in missing:
            print("pip install pyaudio")
            print("Note: You might need system packages like portaudio19-dev")
        
        return False
    
    return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stream audio transcription using faster-whisper")
    parser.add_argument("--model", type=str, default="tiny", 
                      choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"],
                      help="Whisper model size (default: tiny)")
    parser.add_argument("--device", type=int, default=None, 
                      help="Audio input device index (default: auto-detect)")
    parser.add_argument("--language", type=str, default="en", 
                      help="Language code (e.g., 'en', 'fr', 'auto')")
    parser.add_argument("--list", action="store_true", 
                      help="List available audio devices and exit")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU inference even if GPU is available")
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # List devices if requested
    if args.list:
        list_audio_devices()
        get_pulse_monitor_devices()
        return
    
    # Select input device
    device_index = args.device
    if device_index is None:
        device_index = list_audio_devices()
        get_pulse_monitor_devices()
        if device_index is None:
            print("❌ No suitable audio device found.")
            return
    
    # Check environment for LD_PRELOAD
    if os.environ.get("LD_PRELOAD") is None:
        print("\n⚠️ Warning: LD_PRELOAD not set. If you encounter library errors, try:")
        print("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    
    # Confirm with user
    try:
        input("\n⏳ Press Enter to start streaming transcription (Ctrl+C to stop)...")
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return
    
    # Start threads
    audio_thread = threading.Thread(
        target=audio_capture_thread, 
        args=(device_index,),
        daemon=True
    )
    
    transcribe_thread = threading.Thread(
        target=transcription_thread,
        args=(args.model, args.language, args.cpu),
        daemon=True
    )
    
    audio_thread.start()
    transcribe_thread.start()
    
    # Wait for threads to finish (they'll run until Ctrl+C)
    try:
        while audio_thread.is_alive() and transcribe_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        recording = False
    
    print("Done!")

if __name__ == "__main__":
    main()  