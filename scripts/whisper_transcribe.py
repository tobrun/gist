#!/usr/bin/env python3
import whisper
import pyaudio
import numpy as np
import wave
import tempfile
import os
import time
import argparse
import subprocess
from pathlib import Path
import signal
import sys

class AudioTranscriber:
    def __init__(self, model_name="tiny", record_seconds=3, list_only=False):
        self.model_name = model_name
        self.record_seconds = record_seconds
        self.list_only = list_only
        
        # Audio settings - smaller chunk for more responsive updates
        self.chunk = 512
        self.format = pyaudio.paInt16
        self.rate = 16000
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Setup signal handling for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        print("\n👋 Stopping transcription...")
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        sys.exit(0)
        
    def list_devices(self):
        """List all available audio input devices"""
        print("\n📢 Available audio devices:")
        print("-" * 60)
        
        valid_inputs = []
        for i in range(self.p.get_device_count()):
            try:
                info = self.p.get_device_info_by_index(i)
                max_input = int(info.get('maxInputChannels', 0))
                
                if max_input > 0:
                    valid_inputs.append((i, info['name'], max_input))
                    print(f"{i}: {info['name']} (Inputs: {max_input})")
            except Exception as e:
                print(f"⚠️  Error getting info for device {i}: {e}")
                
        if not valid_inputs:
            print("❌ No valid input devices found!")
            return None
            
        print("-" * 60)
        print("\n💡 For system audio capture, look for 'pulse', 'pipewire', or 'Monitor of...'")
        return valid_inputs
    
    def get_pulse_monitor_devices(self):
        """Get PulseAudio monitor devices using pactl command"""
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
                
            return monitors
        except Exception as e:
            print(f"⚠️  Could not get PulseAudio monitor devices: {e}")
            return []
            
    def setup_audio_source(self):
        """Set up the audio source for recording"""
        valid_inputs = self.list_devices()
        pulse_monitors = self.get_pulse_monitor_devices()
        
        if self.list_only:
            print("\nDevice listing complete. Exiting as requested.")
            sys.exit(0)
            
        if not valid_inputs:
            print("❌ No valid input devices. Exiting.")
            sys.exit(1)
            
        # Determine the best source to use
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
        
        device_index = None
        device_name = None
        
        # Try to use a sensible default
        if pulse_index is not None:
            device_index = pulse_index
            device_name = "pulse"
        elif pipewire_index is not None:
            device_index = pipewire_index
            device_name = "pipewire"
        elif default_index is not None:
            device_index = default_index
            device_name = "default"
            
        # Let user choose if they want to
        use_default = False
        if device_index is not None:
            print(f"\n💡 Recommended device: {device_index} ({device_name})")
            try:
                choice = input("Use this device? [Y/n]: ").strip().lower()
                use_default = choice == '' or choice.startswith('y')
            except EOFError:
                use_default = True
                
        if not use_default:
            try:
                device_index = int(input("\n🔢 Enter device index to use: "))
            except ValueError:
                print("❌ Invalid input. Using first available device.")
                device_index = valid_inputs[0][0]
                
        # Get device info and set up channels
        try:
            device_info = self.p.get_device_info_by_index(device_index)
            self.channels = min(int(device_info.get('maxInputChannels', 1)), 2)  # Limit to 1 or 2 channels
            self.device_index = device_index
            self.device_name = device_info.get('name', f"Device {device_index}")
            
            print(f"\n✅ Using device: {self.device_name}")
            print(f"   Channels: {self.channels}")
            
            if pulse_monitors and (pulse_index == device_index or pipewire_index == device_index):
                print("\n💡 TIP: For capturing system audio, open pavucontrol in another terminal:")
                print("   $ pavucontrol")
                print("   Then go to 'Recording' tab, find this Python process, and set source to 'Monitor of [your output device]'")
                
        except Exception as e:
            print(f"❌ Error getting device info: {e}")
            print("Trying fallback to default device...")
            self.device_index = None  # Use default device
            self.channels = 1
            self.device_name = "Default Device"
            
        return True
        
    def setup_model(self):
        """Load the Whisper model"""
        print(f"\n⏳ Loading Whisper model '{self.model_name}'...")
        try:
            # Use faster options for quicker processing
            self.model = whisper.load_model(self.model_name)
            # Set language to English if using tiny model (much faster)
            if self.model_name == "tiny":
                print("✅ Model loaded! Using English-only mode for faster processing")
                self.language = "en"
            else:
                print("✅ Model loaded successfully!")
                self.language = None
            return True
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            return False
            
    def open_stream(self):
        """Open the audio stream for recording"""
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
            return True
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            print("Troubleshooting tips:")
            print("1. Try running with --list to see available devices")
            print("2. Make sure you have the right permissions")
            print("3. Try using a different device index")
            return False
            
    def save_to_temp_file(self, audio_data):
        """Save audio data to a temporary WAV file"""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)  # Whisper expects mono
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
            return tmp_path
        except Exception as e:
            print(f"❌ Error saving audio to temp file: {e}")
            try:
                os.unlink(tmp_path)
            except:
                pass
            return None
            
    def record_and_transcribe(self):
        """Main loop for recording and transcribing audio"""
        if not self.setup_audio_source():
            return
            
        if not self.setup_model():
            return
            
        if not self.open_stream():
            return
            
        # Wait for pavucontrol setup if needed
        input("\n🎧 Press Enter to start transcribing...")
        
        print(f"\n🎤 Listening to audio from '{self.device_name}'")
        print(f"🛑 Press Ctrl+C to stop\n")
        print("-" * 60)
        
        try:
            while True:
                # Collect audio frames
                frames = []
                for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                    try:
                        data = self.stream.read(self.chunk, exception_on_overflow=False)
                        frames.append(data)
                    except Exception as e:
                        print(f"\r⚠️  Audio read error: {e}", end="")
                        continue
                
                # Convert to numpy array and process
                samples = np.frombuffer(b''.join(frames), dtype=np.int16)
                
                # Convert to mono if stereo
                if self.channels == 2:
                    # Simple downmix to mono by averaging channels
                    samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # Skip silent chunks - less aggressive threshold
                if np.abs(samples).mean() < 30:
                    continue
                
                # Save to temporary WAV file for Whisper
                tmp_path = self.save_to_temp_file(samples.tobytes())
                if not tmp_path:
                    continue
                
                try:
                    # Transcribe with Whisper - use optimized settings
                    options = {
                        "fp16": False,
                        "language": self.language,
                        "beam_size": 1 if self.model_name == "tiny" else 5,
                        "best_of": 1 if self.model_name == "tiny" else 5,
                        "initial_prompt": "Transcribe the following audio:"
                    }
                    result = self.model.transcribe(tmp_path, **options)
                    
                    text = result.get("text", "").strip()
                    if text:
                        # Clear the current line and print the text
                        print("\r" + " " * 80, end="\r")  # Clear line
                        print(f"{text}")
                except Exception as e:
                    print(f"\r❌ Transcription error: {e}", end="")
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except KeyboardInterrupt:
            print("\n👋 Stopping transcription...")
        finally:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
            print("✅ Transcription stopped.")

def main():
    parser = argparse.ArgumentParser(description="Transcribe system audio using Whisper")
    parser.add_argument("--model", type=str, default="large", choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use (default: tiny)")
    parser.add_argument("--seconds", type=int, default=3, 
                      help="Length of audio chunks to process in seconds (default: 3)")
    parser.add_argument("--list", action="store_true", 
                      help="List available audio devices and exit")
    args = parser.parse_args()
    
    # Pre-check for common environment issues
    if os.environ.get('LD_PRELOAD') is None:
        print("⚠️  LD_PRELOAD not set - this might cause library issues.")
        print("   Consider using: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6\n")
    
    # Create and run transcriber
    transcriber = AudioTranscriber(
        model_name=args.model,
        record_seconds=args.seconds,
        list_only=args.list
    )
    
    transcriber.record_and_transcribe()

if __name__ == "__main__":
    main()