#!/usr/bin/env python3
"""
Simple voicemail timestamp detector.
Usage: python detector.py <wav_file_path>
Output: Just the timestamp in seconds (e.g., "11.42")
"""

import sys
import torch
import numpy as np
import wave
from pathlib import Path

class VoicemailDetector:
    """Detects when to play message: beep supersedes everything, then last silence after speech"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.silence_threshold = 1.0  # seconds of silence to consider end of speech
        self.beep_duration_threshold = 0.2  # seconds of sustained beep to trigger (lowered from 0.5)

        # Load Silero VAD
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )

        # State tracking
        self.reset()

    def reset(self):
        """Reset detector state for new file"""
        self.speech_started = False
        self.in_speech = False
        self.silence_chunks = 0
        self.last_silence_trigger_time = None
        self.elapsed_time = 0
        self.chunk_duration = 0
        self.beep_chunks = 0
        self.beep_trigger_time = None
        self.vad_model.reset_states()

    def process_chunk(self, audio_int16):
        """Process audio chunk, return trigger if beep detected"""
        audio_float = self._normalize(audio_int16)

        # Calculate timing (first chunk only)
        if self.chunk_duration == 0:
            self.chunk_duration = len(audio_int16) / self.sample_rate
        self.elapsed_time += self.chunk_duration

        # Check for beep tone FIRST (highest priority)
        if self._is_beep(audio_float):
            self.beep_chunks += 1
            beep_time = self.beep_chunks * self.chunk_duration
            if beep_time >= self.beep_duration_threshold:
                # BEEP detected - store it
                self.beep_trigger_time = self.elapsed_time
        else:
            self.beep_chunks = 0

        # Only check speech/silence if no beep
        # Check for speech using Silero VAD
        audio_tensor = torch.from_numpy(audio_float).float()
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob > 0.5

        if is_speech:
            if not self.speech_started:
                self.speech_started = True

            if not self.in_speech:
                self.in_speech = True
                # Reset silence counter and last trigger when speech resumes
                self.silence_chunks = 0
                self.last_silence_trigger_time = None
        else:
            # Silence detected
            if self.speech_started and self.in_speech:
                self.silence_chunks += 1
                silence_time = self.silence_chunks * self.chunk_duration

                if silence_time >= self.silence_threshold:
                    # Mark this as the current trigger point
                    self.last_silence_trigger_time = self.elapsed_time - silence_time + self.chunk_duration
                    self.in_speech = False

        return None, None  # No immediate trigger

    def get_final_trigger(self):
        """Return the final silence trigger time after all speech has ended (beep handled immediately)"""
        # Only return silence trigger - beep triggers are handled immediately in process_chunk
        if self.speech_started and self.last_silence_trigger_time is not None:
            return self.last_silence_trigger_time, "SILENCE"

        return None, None

    def _is_beep(self, audio):
        """Detect ~1000Hz beep using FFT"""
        # Compute frequency spectrum
        spectrum = np.fft.rfft(audio)
        magnitudes = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(audio), d=1/self.sample_rate)

        # Find peak frequency
        peak_idx = np.argmax(magnitudes)
        peak_freq = freqs[peak_idx]

        # Check if peak is in beep range (850-1150 Hz)
        if 850 <= peak_freq <= 1150:
            # Check energy concentration (pure tone has high ratio)
            avg_energy = np.mean(magnitudes)
            peak_energy = magnitudes[peak_idx]
            ratio = peak_energy / avg_energy if avg_energy > 0 else 0

            # Lowered from 15 to 5 for real-world phone audio
            return ratio > 5

        return False

    @staticmethod
    def _normalize(audio_int16):
        """Convert int16 to float32 range [-1, 1]"""
        return audio_int16.astype(np.float32) / 32768.0

def process_wav_file(filepath):
    """Process WAV file and return trigger timestamp"""
    try:
        with wave.open(filepath, 'rb') as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            total_frames = wav.getnframes()

            # Determine target sample rate for VAD (must be 8000 or 16000)
            if rate in [8000, 16000]:
                target_rate = rate
            else:
                target_rate = 16000  # Default to 16kHz

            # Determine chunk size in samples for VAD (Silero requirement)
            vad_chunk_size = 256 if target_rate == 8000 else 512

            # Calculate chunk duration and corresponding frames from original file
            chunk_duration = vad_chunk_size / target_rate  # seconds
            chunk_frames_original = int(chunk_duration * rate)  # frames to read from original file

            detector = VoicemailDetector(sample_rate=target_rate)
            detector.reset()

            # Calculate resampling ratio if needed
            resample_ratio = target_rate / rate if rate != target_rate else 1.0

            # Process in true streaming fashion
            frames_processed = 0
            beep_times = []  # Track all beep detections
            silence_times = []  # Track all silence detections
            
            while frames_processed < total_frames:
                # Read chunk from file (based on original sample rate)
                frames_to_read = min(chunk_frames_original, total_frames - frames_processed)
                chunk_bytes = wav.readframes(frames_to_read)

                if not chunk_bytes:
                    break

                # Convert to int16 array
                audio_chunk = np.frombuffer(chunk_bytes, dtype=np.int16)

                # Stereo to mono
                if channels == 2:
                    audio_chunk = audio_chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Resample if needed
                if resample_ratio != 1.0:
                    # Simple linear interpolation resampling
                    original_length = len(audio_chunk)
                    new_length = int(original_length * resample_ratio)
                    indices = np.linspace(0, original_length - 1, new_length)
                    audio_chunk = np.interp(indices, np.arange(original_length), audio_chunk).astype(np.int16)

                # Pad if needed (for last chunk)
                if len(audio_chunk) < vad_chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, vad_chunk_size - len(audio_chunk)))

                # Process the chunk
                trigger_time, trigger_type = detector.process_chunk(audio_chunk)
                
                # Track detections
                if trigger_time is not None and trigger_type == "BEEP":
                    beep_times.append(trigger_time)
                    print(f"DEBUG: BEEP detected at {trigger_time:.2f}s", file=sys.stderr)
                elif trigger_type == "SILENCE":
                    silence_times.append(trigger_time)
                    print(f"DEBUG: SILENCE detected at {trigger_time:.2f}s", file=sys.stderr)

                frames_processed += frames_to_read

            # Priority: beep first, then last silence
            if detector.beep_trigger_time is not None:
                return detector.beep_trigger_time + 1.5  # Beep + delay
            elif detector.last_silence_trigger_time is not None:
                return detector.last_silence_trigger_time + 1.5  # Last silence + delay
            else:
                # No trigger - return end of file
                return total_frames / rate

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python detector.py <wav_file_path>", file=sys.stderr)
        sys.exit(1)

    wav_file = sys.argv[1]
    if not Path(wav_file).exists():
        print(f"File not found: {wav_file}", file=sys.stderr)
        sys.exit(1)

    timestamp = process_wav_file(wav_file)
    print(f"{timestamp:.2f}")

if __name__ == "__main__":
    main()