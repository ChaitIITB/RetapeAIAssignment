import torch
import numpy as np
import logging
import wave
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class VoicemailDetector:
    """Detects when to play message: after the LAST silence following speech"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.silence_threshold = 1.0  # seconds of silence to consider end of speech
        self.beep_duration_threshold = 0.5  # seconds of sustained beep to trigger
        
        # Load Silero VAD
        logging.info("Loading Silero VAD...")
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
                # BEEP supersedes everything - return immediately
                self.beep_trigger_time = self.elapsed_time
                logging.info(f" BEEP detected at {self.beep_trigger_time:.2f}s (sustained {beep_time:.2f}s)")
                return self.beep_trigger_time, "BEEP"
        else:
            self.beep_chunks = 0
        
        # Only check speech/silence if no beep
        # Check for speech using Silero VAD
        audio_tensor = torch.from_numpy(audio_float).float()
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        
        is_speech = speech_prob > 0.5
        
        if is_speech:
            if not self.speech_started:
                logging.info(f"Voice activity started (prob: {speech_prob:.2f})")
                self.speech_started = True
            
            if not self.in_speech:
                logging.info(f"Speech resumed at {self.elapsed_time:.2f}s")
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
                    logging.info(f"Silence trigger updated to {self.last_silence_trigger_time:.2f}s ({silence_time:.2f}s quiet)")
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
            
            # Lowered from 15 to 10 for real-world phone audio
            return ratio > 10
        
        return False
    
    @staticmethod
    def _normalize(audio_int16):
        """Convert int16 to float32 range [-1, 1]"""
        return audio_int16.astype(np.float32) / 32768.0

# ==========================================
# FILE PROCESSING
# ==========================================

def load_wav(filepath, target_rate=16000):
    """Load WAV file and resample if needed"""
    with wave.open(filepath, 'rb') as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.getnframes()
        data = wav.readframes(frames)
        
        # Convert to int16 array
        audio = np.frombuffer(data, dtype=np.int16)
        
        # Stereo to mono
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Resample if needed (Silero requires 8k or 16k)
        if rate not in [8000, 16000]:
            logging.info(f"  Resampling {rate}Hz  {target_rate}Hz")
            duration = len(audio) / rate
            new_length = int(duration * target_rate)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio.astype(np.float32))
            audio = audio.astype(np.int16)
            rate = target_rate
        
        return audio, rate

def process_file(filepath, detector):
    """Process single voicemail file in true streaming fashion"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: {Path(filepath).name}")
    logging.info(f"{'='*60}")
    
    # Open WAV file for streaming
    with wave.open(filepath, 'rb') as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        total_frames = wav.getnframes()
        bytes_per_sample = wav.getsampwidth()
        bytes_per_frame = bytes_per_sample * channels
        
        # Determine target sample rate for VAD (must be 8000 or 16000)
        if rate in [8000, 16000]:
            target_rate = rate
        else:
            target_rate = 16000  # Default to 16kHz
            logging.info(f"  Will resample from {rate}Hz to {target_rate}Hz for VAD compatibility")
        
        # Determine chunk size in samples for VAD (Silero requirement)
        vad_chunk_size = 256 if target_rate == 8000 else 512
        
        # Calculate chunk duration and corresponding frames from original file
        chunk_duration = vad_chunk_size / target_rate  # seconds
        chunk_frames_original = int(chunk_duration * rate)  # frames to read from original file
        
        detector.sample_rate = target_rate
        detector.reset()
        
        # Calculate resampling ratio if needed
        resample_ratio = target_rate / rate if rate != target_rate else 1.0
        
        # Process in true streaming fashion - read and process chunks sequentially
        frames_processed = 0
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
            
            # If beep detected, convert resampled time back to original time
            if trigger_time is not None:
                # No conversion needed - resampling preserves time duration
                # Add 1.5s delay for natural conversation pause
                delayed_trigger_time = trigger_time + 1.5
                logging.info(f"\n→ MESSAGE TRIGGER at {delayed_trigger_time:.2f}s (original: {trigger_time:.2f}s + 1.5s delay)")
                return delayed_trigger_time, trigger_type
            
            frames_processed += frames_to_read
    
    # No beep found, check for final silence trigger
    final_trigger, trigger_type = detector.get_final_trigger()
    if final_trigger is not None:
        # No conversion needed - resampling preserves time duration
        # Add 1.5s delay for natural conversation pause
        delayed_final_trigger = final_trigger + 1.5
        logging.info(f"\n→ MESSAGE TRIGGER at {delayed_final_trigger:.2f}s (original: {final_trigger:.2f}s + 1.5s delay)")
        return delayed_final_trigger, trigger_type
    else:
        # No trigger - play at end
        logging.warning("⚠ No trigger detected")
        return total_frames / rate, "END"

def main():
    """Process voicemail files - accepts file path as argument or processes all in voicedata"""
    
    # Check if a specific file is provided
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
        if not Path(wav_file).exists():
            logging.error(f"❌ File not found: {wav_file}")
            return
        
        # Process single file
        detector = VoicemailDetector(sample_rate=16000)
        time, reason = process_file(wav_file, detector)
        print(f"Result: {time:.2f}s | {reason}")
        return
    
    # Default: process all files in voicedata folder
    detector = VoicemailDetector(sample_rate=16000)
    
    # Find voicedata folder
    voicedata = Path(__file__).parent / "voicedata"
    if not voicedata.exists():
        logging.error(f"❌ Folder not found: {voicedata}")
        return
    
    wav_files = sorted(voicedata.glob("*.wav"))
    if not wav_files:
        logging.error(f"❌ No .wav files in {voicedata}")
        return
    
    logging.info(f"\n{'#'*60}")
    logging.info(f"CLEARPATH FINANCE - VOICEMAIL SYSTEM")
    logging.info(f"{'#'*60}")
    logging.info(f"Processing {len(wav_files)} files\n")
    
    # Process each file
    results = []
    for wav in wav_files:
        time, reason = process_file(str(wav), detector)
        results.append((wav.name, time, reason))
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY")
    logging.info(f"{'='*60}")
    for name, time, reason in results:
        logging.info(f"{name:25s} | {time:5.2f}s | {reason}")
    
    beeps = sum(1 for _, _, r in results if r == "BEEP")
    silences = sum(1 for _, _, r in results if r == "SILENCE")
    logging.info(f"\nBeeps: {beeps} | Silence: {silences} | Total: {len(results)}\n")

if __name__ == "__main__":
    main()
