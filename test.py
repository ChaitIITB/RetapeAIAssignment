class VoicemailManager:
    def __init__(self):
        self.state = "LISTENING"
        self.silence_start_time = None
        self.VAD_THRESHOLD = 1.5 # seconds

    def process_audio_chunk(self, audio_chunk):
        # 1. Check for Beep (Priority)
        if self.detect_beep(audio_chunk):
            return self.trigger_playback("BEEP_DETECTED")

        # 2. Check for Voice Activity
        is_speech = self.vad_model(audio_chunk)

        if is_speech:
            self.silence_start_time = None # Reset timer if Mike speaks
        else:
            if self.silence_start_time is None:
                self.silence_start_time = now() # Start timer
            
            # Check if silence has lasted long enough
            elapsed_silence = now() - self.silence_start_time
            if elapsed_silence > self.VAD_THRESHOLD:
                return self.trigger_playback("SILENCE_THRESHOLD")

    def detect_beep(self, chunk):
        # Simple FFT to check for >1000Hz pure tone
        pass


if __name__ == "__main__":
    vm = VoicemailManager()
    # Simulate audio processing loop
    while True:
        audio_chunk = get_next_audio_chunk()
        vm.process_audio_chunk(audio_chunk)