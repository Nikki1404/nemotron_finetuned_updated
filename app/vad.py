from collections import deque
import numpy as np


class AdaptiveEnergyVAD:
    """
    Lightweight adaptive energy VAD tuned for realtime ASR.
    Lower threshold + larger pre-roll avoids cutting first words.
    """

    def __init__(self, sample_rate: int, frame_ms: int, start_margin: float, min_noise_rms: float, pre_speech_ms: int):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.start_margin = start_margin
        self.min_noise_rms = min_noise_rms
        self.frame_samples = int(self.sr * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.pre_frames = max(1, int(pre_speech_ms / frame_ms))
        self.ring = deque(maxlen=self.pre_frames)
        self.in_speech = False
        self.noise_rms = min_noise_rms

    def reset(self):
        self.ring.clear()
        self.in_speech = False
        self.noise_rms = self.min_noise_rms

    def _rms(self, pcm16: bytes) -> float:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def push_frame(self, frame_pcm16: bytes):
        e = self._rms(frame_pcm16)
        if not self.in_speech:
            alpha = 0.97
            self.noise_rms = max(self.min_noise_rms, alpha * self.noise_rms + (1.0 - alpha) * e)

        threshold = max(self.min_noise_rms, self.noise_rms) * self.start_margin
        is_speech = e >= threshold
        self.ring.append(frame_pcm16)

        pre_roll = None
        if (not self.in_speech) and is_speech:
            self.in_speech = True
            pre_roll = b"".join(self.ring)

        return is_speech, pre_roll
