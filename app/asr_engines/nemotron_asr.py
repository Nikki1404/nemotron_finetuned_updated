import time
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from app.asr_engines.base import ASREngine, EngineCaps


def safe_text(h: Any) -> str:
    if h is None:
        return ""

    if isinstance(h, str):
        return h

    if isinstance(h, (list, tuple)) and len(h) > 0:
        return safe_text(h[0])

    if hasattr(h, "text"):
        try:
            return h.text or ""
        except Exception:
            return ""

    try:
        return str(h)
    except Exception:
        return ""


@dataclass
class StreamTimings:
    preproc_sec: float = 0.0
    infer_sec: float = 0.0
    flush_sec: float = 0.0


class NemotronStreamingASR(ASREngine):

    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(
        self,
        model_name: str,
        device: str,
        sample_rate: int,
    ):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate

        # Nemotron streaming/finalization defaults tuned for telephony realtime ASR.
        # These can still be overridden using Docker/env variables.
        self.context_right = int(os.getenv("CONTEXT_RIGHT", "2"))
        self.end_silence_ms = int(os.getenv("NEMO_END_SILENCE_MS", "1000"))
        self.min_utt_ms = int(os.getenv("NEMO_MIN_UTT_MS", "200"))
        self.finalize_pad_ms = int(os.getenv("FINALIZE_PAD_MS", "1000"))
        self.max_symbols = int(os.getenv("NEMO_MAX_SYMBOLS", "15"))

        self.model = None

        self.shift_frames: int = 0
        self.pre_cache_frames: int = 0
        self.hop_samples: int = 0
        self.drop_extra: int = 0
        self._frame_stride_sec: float = 0.01

    @property
    def chunk_samples(self) -> int:
        if self.shift_frames <= 0 or self.hop_samples <= 0:
            return int(0.08 * self.sr)
        return int(self.shift_frames * self.hop_samples)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        if self.device == "cuda":
            return x.cuda(non_blocking=True)
        return x.cpu()

    def _move_cache_to_device(
        self,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        c0, c1, c2 = cache
        return (
            self._to_device(c0),
            self._to_device(c1),
            self._to_device(c2),
        )

    def set_language(self, language: str):
        """
        Set language prompt for the next session.
        Called per WebSocket connection before session starts.

        Note:
        This is not thread-safe across concurrent sessions with different languages
        because the prompt is set on the shared model object.
        """
        if self.model is None:
            return

        try:
            self.model.set_inference_prompt(language)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"set_inference_prompt('{language}') failed: {e}"
            )

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        if self.model_name.endswith(".nemo"):
            self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from(
                self.model_name,
                map_location="cpu",
            )
        else:
            self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.from_pretrained(
                self.model_name,
                map_location="cpu",
            )

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        try:
            self.model.encoder.set_default_att_context_size(
                [70, int(self.context_right)]
            )
        except Exception:
            self.model.encoder.set_default_att_context_size(
                (70, int(self.context_right))
            )

        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create({
                "strategy": "greedy",
                "greedy": {
                    "max_symbols": int(self.max_symbols),
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                },
            })
        )

        self.model.eval()

        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except Exception:
            pass

        scfg = self.model.encoder.streaming_cfg

        self.shift_frames = (
            scfg.shift_size[1]
            if isinstance(scfg.shift_size, (list, tuple))
            else scfg.shift_size
        )

        pre_cache = scfg.pre_encode_cache_size
        self.pre_cache_frames = (
            pre_cache[1]
            if isinstance(pre_cache, (list, tuple))
            else pre_cache
        )

        self.drop_extra = int(getattr(scfg, "drop_extra_pre_encoded", 0))

        self._frame_stride_sec = float(
            self.model.cfg.preprocessor.get("window_stride", 0.01)
        )
        self.hop_samples = int(self._frame_stride_sec * self.sr)

        self._warmup()

        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        try:
            sess = self.new_session(max_buffer_ms=3000)
            silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
            pcm16 = (
                np.clip(silence, -1.0, 1.0) * 32767
            ).astype(np.int16).tobytes()
            sess.accept_pcm16(pcm16)
            _ = sess.finalize(pad_ms=400)
        except Exception:
            pass

    def new_session(self, max_buffer_ms: int):
        return StreamingSession(self, max_buffer_ms=max_buffer_ms)

    @torch.inference_mode()
    def stream_transcribe(
        self,
        audio_f32: np.ndarray,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prev_hyp: Any,
        prev_pred_out: Any,
        emitted_frames: int,
        force_flush: bool = False,
    ):
        assert self.model is not None

        timings = StreamTimings()

        t0 = time.perf_counter()

        audio_tensor = torch.from_numpy(audio_f32).unsqueeze(0)
        audio_tensor = self._to_device(audio_tensor)

        audio_len = torch.tensor(
            [len(audio_f32)],
            device=audio_tensor.device,
        )

        mel, mel_len = self.model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )

        timings.preproc_sec += time.perf_counter() - t0

        available = int(mel.shape[-1]) - 1

        if available <= 0:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        enough = (available - emitted_frames) >= self.shift_frames

        if not enough and not force_flush:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        if emitted_frames == 0:
            chunk_start = 0
            chunk_end = min(self.shift_frames, available)
            drop_extra = 0
        else:
            chunk_start = max(0, emitted_frames - self.pre_cache_frames)
            chunk_end = min(emitted_frames + self.shift_frames, available)
            drop_extra = self.drop_extra

        chunk_mel = mel[:, :, chunk_start:chunk_end]

        chunk_len = torch.tensor(
            [chunk_mel.shape[-1]],
            device=chunk_mel.device,
        )

        cache = self._move_cache_to_device(cache)

        t1 = time.perf_counter()

        (
            prev_pred_out,
            texts,
            cache0,
            cache1,
            cache2,
            prev_hyp,
        ) = self.model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=False,
            previous_hypotheses=prev_hyp,
            previous_pred_out=prev_pred_out,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )

        timings.infer_sec += time.perf_counter() - t1

        new_cache = (cache0, cache1, cache2)

        if emitted_frames < available:
            emitted_frames = min(
                emitted_frames + self.shift_frames,
                available,
            )

        text = safe_text(texts).strip() if texts is not None else ""

        return text, new_cache, prev_hyp, prev_pred_out, emitted_frames, timings


class StreamingSession:

    def __init__(self, engine: NemotronStreamingASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))

        self.audio = np.array([], dtype=np.float32)
        self.cache = None
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        self.current_text = ""
        self.last_final_text = ""

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._trimmed_since_last_step = False

        self.reset_stream_state()

    def reset_stream_state(self):
        cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)

        self.cache = self.engine._move_cache_to_device(
            (cache[0], cache[1], cache[2])
        )

        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        self.current_text = ""
        self.audio = np.array([], dtype=np.float32)

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._trimmed_since_last_step = False

    def accept_pcm16(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        self.audio = np.concatenate([self.audio, x])

        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]
            self._trimmed_since_last_step = True

    def backlog_ms(self) -> int:
        return int(1000 * (len(self.audio) / self.engine.sr))

    def _is_new_text(self, new_text: Optional[str]) -> bool:
        if new_text is None:
            return False

        new = new_text.strip()
        old = self.current_text.strip()

        if new == "":
            return False

        if new == old:
            return False

        if new.startswith(old):
            return True

        if old.startswith(new):
            return False

        return True

    def step_if_ready(self) -> Optional[str]:
        if self._trimmed_since_last_step and self.emitted_frames > 0:
            cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)

            self.cache = self.engine._move_cache_to_device(
                (cache[0], cache[1], cache[2])
            )

            self.prev_hyp = None
            self.prev_pred = None
            self.emitted_frames = 0
            self._trimmed_since_last_step = False

        (
            text,
            self.cache,
            self.prev_hyp,
            self.prev_pred,
            self.emitted_frames,
            t,
        ) = self.engine.stream_transcribe(
            audio_f32=self.audio,
            cache=self.cache,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred,
            emitted_frames=self.emitted_frames,
            force_flush=False,
        )

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec

        if not self._is_new_text(text):
            return None

        self.current_text = text.strip()
        self.chunks += 1

        return self.current_text

    def finalize(self, pad_ms: int) -> str:
        pad = np.zeros(
            int(self.engine.sr * (pad_ms / 1000.0)),
            dtype=np.float32,
        )

        self.audio = np.concatenate([self.audio, pad])

        t0 = time.perf_counter()

        (
            text,
            self.cache,
            self.prev_hyp,
            self.prev_pred,
            self.emitted_frames,
            t,
        ) = self.engine.stream_transcribe(
            audio_f32=self.audio,
            cache=self.cache,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred,
            emitted_frames=self.emitted_frames,
            force_flush=True,
        )

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec
        self.utt_flush += time.perf_counter() - t0

        if text:
            self.current_text = text.strip()

        final = self.current_text.strip()

        self.last_final_text = (
            (self.last_final_text + " " + final).strip()
            if final
            else self.last_final_text
        )

        self.reset_stream_state()

        return final
