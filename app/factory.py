from app.config import Config
from app.asr_engines.nemotron_asr import NemotronStreamingASR


def build_engine(cfg: Config):
    """
    Instantiate and return the correct ASR engine from config.
    Does NOT call engine.load() — that happens at startup.
    """
    if cfg.asr_backend == "nemotron":
        return NemotronStreamingASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )

    raise ValueError(f"Unsupported ASR_BACKEND='{cfg.asr_backend}'")
