# Nemotron Fine-tuning Project - Fixed Version

This version fixes the main causes of poor realtime performance:

- call/use-case level split instead of random chunk split
- train-only augmentation for clean, telephony, volume, speed and light noise
- safer decoder-only fine-tuning with gradient accumulation
- optional last-encoder adaptation with very low LR
- realtime VAD defaults tuned to avoid cutting first/last words
- EOF-fixed client so file realtime mode produces final output
- OpenAI-compatible endpoint remains available at `/v1/audio/transcriptions`
- ASR number normalizer for Inspira IDs/tickets/numeric phrases

## Important

Do not use `scripts/split_aligned_manifest.py` for final training. It was removed because it randomly splits chunks and can leak the same call into train/test.

## Project Structure

```text
nemotron_finetuned/
├── Dockerfile
├── requirements.txt
├── README.md
├── client.py
├── app/
├── scripts/
├── data/
├── raw_wavs/
├── ft_models/
└── results/
```

## Run Training

From host:

```bash
cd /home/CORP/re_nikitav/nemotron_finetuned && chmod +x 01_enter_training_container.sh 03_run_finetuned_server.sh scripts/run_hyparam_tuning.sh
```

```bash
cd /home/CORP/re_nikitav/nemotron_finetuned && ./01_enter_training_container.sh
```

Inside container:

```bash
apt update && apt install -y cuda-nvvm-12-4
```

```bash
cd /workspace && bash scripts/run_hyparam_tuning.sh
```

If v1 is best:

```bash
cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo && ls -lh /srv/models/finetuned_nemotron_final.nemo
```

Exit:

```bash
exit
```

## Run Server

```bash
cd /home/CORP/re_nikitav/nemotron_finetuned && ./03_run_finetuned_server.sh
```

## Test Realtime WebSocket

```bash
cd /home/CORP/re_nikitav/nemotron_finetuned && python3 client.py --file raw_wavs/withdraw_money.wav --language en-US --realtime --url ws://localhost:8003/asr/realtime-custom-vad
```

## Test OpenAI-compatible API

```bash
cd /home/CORP/re_nikitav/nemotron_finetuned && curl -X POST "http://localhost:8003/v1/audio/transcriptions" -F "file=@raw_wavs/withdraw_money.wav" -F "model=nemotron-3.5-asr-streaming-0.6b" -F "language=auto"
```
