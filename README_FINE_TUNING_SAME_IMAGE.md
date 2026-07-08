# Nemotron fine-tuning using your existing Dockerfile/image

This package keeps your original server code and Dockerfile. Your Dockerfile already downloads the base model during `docker build` and saves it at:

```text
/srv/nemotron-3.5-asr-streaming-0.6b.nemo
```

So the same Docker image is used for:

1. baseline evaluation
2. dataset preparation
3. fine-tuning
4. fine-tuned model evaluation
5. running the final server

The only new thing is that the fine-tuned model is saved to a host-mounted folder:

```text
./ft_models/nemotron_inspira_decoder_ft.nemo
```

Inside the container this same folder is visible as:

```text
/srv/models/nemotron_inspira_decoder_ft.nemo
```

---

## Exact folder structure

```text
nemotron_asr_same_image_ft/
├── app/                                  # your original server code
├── client.py                             # your original client
├── Dockerfile                            # your original Dockerfile; downloads base model
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py                # new: wav + csv -> NeMo manifests
│   ├── evaluate_manifest.py              # new: WER/CER evaluation
│   └── finetune_nemotron.py              # new: fine-tuning script
├── raw_wavs/                             # your 7 WAV files
│   ├── withdraw_money.wav
│   ├── Card_lost.wav
│   ├── Card_Delivery_Status.wav
│   ├── COBRA_coverage.wav
│   ├── Profile_Update.wav
│   ├── bank_issue.wav
│   └── Verification_Code_Issue.wav
├── data/
│   └── inspira_transcripts.csv           # use_case,transcript
├── ft_models/                            # output model will appear here
├── 01_enter_training_container.sh
├── 02_prepare_baseline_train_eval_inside_container.sh
├── 03_run_finetuned_server.sh
└── 04_run_base_server.sh
```

---

## Step 1: unzip and enter folder

```bash
unzip nemotron_asr_same_image_ft.zip
cd nemotron_asr_same_image_ft
```

Check:

```bash
ls
```

You should see:

```text
app  client.py  Dockerfile  scripts  raw_wavs  data  ft_models
```

---

## Step 2: build your original Docker image

```bash
docker build -t nemotron-asr .
```

What happens:

- installs Python 3.11
- installs NeMo
- installs Torch CUDA
- downloads `nvidia/nemotron-3.5-asr-streaming-0.6b`
- saves it inside the image at `/srv/nemotron-3.5-asr-streaming-0.6b.nemo`
- copies your `app/` server code

You can verify later from inside container:

```bash
ls -lh /srv/nemotron-3.5-asr-streaming-0.6b.nemo
```

---

## Step 3: enter the same container for fine-tuning

```bash
mkdir -p ft_models

docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  -v $PWD/ft_models:/srv/models \
  nemotron-asr \
  bash
```

What this means:

```text
/workspace                         = your project folder from host
/srv/nemotron-3.5-asr-streaming... = base model inside image
/srv/models                        = ./ft_models on host
```

Inside container:

```bash
cd /workspace
ls
```

---

## Step 4: prepare dataset

Run inside the container:

```bash
python3.11 scripts/prepare_dataset.py \
  --csv data/inspira_transcripts.csv \
  --wav-dir raw_wavs \
  --out-dir data
```

What happens:

```text
raw_wavs/*.wav
  -> converted to 16kHz mono WAV
  -> saved in data/audio_16k/

data/inspira_transcripts.csv
  -> converted to NeMo JSON manifest
  -> saved in data/manifests/
```

After this, check:

```bash
ls data/audio_16k
ls data/manifests
```

Expected:

```text
train_manifest.json
val_manifest.json
test_manifest.json
```

---

## Step 5: check base model WER before fine-tuning

Run inside container:

```bash
mkdir -p logs

python3.11 scripts/evaluate_manifest.py \
  --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --manifest data/manifests/test_manifest.json \
  --language en-US \
  --output logs/base_predictions.jsonl
```

What happens:

- loads original base model
- transcribes your test WAV
- compares prediction with reference transcript
- prints WER/CER
- saves prediction to `logs/base_predictions.jsonl`

Keep this WER as your baseline.

---

## Step 6: fine-tune inside same container

Run inside container:

```bash
python3.11 scripts/finetune_nemotron.py \
  --train-manifest data/manifests/train_manifest.json \
  --val-manifest data/manifests/val_manifest.json \
  --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo \
  --freeze-mode decoder_only \
  --max-epochs 5 \
  --batch-size 1 \
  --lr 5e-6 \
  --language en-US
```

What happens:

- loads `/srv/nemotron-3.5-asr-streaming-0.6b.nemo`
- freezes the main encoder
- trains decoder/joint layers only
- uses low learning rate because dataset is tiny
- saves the fine-tuned model to:

```text
Inside container: /srv/models/nemotron_inspira_decoder_ft.nemo
On host:          ./ft_models/nemotron_inspira_decoder_ft.nemo
```

---

## Step 7: evaluate fine-tuned model

Run inside container:

```bash
python3.11 scripts/evaluate_manifest.py \
  --model /srv/models/nemotron_inspira_decoder_ft.nemo \
  --manifest data/manifests/test_manifest.json \
  --language en-US \
  --output logs/finetuned_predictions.jsonl
```

Compare output with Step 5:

```text
Base WER       = before fine-tuning
Fine-tuned WER = after fine-tuning
```

Also compare files:

```bash
cat logs/base_predictions.jsonl
cat logs/finetuned_predictions.jsonl
```

---

## Step 8: exit training container

```bash
exit
```

Now on host check:

```bash
ls -lh ft_models/
```

Expected:

```text
nemotron_inspira_decoder_ft.nemo
```

---

## Step 9: run server with fine-tuned model

Use same original image, but override `MODEL_NAME`:

```bash
docker run --gpus all -it --rm \
  -p 8003:8003 \
  -v $PWD/ft_models:/srv/models \
  -e MODEL_NAME=/srv/models/nemotron_inspira_decoder_ft.nemo \
  nemotron-asr
```

What happens:

- the image still contains the original base model
- but `MODEL_NAME` points your server to the fine-tuned model
- your existing `app/config.py` reads `MODEL_NAME`
- server loads `/srv/models/nemotron_inspira_decoder_ft.nemo`

---

## Step 10: test fine-tuned server with client

Open a second terminal from same folder:

```bash
python3.11 client.py \
  --file raw_wavs/withdraw_money.wav \
  --url ws://localhost:8003/asr/ws
```

If your client argument is different, use your existing client help:

```bash
python3.11 client.py --help
```

---

## Step 11: run base server again for comparison

To run the original model again:

```bash
docker run --gpus all -it --rm \
  -p 8003:8003 \
  nemotron-asr
```

No `MODEL_NAME` override means it uses:

```text
/srv/nemotron-3.5-asr-streaming-0.6b.nemo
```

---

## Short version

Host:

```bash
docker build -t nemotron-asr .
mkdir -p ft_models

docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  -v $PWD/ft_models:/srv/models \
  nemotron-asr bash
```

Inside container:

```bash
cd /workspace
bash 02_prepare_baseline_train_eval_inside_container.sh
exit
```

Host, run fine-tuned server:

```bash
bash 03_run_finetuned_server.sh
```

---

## Important note

With only 7 WAVs this is a POC fine-tune. It can improve the exact Inspira scripts, but it can also overfit. For stable production improvement, add many more WAV variations with different speakers, IDs, ticket numbers, accents, background noise, and telephony audio.
