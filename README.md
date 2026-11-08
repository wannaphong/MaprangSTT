# MaprangSTT ([`pythainlp/MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT)) — On-Device Thai & Dialect ASR

**[`MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT)** (`whistle-thai-combined`) is a fine-tuned [Cactus Whistle](https://cactuscompute.com/blog/finetuning-whistle) automatic speech recognition (ASR) model for **Standard Thai and Regional Thai Dialects** (Central, Isan / Northeastern, Northern / Khummuang, Korat, Southern / Pattani, and Elderly speech).

Trained on **1,022.78 hours (594,231 clips)** of curated Thai speech using quantization-aware LoRA (`CQ W4 STE + A8`) and an expanded Thai SentencePiece vocabulary, **MaprangSTT** compiles into a standalone **32.49 MB** `.cact` archive ready for fast on-device CPU/GPU/mobile inference.

- **Hugging Face Repository**: [https://huggingface.co/pythainlp/MaprangSTT](https://huggingface.co/pythainlp/MaprangSTT)

---

## Key Highlights

- **6.74% CER / 18.57% WER** on Mozilla Common Voice v27 (`th`) test set (down from `14.01%` CER / `31.65%` WER on the Common Voice-only model).
- **13.58% CER / 31.05% WER** on Google FLEURS (`th_th`) test set (`1,021` clips) — nearly a **3× reduction in Character Error Rate** compared to training on Common Voice alone (`37.57%` CER).
- **Spoken Thai Number Transcription**: Trained with automatic digit/number-to-Thai-words normalization (`1967` $\rightarrow$ `หนึ่ง พัน เก้า ร้อย หกสิบ เจ็ด`).
- **Compact & Edge-Ready**: Standalone **32.49 MB** `W4A8` quantized `.cact` archive (`maprang.cact` / `MaprangSTT.cact`) plus **29.0 MB** LoRA + embedding adapter (`adapter.safetensors`).

---

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model Name** | **[`pythainlp/MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT)** (`whistle-thai-combined`) |
| **Base Architecture** | Cactus Whistle (8 encoder/decoder layers) |
| **Fine-Tuning Method** | Quantization-Aware LoRA (`rank=16`, `alpha=32`, 15 weight groups) + Trainable New Embedding Rows |
| **Precision / Quantization** | `W4A8` (4-bit weights with straight-through estimator, 8-bit activations & KV cache) |
| **Supported Languages** | `th` (Thai), plus base `en`, `de`, `fr`, `es`, `it`, `nl`, `pl` |
| **Vocabulary Expansion** | `+2,067` Thai tokens (`+10,958` SentencePiece pieces + `<\|th\|>` language tag) |
| **Standalone `.cact` Size** | **32.49 MB** (`maprang.cact`, 681 tensors) |
| **LoRA Adapter Size** | **29.0 MB** (`adapter.safetensors`) |

---

## Training Datasets (1,022.78 Hours / 594,231 Clips)

The training corpus combines five Thai speech datasets spanning scripted read speech, conversational regional dialects, and elderly speech:

| Dataset | Split Used | Text Column | Valid Clips | Duration (Hours) |
| :--- | :--- | :--- | ---: | ---: |
| **Mozilla Common Voice v27.0 (`th`)** | `validated` − (`dev` ∪ `test`) | `sentence` | 126,690 | 149.41 h |
| **[`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (`th_th`)** | `train` | `transcription` | 2,174 | 6.84 h |
| **[`n-order/Thai-dialect-corpus`](https://huggingface.co/datasets/n-order/Thai-dialect-corpus)** | `central_train`, `khummuang_train`, `korat_train`, `pattani_train` | `sentence` | 437,563 | 816.74 h |
| **[`huangjianbyte2024/thai_Elderly_Speech_dataset`](https://huggingface.co/datasets/huangjianbyte2024/thai_Elderly_Speech_dataset)** | `train` | `transcription` | 19,198 | 26.56 h |
| **[`typhoon-ai/thai-dialect-isan-dataset`](https://huggingface.co/datasets/typhoon-ai/thai-dialect-isan-dataset)** | `train` | `isan_spelling` | 8,606 | 23.22 h |
| **Total Training Set (`train.jsonl`)** | **Combined** | — | **594,231** | **1,022.78 h** |
| **Validation Set (`validation.jsonl`)** | Common Voice `dev` | `sentence` | 11,063 | 14.44 h |
| **Test Set (`test.jsonl`)** | Common Voice `test` | `sentence` | 11,064 | 15.06 h |

---

## Evaluation Results

All benchmarks below were evaluated on the exported `W4A8` `.cact` archive using `cactus.transcribe(..., language="th")`.
- **Character Error Rate (CER)**: Computed over Thai characters (whitespace removed).
- **Word Error Rate (WER)**: Computed over space-delimited tokens segmented via `pythainlp` (`newmm`).

### Google FLEURS (`th_th`) Test Set — Comparison with Popular Thai ASR Models (`1,021` Clips)

Evaluated on all **1,021 clips** of the [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (`th_th`) test set alongside popular open-source Thai ASR models via [`pythaiasr`](https://github.com/PyThaiNLP/pythaiasr). All outputs were normalized with the same PyThaiNLP text normalizer, number-to-Thai-words converter, and `newmm` word tokenizer:

| Model | Architecture / Precision | Model Size | Google FLEURS (`th_th`) CER | Google FLEURS (`th_th`) WER | Size Ratio vs MaprangSTT |
| :--- | :--- | ---: | :---: | :---: | :---: |
| **[`pythainlp/MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT) (Ours)** | **Cactus Whistle (`W4A8` `.cact`)** | **32.5 MB** | **`13.58%`** | **`31.05%`** | **1.0×** |
| [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | Whisper Small (`244M`, FP32) | 967.0 MB | `24.21%` | `75.74%` | 29.8× larger |
| [`airesearch/wav2vec2-large-xlsr-53-th`](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th) | Wav2Vec2-Large-XLSR-53 (`315M`, FP32) | 1,260.0 MB | `9.56%` | `35.98%` | 38.8× larger |
| `typhoon_nemotron_asr` | Nemotron-3.5-ASR (`0.6B`, INT4 ONNX) | ~450.0 MB | `12.28%` | `27.52%` | 13.8× larger |
| [`biodatlab/whisper-small-th-combined`](https://huggingface.co/biodatlab/whisper-small-th-combined) | Whisper Small Thai (`244M`, FP16) | 563.0 MB | `9.32%` | `24.30%` | 17.3× larger |
| `typhoon_asr` | FastConformer RNN-T (ONNX) | 435.0 MB | `9.02%` | `20.11%` | 13.4× larger |

> **Key Takeaways**:
> - At only **32.5 MB**, **[`MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT)** outperforms [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (`967 MB`) by **10.63% absolute CER** (`13.58%` vs `24.21%`) and **44.69% absolute WER** (`31.05%` vs `75.74%`).
> - It achieves a **4.93% lower WER** (`31.05%` vs `35.98%`) than [`airesearch/wav2vec2-large-xlsr-53-th`](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th) while being **38.8× smaller** (`32.5 MB` vs `1.26 GB`).
> - It comes within **1.30% CER** of the `0.6B`-parameter `typhoon_nemotron_asr` (`13.58%` vs `12.28%`) at **1/14th the model size**.

### Benchmark Comparison Across Training Runs

| Model Variant | Training Data | Hours | Vocab (`tokens.txt`) | Final Val Loss | Common Voice Test CER | Common Voice Test WER | Google FLEURS (`th_th`) Test CER | Google FLEURS (`th_th`) Test WER |
| :--- | :--- | ---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `whistle-thai-v1` | Common Voice `train.tsv` | 37.5 h | 67 Thai chars | `1.8832` | `95.94%` | — | — | — |
| `whistle-thai-v2` | Common Voice `train.tsv` | 37.5 h | 67 chars + 2,000 TNC words | `1.1960` | `22.08%` | — | — | — |
| `whistle-thai-full` | Common Voice `validated.tsv` | 149.4 h | 67 chars + 2,000 TNC words | `0.7421` | `14.01%` | `31.65%` | `37.57%` | `60.19%` |
| **[`MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT) (`whistle-thai-combined`)** | **5-Dataset Combined Corpus** | **1,022.8 h** | **67 chars + 2,000 TNC words** | **`0.3867`** | **`6.74%`** | **`18.57%`** | **`13.58%`** | **`31.05%`** |

### Epoch-by-Epoch Training Progression (`MaprangSTT`)

- **Epochs**: 3 (`111,420` total steps, `37,140` steps/epoch, batch size `16`)
- **Optimizer**: AdamW (`peak_lr=1e-3`, warmup `5,571` steps, cosine decay to `0.0`, global grad norm clip `1.0`)

| Epoch | Step | Train Loss | Validation Loss (`Common Voice dev`) |
| :---: | :---: | :---: | :---: |
| **Epoch 1 / 3** | `37,140 / 111,420` | `1.3467` | `0.5841` |
| **Epoch 2 / 3** | `74,280 / 111,420` | `0.2417` | `0.4239` |
| **Epoch 3 / 3** | `111,420 / 111,420` | **`0.1803`** | **`0.3867`** |

### Sample Predictions (`MaprangSTT`)

**Google FLEURS (`th_th`) Test Set**:
```text
Ref: ตาม ข่าว ที่ รั่วไหล ออกมา เอกสาร ฉบับ นี้ จะ อ้างอิง ถึง ข้อพิพาท เรื่อง พรมแดน ซึ่ง ปาเลสไตน์ ต้องการ โดย อ้างอิง จาก พรมแดน ก่อน สงคราม ตะวันออกกลาง ใน ปี หนึ่ง พัน เก้า ร้อย หกสิบ เจ็ด
Hyp: ตาม ข่าว ที่ ดูแล ออกมา เอกสาร ฉบับ นี้ จะ อ้างอิง ถึง ข้อพิพาท เรื่อง พรมแดน ซึ่ง ตาเลสไตน์ ต้องการ โดย อ้างอิง จะ พรมแดน ก่อน สงคราม ตะวันออกกลาง ใน ปี หนึ่ง พัน เก้า ร้อย หกสิบ เจ็ด
```

**Mozilla Common Voice v27 (`th`) Test Set**:
```text
Ref: ฉัน เช่นกัน ดังนั้น ฉัน พยายาม ที่จะ ได้รับ มากมาย ทุกที่ ที่ ฉัน ไป
Hyp: ฉัน เช่นกัน ดังนั้น ฉัน พยายาม ที่จะ ได้รับ มากมาย ทุกที่ ที่ ฉัน ไป
```

---

## Preprocessing & Tokenizer Expansion

1. **Thai Text Normalization & Cleaning**:
   - Normalized Thai character order and vowel sequences via `pythainlp.util.normalize` (including `ํา` $\rightarrow$ `ำ`).
   - Applied community-verified transcript fixes from [`thai_commonvoice_dataset`](https://github.com/wannaphong/thai_commonvoice_dataset) (`db-lastest.db` and `list_file.tsv`).
2. **Number-to-Thai-Words Conversion**:
   - Converted both Thai digits (`๐-๙`) and Arabic digits (`0-9`), comma-formatted numbers, and decimals into spoken Thai words using `pythainlp.util.num_to_thaiword` and `pythainlp.util.digit_to_text`.
   - Filtered out any remaining clips containing non-Thai characters or exceeding 30 seconds.
3. **Word Tokenization**:
   - Segmented all reference transcripts into space-delimited words using `pythainlp.tokenize.word_tokenize(..., engine="newmm")`.
4. **Thai National Corpus (TNC) Vocabulary Expansion (`tokens.txt`)**:
   - Whistle's base SentencePiece model only covers Latin/European scripts (`en, de, fr, es, it, nl, pl`).
   - We expanded the vocabulary with **2,067 tokens** consisting of all **67 base Thai characters** (`ก-ฮ`, vowels, and tone marks) plus the **top 2,000 most frequent Thai words** from the Thai National Corpus (`pythainlp.corpus.tnc`), adding `10,958` prefix-closed SentencePiece pieces.

---

## Usage

### 1. Install Cactus and Download `MaprangSTT`

```bash
pip install cactus-compute huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import cactus

weights_path = hf_hub_download(
    repo_id="pythainlp/MaprangSTT",
    filename="maprang.cact",
)

result = cactus.transcribe(
    "sample.wav",
    weights=weights_path,
    language="th",
)
print(result["text"])
```

### 2. Transcribe `.mp3` or Non-16kHz Audio Files

`cactus.transcribe()` expects a 16 kHz mono PCM `.wav` file. If your input audio is `.mp3` or uses a different sample rate, convert it to 16 kHz mono `.wav` first:

```python
import tempfile
import wave
import numpy as np
import soundfile
import soxr
import cactus
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="pythainlp/MaprangSTT",
    filename="maprang.cact",
)

def transcribe_any_audio(audio_path: str, weights: str = weights_path) -> str:
    data, sr = soundfile.read(audio_path, dtype="float32", always_2d=True)
    samples = data.mean(1)
    if sr != 16000:
        samples = soxr.resample(samples, sr, 16000, quality="HQ")
    pcm16 = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        with wave.open(tmp_wav.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        res = cactus.transcribe(tmp_wav.name, weights=weights, language="th")
        return res["text"].strip()
```

### 3. Rebuild `.cact` Archive from the LoRA Adapter

If you want to build a platform-specific runnable bundle or slice decoder layers (`--layers 4` or `--layers 6`) from `adapter.safetensors`:

```bash
pip install "cactus-compute[train]"

# Build full 8-layer standalone .cact archive
cactus whistle build \
    --lora adapter.safetensors \
    --out maprang.cact

# Or build a smaller/faster 4-layer decoder archive
cactus whistle build \
    --lora adapter.safetensors \
    --layers 4 \
    --out MaprangSTT-4L.cact
```

---

## Repository Files ([`pythainlp/MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT))

### Hugging Face Model Artifacts ([`pythainlp/MaprangSTT`](https://huggingface.co/pythainlp/MaprangSTT))
- `maprang.cact` (`32.49 MB`): Standalone quantized (`W4A8`) Whistle model archive ready for `cactus.transcribe()`.
- `adapter.safetensors` (`29.0 MB`): Trained LoRA weights (`rank=16`, `alpha=32`) and expanded Thai vocabulary embedding rows.
- `tokens.txt` (`2,067` lines): Base Thai characters and top 2,000 TNC words used to expand the SentencePiece tokenizer.

### Training, Preprocessing & Evaluation Scripts (This GitHub Repo)

```bash
# Clone with submodule for Thai Common Voice sentence fixes
git clone --recurse-submodules https://github.com/wannaphong/whistle-thai.git
cd whistle-thai
pip install -r requirements.txt
```

- **Vocabulary Construction**:
  - `build_tnc_vocab.py`: Builds `tokens.txt` (`67` base Thai characters + top `2,000` Thai National Corpus words via `pythainlp.corpus.tnc`).
  - `build_vocab.py`: Initial experiment building vocabulary from Common Voice training word frequencies.
- **Final Combined Pipeline (`MaprangSTT` / `whistle-thai-combined`)**:
  - `download.sh`: Template script to download Mozilla Common Voice v27 (`th`).
  - `prepare_combined_dataset.py`: Preprocesses, cleans, converts numbers to Thai words, and word-tokenizes all 5 training datasets (`594,231` clips / `1,022.78` hours) into `dataset_whistle_combined/`.
  - `add_isan_dataset.py`: Downloads, resamples (`48 kHz` $\rightarrow$ `16 kHz` mono PCM), and processes [`typhoon-ai/thai-dialect-isan-dataset`](https://huggingface.co/datasets/typhoon-ai/thai-dialect-isan-dataset) (`isan_spelling`).
  - `run_finetune_combined.sh`: Runs 3-epoch `cactus whistle finetune` on the combined corpus and compiles `maprang.cact`.
- **Earlier Common Voice Experiments (`v1`, `v2`, `full`)**:
  - `preprocess_dataset.py`, `run_finetune.sh`, `run_finetune_v2.sh`: Initial `37.5 h` Common Voice `train.tsv` experiments (`v1` character-only vs `v2` TNC vocabulary).
  - `prepare_full_dataset.py`, `tokenize_full_dataset.py`, `run_finetune_full.sh`: Full `149.4 h` Common Voice `validated.tsv` training pipeline (`whistle-thai-full`).
- **Evaluation & Monitoring**:
  - `evaluate_model.py`: Evaluates a `.cact` model on Common Voice `test.jsonl` (CER & WER).
  - `evaluate_fleurs.py`: Evaluates a `.cact` model on Google FLEURS (`th_th`) test set (`1,021` clips).
  - `compare_fleurs.py` & `fleurs_comparison.json`: Benchmarks `MaprangSTT` against popular Thai ASR models on Google FLEURS via `pythaiasr`.
  - `eval.md`: Full evaluation logs and benchmark outputs.
  - `show_train_status.sh`, `watch_progress.py`, `run_with_logging.py`: Training progress and GPU monitoring utilities.

---

## Citation & Acknowledgements

- **Model**: [PyThaiNLP — MaprangSTT (`https://huggingface.co/pythainlp/MaprangSTT`)](https://huggingface.co/pythainlp/MaprangSTT)
- **Base Model & Framework**: [Cactus Compute — Finetuning Whistle](https://cactuscompute.com/blog/finetuning-whistle)
- **NLP Preprocessing & Evaluation**: [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp), [PyThaiASR](https://github.com/PyThaiNLP/pythaiasr), & [thai_commonvoice_dataset](https://github.com/wannaphong/thai_commonvoice_dataset)
- **Speech Datasets**:
  - [Mozilla Common Voice](https://commonvoice.mozilla.org/)
  - [Google FLEURS (`google/fleurs`)](https://huggingface.co/datasets/google/fleurs)
  - [`n-order/Thai-dialect-corpus`](https://huggingface.co/datasets/n-order/Thai-dialect-corpus)
  - [`huangjianbyte2024/thai_Elderly_Speech_dataset`](https://huggingface.co/datasets/huangjianbyte2024/thai_Elderly_Speech_dataset)
  - [`typhoon-ai/thai-dialect-isan-dataset`](https://huggingface.co/datasets/typhoon-ai/thai-dialect-isan-dataset)
