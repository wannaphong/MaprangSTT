import os
import glob
import json
import wave
import numpy as np
import pyarrow.parquet as pq
import soxr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
from prepare_combined_dataset import preprocess_and_tokenize, OUT_DIR

REPO_ID = "typhoon-ai/thai-dialect-isan-dataset"
ISAN_AUDIO_DIR = os.path.join(OUT_DIR, "isan_audio")
os.makedirs(ISAN_AUDIO_DIR, exist_ok=True)


def download_shard(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")


def write_wav_pcm16(path, samples_f32, sr=16000):
    pcm16 = np.clip(samples_f32 * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def process_isan_shard(pq_path):
    shard_name = os.path.splitext(os.path.basename(pq_path))[0]
    table = pq.read_table(pq_path, columns=["id", "audio", "isan_spelling", "duration"])
    df = table.to_pandas()
    records = []
    total_sec = 0.0
    skipped = 0

    for idx, row in df.iterrows():
        raw_text = row.get("isan_spelling")
        cleaned = preprocess_and_tokenize(raw_text)
        if not cleaned:
            skipped += 1
            continue

        audio_info = row["audio"]
        arr = np.asarray(audio_info["array"], dtype=np.float32)
        sr = int(audio_info.get("sampling_rate") or 48000)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if sr != 16000:
            arr = soxr.resample(arr, sr, 16000, quality="HQ")

        n_samples = len(arr)
        if n_samples == 0 or n_samples > 30 * 16000:
            skipped += 1
            continue

        dur = n_samples / 16000.0
        rec_id = str(row.get("id") or f"{shard_name}_{idx}")
        fname = f"isan_{shard_name}_{idx}_{rec_id}.wav"
        full_wav_path = os.path.join(ISAN_AUDIO_DIR, fname)
        if not os.path.exists(full_wav_path):
            write_wav_pcm16(full_wav_path, arr, 16000)

        rel_path = os.path.relpath(full_wav_path, OUT_DIR)
        records.append({
            "audio": rel_path,
            "text": cleaned,
            "language": "th",
        })
        total_sec += dur

    return shard_name, records, total_sec, skipped


def main():
    api = HfApi()
    info = api.dataset_info(REPO_ID)
    train_files = sorted(
        f.rfilename for f in info.siblings
        if f.rfilename.startswith("data/train-") and f.rfilename.endswith(".parquet")
    )
    print(f"Found {len(train_files)} train parquet shards in {REPO_ID}.")

    print("Downloading shards with 8 threads...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        local_paths = list(ex.map(download_shard, train_files))
    print(f"Downloaded all {len(local_paths)} shards.")

    print("Processing shards (converting isan_spelling + resampling 48kHz -> 16kHz WAV)...")
    all_records = []
    total_seconds = 0.0
    total_skipped = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        for shard_name, recs, sec, sk in ex.map(process_isan_shard, local_paths):
            all_records.extend(recs)
            total_seconds += sec
            total_skipped += sk
            print(f"  {shard_name}: kept {len(recs):,} clips ({sec/60:.1f} min), skipped {sk}", flush=True)

    print(f"\n=== {REPO_ID} (train, isan_spelling) ===")
    print(f"Kept clips:   {len(all_records):,}")
    print(f"Skipped:      {total_skipped:,}")
    print(f"Total hours:  {total_seconds/3600:.2f} hours ({total_seconds/60:.1f} min)")

    # Save standalone JSONL and append to train.jsonl if not already appended
    isan_jsonl = os.path.join(OUT_DIR, "isan_train.jsonl")
    with open(isan_jsonl, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_jsonl = os.path.join(OUT_DIR, "train.jsonl")
    existing_isan = 0
    kept_existing = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if "isan_audio/" in line:
                existing_isan += 1
            else:
                kept_existing.append(line)

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for line in kept_existing:
            f.write(line)
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Updated {train_jsonl}: {len(kept_existing):,} previous + {len(all_records):,} Isan = {len(kept_existing) + len(all_records):,} total train clips!")


if __name__ == "__main__":
    main()
