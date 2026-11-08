import os
import re
import glob
import json
import tarfile
import sqlite3
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import hf_hub_download
from pythainlp.util import normalize, num_to_thaiword, thai_digit_to_arabic_digit, digit_to_text
from pythainlp.tokenize import word_tokenize

BASE_DIR = "/workspace/whistle-thai"
DB_PATH = os.path.join(BASE_DIR, "thai_commonvoice_dataset", "make_datasets", "db-lastest.db")
LIST_FILE_PATH = os.path.join(BASE_DIR, "thai_commonvoice_dataset", "make_datasets", "list_file.tsv")
CV_DIR = os.path.join(BASE_DIR, "cv-corpus-27.0-2026-09-11", "th")
OUT_DIR = os.path.join(BASE_DIR, "dataset_whistle_combined")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load SQLite fixes from db-lastest.db
print("Loading SQLite sentence fixes...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.execute("SELECT sent, sent_fix FROM cv")

regex = re.compile(r"[^\w\u0E00-\u0E7Fa-zA-Z]")
clean_whitespace = re.compile(r"\s+")
thai_only_pattern = re.compile(r"^[ \u0E00-\u0E7F]+$")
has_digit = re.compile(r"[0-9\u0E50-\u0E59]")

def call(s):
    s = normalize(s.strip())
    return re.sub(r"\s\s+", " ", regex.sub(" ", s).replace("  ", " ").replace("“", "").replace(" ”", "").replace("”", "")).strip()

sent_fix = {}
for row in cursor:
    if row[1] is not None:
        sent_fix[call(row[0])] = call(row[1])
    else:
        sent_fix[call(row[0])] = call(row[0])
conn.close()
print(f"Loaded {len(sent_fix)} rules from DB.")

# 2. Load list_file.tsv overrides
replace_sent_df = pd.read_csv(LIST_FILE_PATH, sep="\t")
file_replace = dict(zip(replace_sent_df["file"], replace_sent_df["sentence"]))

def convert_numbers_to_thai(text):
    text = thai_digit_to_arabic_digit(text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    def replace_decimal(m):
        int_part, dec_part = m.group(1), m.group(2)
        try:
            int_word = num_to_thaiword(int(int_part)) if int_part else ""
        except Exception:
            int_word = digit_to_text(int_part)
        dec_word = digit_to_text(dec_part)
        return f" {int_word}จุด{dec_word} "

    text = re.sub(r"(\d+)\.(\d+)", replace_decimal, text)

    def replace_int(m):
        s = m.group(0)
        if (len(s) > 1 and s.startswith("0")) or len(s) > 12:
            return f" {digit_to_text(s)} "
        try:
            return f" {num_to_thaiword(int(s))} "
        except Exception:
            return f" {digit_to_text(s)} "

    text = re.sub(r"\d+", replace_int, text)
    return text

def preprocess_and_tokenize(raw_sent, filename=None):
    if not raw_sent or not isinstance(raw_sent, str):
        return None
    sent = raw_sent
    if filename and filename in file_replace:
        sent = file_replace[filename]
    sent = sent.replace("ํา", "ำ")
    # Convert digits/numbers to Thai words before stripping symbols
    sent = convert_numbers_to_thai(sent)
    cleaned = call(sent)
    if cleaned in sent_fix:
        cleaned = sent_fix[cleaned]
        # In case sent_fix introduced any digits, convert them too
        if has_digit.search(cleaned):
            cleaned = call(convert_numbers_to_thai(cleaned))
    if not cleaned:
        return None
    if has_digit.search(cleaned):
        return None
    if not thai_only_pattern.match(cleaned):
        return None
    tokenized = clean_whitespace.sub(" ", " ".join(word_tokenize(cleaned, engine="newmm"))).strip()
    return tokenized if tokenized else None

def process_parquet_shard(args):
    pq_path, out_audio_dir, text_col, prefix = args
    os.makedirs(out_audio_dir, exist_ok=True)
    df = pd.read_parquet(pq_path)
    records = []
    shard_name = os.path.splitext(os.path.basename(pq_path))[0]
    for idx, row in df.iterrows():
        raw_text = row.get(text_col)
        cleaned = preprocess_and_tokenize(raw_text)
        if not cleaned:
            continue
        audio_info = row["audio"]
        audio_bytes = audio_info["bytes"]
        orig_path = audio_info.get("path") or f"{prefix}_{shard_name}_{idx}.wav"
        base_fname = f"{prefix}_{shard_name}_{idx}_{os.path.basename(orig_path)}"
        full_audio_path = os.path.join(out_audio_dir, base_fname)
        if not os.path.exists(full_audio_path):
            with open(full_audio_path, "wb") as f:
                f.write(audio_bytes)
        rel_path = os.path.relpath(full_audio_path, OUT_DIR)
        records.append({
            "audio": rel_path,
            "text": cleaned,
            "language": "th"
        })
    return shard_name, records

def main():
    # Setup symlinks/directories in OUT_DIR
    cv_clips_link = os.path.join(OUT_DIR, "clips")
    if not os.path.exists(cv_clips_link):
        os.symlink(os.path.join(CV_DIR, "clips"), cv_clips_link)

    # --- 1. Common Voice v27 ---
    print("\n--- Step 1: Processing Common Voice v27 ---")
    dev_df = pd.read_csv(os.path.join(CV_DIR, "dev.tsv"), sep="\t", low_memory=False)
    test_df = pd.read_csv(os.path.join(CV_DIR, "test.tsv"), sep="\t", low_memory=False)
    val_df = pd.read_csv(os.path.join(CV_DIR, "validated.tsv"), sep="\t", low_memory=False)

    dev_paths = set(dev_df["path"])
    test_paths = set(test_df["path"])
    cv_train_df = val_df[~val_df["path"].isin(dev_paths | test_paths)].copy()

    def process_cv_df(df):
        recs = []
        for _, row in df.iterrows():
            p = row["path"]
            raw = str(row["sentence"]) if pd.notna(row["sentence"]) else ""
            cleaned = preprocess_and_tokenize(raw, p)
            if not cleaned:
                continue
            recs.append({
                "audio": f"clips/{p}",
                "text": cleaned,
                "language": "th"
            })
        return recs

    dev_records = process_cv_df(dev_df)
    test_records = process_cv_df(test_df)
    cv_train_records = process_cv_df(cv_train_df)
    print(f"Common Voice -> Train: {len(cv_train_records):,}, Val: {len(dev_records):,}, Test: {len(test_records):,}")

    # Check if any newly rescued CV clips (that previously had digits) need extracting from tar
    existing_cv = set(os.listdir(os.path.join(CV_DIR, "clips")))
    needed_cv = {r["audio"].replace("clips/", "") for r in dev_records + test_records + cv_train_records}
    missing_cv = needed_cv - existing_cv
    if missing_cv:
        print(f"Extracting {len(missing_cv)} additional Common Voice clips that had numbers converted...")
        missing_txt = os.path.join(OUT_DIR, "missing_cv.txt")
        with open(missing_txt, "w") as f:
            for m in sorted(missing_cv):
                f.write(f"cv-corpus-27.0-2026-09-11/th/clips/{m}\n")
        os.system(f"tar -xzf /workspace/whistle-thai/common-voice-scripted-speech-27-0-thai-3aa1b7de.tar.gz -T {missing_txt} -C /workspace/whistle-thai/")

    # --- 2. Google FLEURS (th_th) train ---
    print("\n--- Step 2: Processing Google FLEURS (th_th) train set ---")
    fleurs_tsv = hf_hub_download(repo_id="google/fleurs", filename="data/th_th/train.tsv", repo_type="dataset")
    fleurs_tar = hf_hub_download(repo_id="google/fleurs", filename="data/th_th/audio/train.tar.gz", repo_type="dataset")
    fleurs_extract_dir = os.path.join(OUT_DIR, "fleurs_audio")
    fleurs_train_audio_dir = os.path.join(fleurs_extract_dir, "train")
    if not os.path.isdir(fleurs_train_audio_dir) or len(os.listdir(fleurs_train_audio_dir)) < 2000:
        print("Extracting FLEURS train.tar.gz...")
        os.makedirs(fleurs_extract_dir, exist_ok=True)
        with tarfile.open(fleurs_tar, "r:gz") as tf:
            tf.extractall(fleurs_extract_dir)

    fleurs_records = []
    with open(fleurs_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            wav_name = parts[1]
            # Use raw or normalized transcript (convert_numbers_to_thai handles digits in either)
            transcript = parts[3].strip() or parts[2].strip()
            cleaned = preprocess_and_tokenize(transcript)
            if not cleaned:
                continue
            wav_full = os.path.join(fleurs_train_audio_dir, wav_name)
            if os.path.isfile(wav_full):
                fleurs_records.append({
                    "audio": os.path.relpath(wav_full, OUT_DIR),
                    "text": cleaned,
                    "language": "th"
                })
    print(f"Google FLEURS train -> Kept: {len(fleurs_records):,} clips")

    # --- 3. n-order/Thai-dialect-corpus (all *_train subsets, using 'sentence') ---
    print("\n--- Step 3: Processing n-order/Thai-dialect-corpus (*_train) ---")
    dialect_pqs = sorted(glob.glob("/root/.cache/huggingface/hub/datasets--n-order--Thai-dialect-corpus/**/data/*_train-*.parquet", recursive=True))
    dialect_audio_dir = os.path.join(OUT_DIR, "dialect_audio")
    dialect_tasks = [(pq, dialect_audio_dir, "sentence", "dialect") for pq in dialect_pqs]

    dialect_records = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for shard_name, recs in ex.map(process_parquet_shard, dialect_tasks):
            print(f"  Processed {shard_name}: {len(recs):,} records")
            dialect_records.extend(recs)
    print(f"n-order/Thai-dialect-corpus (*_train) -> Total Kept: {len(dialect_records):,} clips")

    # --- 4. huangjianbyte2024/thai_Elderly_Speech_dataset ---
    print("\n--- Step 4: Processing huangjianbyte2024/thai_Elderly_Speech_dataset ---")
    elderly_pqs = sorted(glob.glob("/root/.cache/huggingface/hub/datasets--huangjianbyte2024--thai_Elderly_Speech_dataset/**/default/train/*.parquet", recursive=True))
    elderly_audio_dir = os.path.join(OUT_DIR, "elderly_audio")
    elderly_tasks = [(pq, elderly_audio_dir, "transcription", "elderly") for pq in elderly_pqs]

    elderly_records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for shard_name, recs in ex.map(process_parquet_shard, elderly_tasks):
            print(f"  Processed elderly {shard_name}: {len(recs):,} records")
            elderly_records.extend(recs)
    print(f"thai_Elderly_Speech_dataset -> Total Kept: {len(elderly_records):,} clips")

    # --- 5. typhoon-ai/thai-dialect-isan-dataset (train, using 'isan_spelling') ---
    print("\n--- Step 5: Processing typhoon-ai/thai-dialect-isan-dataset (isan_spelling) ---")
    from add_isan_dataset import process_isan_shard
    isan_pqs = sorted(glob.glob("/root/.cache/huggingface/hub/datasets--typhoon-ai--thai-dialect-isan-dataset/**/data/train-*.parquet", recursive=True))
    isan_records = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for shard_name, recs, _, _ in ex.map(process_isan_shard, isan_pqs):
            print(f"  Processed isan {shard_name}: {len(recs):,} records")
            isan_records.extend(recs)
    print(f"typhoon-ai/thai-dialect-isan-dataset -> Total Kept: {len(isan_records):,} clips")

    # --- Combine & Write JSONL files ---
    all_train_records = cv_train_records + fleurs_records + dialect_records + elderly_records + isan_records
    print("\n=== Combined Dataset Summary ===")
    print(f"  Common Voice Train:       {len(cv_train_records):,}")
    print(f"  Google FLEURS Train:      {len(fleurs_records):,}")
    print(f"  Thai Dialect Corpus:      {len(dialect_records):,}")
    print(f"  Thai Elderly Speech:      {len(elderly_records):,}")
    print(f"  Typhoon Isan Speech:      {len(isan_records):,}")
    print(f"  ----------------------------------------")
    print(f"  TOTAL TRAIN CLIPS:        {len(all_train_records):,}")
    print(f"  VALIDATION CLIPS (CV):    {len(dev_records):,}")
    print(f"  TEST CLIPS (CV):          {len(test_records):,}")

    with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in all_train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "validation.jsonl"), "w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    shutil.copyfile("/workspace/whistle-thai/dataset_whistle/tokens.txt", os.path.join(OUT_DIR, "tokens.txt"))
    print(f"\nSuccessfully wrote all files to {OUT_DIR}!")

if __name__ == "__main__":
    main()
