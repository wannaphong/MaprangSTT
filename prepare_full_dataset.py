import os
import re
import json
import sqlite3
import pandas as pd
from pythainlp.util import normalize

BASE_DIR = "/workspace/whistle-thai"
DB_PATH = os.path.join(BASE_DIR, "thai_commonvoice_dataset", "make_datasets", "db-lastest.db")
LIST_FILE_PATH = os.path.join(BASE_DIR, "thai_commonvoice_dataset", "make_datasets", "list_file.tsv")
CV_DIR = os.path.join(BASE_DIR, "cv-corpus-27.0-2026-09-11", "th")
OUT_DIR = os.path.join(BASE_DIR, "dataset_whistle_full")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load SQLite fixes from db-lastest.db
print("Loading SQLite sentence fixes...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.execute("SELECT sent, sent_fix FROM cv")

regex = re.compile(r"[^\w\u0E00-\u0E7Fa-zA-Z]")

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
print(f"Loaded {len(file_replace)} file-specific overrides.")

def preprocess_text(sent, filename):
    if filename in file_replace:
        sent = file_replace[filename]
    sent = sent.replace("ํา", "ำ")
    cleaned = call(sent)
    if cleaned in sent_fix:
        cleaned = sent_fix[cleaned]
    return cleaned

from pythainlp.tokenize import word_tokenize
clean_whitespace = re.compile(r"\s+")

thai_only_pattern = re.compile(r"^[ \u0E00-\u0E7F]+$")
has_digit = re.compile(r"[0-9\u0E50-\u0E59]")

# 3. Process dev and test splits first
dev_df = pd.read_csv(os.path.join(CV_DIR, "dev.tsv"), sep="\t", low_memory=False)
test_df = pd.read_csv(os.path.join(CV_DIR, "test.tsv"), sep="\t", low_memory=False)
val_df = pd.read_csv(os.path.join(CV_DIR, "validated.tsv"), sep="\t", low_memory=False)

dev_paths = set(dev_df["path"])
test_paths = set(test_df["path"])

# Process function
def process_records(df):
    records = []
    for _, row in df.iterrows():
        p = row["path"]
        raw = str(row["sentence"]) if pd.notna(row["sentence"]) else ""
        cleaned = preprocess_text(raw, p)
        if not cleaned:
            continue
        if has_digit.search(cleaned):
            continue
        if not thai_only_pattern.match(cleaned):
            continue
        tokenized = clean_whitespace.sub(" ", " ".join(word_tokenize(cleaned, engine="newmm"))).strip()
        records.append({
            "audio": f"clips/{p}",
            "text": tokenized,
            "language": "th"
        })
    return records

dev_records = process_records(dev_df)
test_records = process_records(test_df)

# Train split = all validated minus dev and test
train_source_df = val_df[~val_df["path"].isin(dev_paths | test_paths)].copy()
train_records = process_records(train_source_df)

print(f"Dev records: {len(dev_records):,}")
print(f"Test records: {len(test_records):,}")
print(f"Train records (from full validated): {len(train_records):,}")

# Write jsonl files
with open(os.path.join(OUT_DIR, "validation.jsonl"), "w", encoding="utf-8") as f:
    for r in dev_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(os.path.join(OUT_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
    for r in test_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Copy tokens.txt from dataset_whistle
import shutil
shutil.copyfile("/workspace/whistle-thai/dataset_whistle/tokens.txt", os.path.join(OUT_DIR, "tokens.txt"))

# Create clips needed list
all_needed = {r["audio"].replace("clips/", "") for r in dev_records + test_records + train_records}
print(f"Total unique clips needed across full dataset: {len(all_needed):,}")
with open(os.path.join(OUT_DIR, "needed_clips.txt"), "w", encoding="utf-8") as f:
    for p in sorted(all_needed):
        f.write(f"cv-corpus-27.0-2026-09-11/th/clips/{p}\n")

print("Finished preparing dataset_whistle_full!")
