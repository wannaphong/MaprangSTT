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
OUT_DIR = os.path.join(BASE_DIR, "dataset_whistle")
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

def check(i):
    i = i.replace("ํา", "ำ")
    t = call(i)
    if t not in sent_fix:
        return None
    m = sent_fix[t]
    if re.search(r"[a-zA-Z]", m):
        return None
    return m

def check_all_ok(sent, filename):
    c1 = check(sent)
    if c1 is None and filename in file_replace:
        c1 = file_replace[filename]
    return c1

# 3. Filtering constraints: No digits (Arabic 0-9 and Thai ๐-๙) and Thai+spaces only
thai_only_pattern = re.compile(r"^[ \u0E00-\u0E7F]+$")
has_digit = re.compile(r"[0-9\u0E50-\u0E59]")

clips_needed = set()
stats = {}
all_thai_chars = set()

for split_name in ["train", "dev", "test"]:
    tsv_file = os.path.join(CV_DIR, f"{split_name}.tsv")
    df = pd.read_csv(tsv_file, sep="\t", low_memory=False)
    
    out_jsonl_name = "validation.jsonl" if split_name == "dev" else f"{split_name}.jsonl"
    out_jsonl_path = os.path.join(OUT_DIR, out_jsonl_name)
    
    total = len(df)
    kept_records = []
    
    for _, row in df.iterrows():
        p = row["path"]
        raw_text = str(row["sentence"]) if pd.notna(row["sentence"]) else ""
        cleaned = check_all_ok(raw_text, p)
        
        # Check conditions
        if not cleaned:
            continue
        if has_digit.search(cleaned):
            continue
        if not thai_only_pattern.match(cleaned):
            continue
            
        audio_rel_path = f"clips/{p}"
        kept_records.append({
            "audio": audio_rel_path,
            "text": cleaned,
            "language": "th"
        })
        clips_needed.add(p)
        for ch in cleaned:
            if ch != " ":
                all_thai_chars.add(ch)
                
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    stats[split_name] = {"total": total, "kept": len(kept_records), "out": out_jsonl_name}
    print(f"[{split_name}] Total: {total} -> Kept: {len(kept_records)} -> Saved to {out_jsonl_name}")

# Also generate tokens.txt
tokens_path = os.path.join(OUT_DIR, "tokens.txt")
sorted_chars = sorted(all_thai_chars)
with open(tokens_path, "w", encoding="utf-8") as f:
    for ch in sorted_chars:
        f.write(ch + "\n")
print(f"Generated {tokens_path} with {len(sorted_chars)} Thai characters.")

# Save needed clip list
needed_list_path = os.path.join(OUT_DIR, "needed_clips.txt")
with open(needed_list_path, "w", encoding="utf-8") as f:
    for p in sorted(clips_needed):
        f.write(f"cv-corpus-27.0-2026-09-11/th/clips/{p}\n")
print(f"Total unique clips needed across splits: {len(clips_needed)}. Written to {needed_list_path}.")
