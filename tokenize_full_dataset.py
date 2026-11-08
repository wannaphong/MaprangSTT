import os
import re
import json
from pythainlp.tokenize import word_tokenize

FULL_DIR = "/workspace/whistle-thai/dataset_whistle_full"
clean_whitespace = re.compile(r"\s+")

for split in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
    path = os.path.join(FULL_DIR, split)
    temp_path = path + ".tmp"
    count = 0
    with open(path, "r", encoding="utf-8") as fin, open(temp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"].strip()
            tokenized = clean_whitespace.sub(" ", " ".join(word_tokenize(text, engine="newmm"))).strip()
            rec["text"] = tokenized
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temp_path, path)
    print(f"Tokenized {split}: {count:,} records")

