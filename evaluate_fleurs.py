import os
import re
import tarfile
import argparse
import cactus
from huggingface_hub import hf_hub_download
from pythainlp.util import normalize, num_to_thaiword, thai_digit_to_arabic_digit, digit_to_text
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm

regex = re.compile(r"[^\w\u0E00-\u0E7Fa-zA-Z]")
clean_whitespace = re.compile(r"\s+")

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

def clean_text(s):
    s = s.replace("ํา", "ำ")
    s = convert_numbers_to_thai(s)
    s = normalize(s.strip())
    s = re.sub(r"\s\s+", " ", regex.sub(" ", s).replace("  ", " ").replace("“", "").replace(" ”", "").replace("”", "")).strip()
    return clean_whitespace.sub(" ", " ".join(word_tokenize(s, engine="newmm"))).strip()

def compute_cer(ref, hyp):
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    row = list(range(len(hyp_chars) + 1))
    for i, r in enumerate(ref_chars, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp_chars, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref_chars)

def compute_wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    row = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp_words, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref_words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whistle Thai model on Google FLEURS (th_th) test set")
    parser.add_argument("--weights", default="/workspace/whistle-thai/output_combined/maprang.cact", help="Path to .cact weights")
    parser.add_argument("--extract-dir", default="/workspace/whistle-thai/fleurs_th", help="Directory to cache extracted FLEURS audio")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test clips (0 for full 1,021 clips)")
    parser.add_argument("--skip-non-thai", action="store_true", help="Skip test sentences containing digits or Latin characters")
    args = parser.parse_args()

    print("Downloading / locating Google FLEURS (th_th) test set...")
    tsv_path = hf_hub_download(repo_id="google/fleurs", filename="data/th_th/test.tsv", repo_type="dataset")
    tar_path = hf_hub_download(repo_id="google/fleurs", filename="data/th_th/audio/test.tar.gz", repo_type="dataset")

    audio_dir = os.path.join(args.extract_dir, "test")
    if not os.path.isdir(audio_dir) or len(os.listdir(audio_dir)) < 1000:
        print(f"Extracting FLEURS test audio to {args.extract_dir}...")
        os.makedirs(args.extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(args.extract_dir)

    records = []
    thai_only_pattern = re.compile(r"^[ \u0E00-\u0E7F]+$")
    has_digit = re.compile(r"[0-9\u0E50-\u0E59]")

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            wav_name = parts[1]
            # parts[2] is raw transcription, parts[3] is normalized transcription (numbers spelled out in Thai)
            transcript = parts[3].strip()
            cleaned = clean_text(transcript)
            if args.skip_non_thai:
                if has_digit.search(cleaned) or not thai_only_pattern.match(cleaned):
                    continue
            wav_path = os.path.join(audio_dir, wav_name)
            if os.path.isfile(wav_path):
                records.append((wav_path, cleaned))

    if args.limit > 0:
        records = records[:args.limit]

    print(f"Evaluating {len(records)} clips from Google FLEURS (th_th) test set...")
    edits_c = units_c = 0
    edits_w = units_w = 0

    for i, (wav_path, ref) in enumerate(tqdm(records, desc="FLEURS Eval")):
        res = cactus.transcribe(wav_path, weights=args.weights, language="th")
        hyp = res["text"].strip()

        ec, nc = compute_cer(ref, hyp)
        ew, nw = compute_wer(ref, hyp)
        edits_c += ec
        units_c += nc
        edits_w += ew
        units_w += nw

        if i < 5:
            tqdm.write(f"[{i+1}] Ref: {ref}")
            tqdm.write(f"    Hyp: {hyp}\n")

    print(f"\n=== Google FLEURS (th_th) Test Results ({len(records)} clips) ===")
    print(f"Model: {args.weights}")
    print(f"Character Error Rate (CER): {edits_c / units_c:.2%}")
    print(f"Word Error Rate (WER):      {edits_w / units_w:.2%}")

if __name__ == "__main__":
    main()
