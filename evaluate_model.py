import os
import json
import wave
import tempfile
import numpy as np
import soundfile
import soxr
import cactus
from tqdm import tqdm

def compute_cer(ref, hyp):
    # Remove spaces for character error rate in Thai
    ref = list(ref.replace(" ", ""))
    hyp = list(hyp.replace(" ", ""))
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref)

def compute_wer(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/workspace/whistle-thai/output_combined/maprang.cact")
    parser.add_argument("--test-jsonl", default="/workspace/whistle-thai/dataset_whistle_combined/test.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Number of test clips to evaluate (0 for all)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.test_jsonl))
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        lines = lines[:args.limit]

    edits_c = units_c = 0
    edits_w = units_w = 0

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        for i, rec in enumerate(tqdm(lines, desc="Evaluating")):
            mp3_path = os.path.join(base_dir, rec["audio"])
            data, sr = soundfile.read(mp3_path, dtype="float32", always_2d=True)
            samples = data.mean(1)
            if sr != 16000:
                samples = soxr.resample(samples, sr, 16000, quality="HQ")
            pcm16 = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
            with wave.open(tmp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())

            res = cactus.transcribe(tmp_wav.name, weights=args.weights, language="th")
            hyp = res["text"].strip()
            ref = rec["text"]

            ec, nc = compute_cer(ref, hyp)
            ew, nw = compute_wer(ref, hyp)
            edits_c += ec
            units_c += nc
            edits_w += ew
            units_w += nw

            if i < 5:
                tqdm.write(f"[{i+1}] Ref: {ref}")
                tqdm.write(f"    Hyp: {hyp}\n")

    print(f"\n=== Evaluation Results ({len(lines)} test clips) ===")
    print(f"Model: {args.weights}")
    print(f"Character Error Rate (CER): {edits_c / units_c:.2%}")
    print(f"Word Error Rate (WER):      {edits_w / units_w:.2%}")
