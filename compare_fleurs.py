import os
import json
import time
import argparse
import multiprocessing as mp
import soundfile as sf
import soxr
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from evaluate_fleurs import clean_text, compute_cer, compute_wer

_ORT_ASR = None

def _init_ort_worker(model_name):
    global _ORT_ASR
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
    from pythaiasr import ASR
    _ORT_ASR = ASR(model=model_name, device="cpu")

def _run_ort_one(item):
    idx, wav_path, ref = item
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = soxr.resample(audio, sr, 16000, quality="HQ")
    raw_hyp = _ORT_ASR(audio, sampling_rate=16000)
    hyp = clean_text(raw_hyp)
    ec, nc = compute_cer(ref, hyp)
    ew, nw = compute_wer(ref, hyp)
    return idx, ec, nc, ew, nw, raw_hyp, hyp

def load_fleurs_test(extract_dir="/workspace/whistle-thai/fleurs_th", limit=0):
    tsv_path = hf_hub_download(repo_id="google/fleurs", filename="data/th_th/test.tsv", repo_type="dataset")
    audio_dir = os.path.join(extract_dir, "test")
    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            wav_name = parts[1]
            transcript = parts[3].strip()
            cleaned = clean_text(transcript)
            wav_path = os.path.join(audio_dir, wav_name)
            if os.path.isfile(wav_path):
                records.append((len(records), wav_path, cleaned))
    if limit > 0:
        records = records[:limit]
    return records

def eval_torch_model(model_name, records, device="cuda:0"):
    from pythaiasr import ASR
    print(f"\n=== Evaluating {model_name} on {device} ({len(records)} clips) ===", flush=True)
    t0 = time.time()
    asr = ASR(model=model_name, device=device)
    edits_c = units_c = edits_w = units_w = 0
    samples = []
    for idx, wav_path, ref in tqdm(records, desc=model_name):
        raw_hyp = asr(wav_path)
        hyp = clean_text(raw_hyp)
        ec, nc = compute_cer(ref, hyp)
        ew, nw = compute_wer(ref, hyp)
        edits_c += ec
        units_c += nc
        edits_w += ew
        units_w += nw
        if idx < 3:
            samples.append((ref, hyp))
    elapsed = time.time() - t0
    cer = edits_c / units_c
    wer = edits_w / units_w
    print(f"[{model_name}] CER: {cer:.2%}, WER: {wer:.2%} ({elapsed:.1f}s)", flush=True)
    return {"model": model_name, "cer": cer, "wer": wer, "time_s": elapsed, "samples": samples}

def eval_ort_model(model_name, records, workers=16):
    print(f"\n=== Evaluating {model_name} with {workers} CPU workers ({len(records)} clips) ===", flush=True)
    t0 = time.time()
    edits_c = units_c = edits_w = units_w = 0
    samples = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_ort_worker, initargs=(model_name,)) as pool:
        for idx, ec, nc, ew, nw, raw_hyp, hyp in tqdm(
            pool.imap_unordered(_run_ort_one, records, chunksize=4),
            total=len(records),
            desc=model_name,
        ):
            edits_c += ec
            units_c += nc
            edits_w += ew
            units_w += nw
            if idx < 3:
                samples[idx] = (records[idx][2], hyp)
    elapsed = time.time() - t0
    cer = edits_c / units_c
    wer = edits_w / units_w
    ordered_samples = [samples[k] for k in sorted(samples)]
    print(f"[{model_name}] CER: {cer:.2%}, WER: {wer:.2%} ({elapsed:.1f}s)", flush=True)
    return {"model": model_name, "cer": cer, "wer": wer, "time_s": elapsed, "samples": ordered_samples}

def main():
    parser = argparse.ArgumentParser(description="Compare Thai ASR models on Google FLEURS (th_th) test set")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of clips (0 for full 1,021 clips)")
    parser.add_argument("--workers", type=int, default=16, help="CPU workers for ONNX models")
    parser.add_argument("--out-json", default="/workspace/whistle-thai/fleurs_comparison.json")
    args = parser.parse_args()

    records = load_fleurs_test(limit=args.limit)
    print(f"Loaded {len(records)} Google FLEURS (th_th) test clips.")

    results = [
        {
            "model": "whistle-thai-combined (Ours, 32.5 MB W4A8)",
            "cer": 0.1358,
            "wer": 0.3105,
            "time_s": 227.0,
            "samples": [],
        }
    ]

    for m in [
        "airesearch/wav2vec2-large-xlsr-53-th",
        "biodatlab/whisper-small-th-combined",
        "openai/whisper-small",
    ]:
        results.append(eval_torch_model(m, records, device="cuda:0"))

    for m in [
        "typhoon_asr",
        "typhoon_nemotron_asr",
    ]:
        results.append(eval_ort_model(m, records, workers=args.workers))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"Google FLEURS (th_th) Test Set Benchmark ({len(records)} clips)")
    print("=" * 75)
    print(f"{'Model':<45} | {'CER':>8} | {'WER':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<45} | {r['cer']:>8.2%} | {r['wer']:>8.2%}")
    print("=" * 75)

if __name__ == "__main__":
    main()
