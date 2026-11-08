import pty
import os
import subprocess
import select

LOG_FILE = "/workspace/whistle-thai/train.log"
CMD = [
    "cactus", "whistle", "finetune",
    "train.jsonl", "validation.jsonl", "test.jsonl",
    "--language", "th",
    "--tokens", "tokens.txt",
    "--epochs", "3",
    "--batch-size", "16",
    "--lr", "1e-4",
    "--checkpoint-dir", "/workspace/whistle-thai/output/checkpoints",
    "--out", "/workspace/whistle-thai/output/adapter.safetensors"
]

os.chdir("/workspace/whistle-thai/dataset_whistle")

master, slave = pty.openpty()
proc = subprocess.Popen(CMD, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)

with open(LOG_FILE, "wb", buffering=0) as f:
    while True:
        r, _, _ = select.select([master], [], [], 0.5)
        if master in r:
            try:
                data = os.read(master, 1024)
                if not data:
                    break
                f.write(data)
            except OSError:
                break
        if proc.poll() is not None:
            break

proc.wait()
