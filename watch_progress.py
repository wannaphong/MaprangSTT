import re
import os

LOG_FILES = [
    "/root/.gemini/antigravity/brain/a44b5e4b-91d1-43d5-b475-76136a51f31d/.system_generated/tasks/task-152.log",
    "/root/.gemini/antigravity/brain/a44b5e4b-91d1-43d5-b475-76136a51f31d/.system_generated/tasks/task-175.log",
    "/workspace/whistle-thai/train.log"
]

for lf in LOG_FILES:
    if os.path.exists(lf) and os.path.getsize(lf) > 0:
        print(f"=== {lf} ({os.path.getsize(lf)} bytes) ===")
        with open(lf, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines[-15:]:
                print(line.rstrip())
