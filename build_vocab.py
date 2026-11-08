import json
from collections import Counter
from pythainlp.tokenize import word_tokenize

CHAR_PATH = "/workspace/whistle-thai/dataset_whistle/tokens.txt"
with open(CHAR_PATH, "r", encoding="utf-8") as f:
    base_chars = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(base_chars)} base Thai characters.")

# Count word frequencies using PyThaiNLP
word_counter = Counter()
with open("/workspace/whistle-thai/dataset_whistle/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        words = word_tokenize(rec["text"], engine="newmm")
        for w in words:
            w = w.strip()
            if w and len(w) > 1 and not w.isascii():
                word_counter[w] += 1

print(f"Total unique multi-char Thai words: {len(word_counter)}")

# Select top 1,500 words with frequency >= 5
top_words = [word for word, count in word_counter.most_common(2000) if count >= 5]
print(f"Selected {len(top_words)} high-frequency words.")

# Combine base characters + top words
all_tokens = list(dict.fromkeys(base_chars + top_words))
print(f"Total tokens in new tokens.txt: {len(all_tokens)}")

TOKENS_OUT = "/workspace/whistle-thai/dataset_whistle/tokens.txt"
with open(TOKENS_OUT, "w", encoding="utf-8") as f:
    for tok in all_tokens:
        f.write(tok + "\n")

print(f"Successfully saved enhanced vocabulary to {TOKENS_OUT}")
