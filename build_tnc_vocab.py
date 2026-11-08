import re
from pythainlp.corpus import tnc

# 1. Base Thai characters from previous tokens.txt (67 characters)
CHAR_PATH = "/workspace/whistle-thai/dataset_whistle/tokens.txt"
base_chars = [
    'ก', 'ข', 'ฃ', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ',
    'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล',
    'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ',
    'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๅ', '็', '่', '้', '๊', '๋', '์'
]

# 2. Load TNC word frequencies
freqs = tnc.word_freqs()
# Sort descending by frequency
sorted_tnc = sorted(freqs, key=lambda x: x[1], reverse=True)

thai_only_pattern = re.compile(r"^[\u0E00-\u0E7F]+$")
has_digit = re.compile(r"[0-9\u0E50-\u0E59]")

selected_words = []
for word, count in sorted_tnc:
    word = word.strip()
    # Must be pure Thai script, no digits, length >= 2
    if len(word) >= 2 and thai_only_pattern.match(word) and not has_digit.search(word):
        selected_words.append(word)
        if len(selected_words) >= 2000:
            break

print(f"Selected top {len(selected_words)} Thai words from TNC.")
print("Top 20 words from TNC:", [(w, c) for w, c in sorted_tnc if w in selected_words[:20]][:20])

# Combine base characters + TNC words
combined = list(dict.fromkeys(base_chars + selected_words))
print(f"Total vocabulary size: {len(combined)}")

TOKENS_OUT = "/workspace/whistle-thai/dataset_whistle/tokens.txt"
with open(TOKENS_OUT, "w", encoding="utf-8") as f:
    for item in combined:
        f.write(item + "\n")

print(f"Saved to {TOKENS_OUT}")
