import re, collections

LINE = re.compile(r"^\[(\d+:\d+)\]\s+([A-Z]{3}\d{3}):\s*(.*)$")
FILLERS = {"mm", "hmm", "mm-hmm", "mm-hm", "uh-huh", "mhm", "um", "uh", "ah", "oh", "hm"}

def clean_audio(text):
    turns, words = [], collections.Counter()
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        t, spk, utt = m.groups()
        toks = utt.lower().split()
        if not toks or all(w.strip(".,'?") in FILLERS for w in toks):
            continue                      # drop pure backchannel like "MM", "HMM"
        words[spk] += len(toks)
        if turns and turns[-1][1] == spk:
            turns[-1][2].extend(toks)     # merge consecutive lines of same speaker
        else:
            turns.append([t, spk, toks])
    out = "\n".join(f"[{t}] {s}: {' '.join(w)}" for t, s, w in turns)
    total = sum(words.values()) or 1
    stats = {s: {"words": c, "share": round(c / total, 3)} for s, c in words.items()}
    return out, stats

def clean_ocr(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join hyphenated line breaks
    return re.sub(r"[ \t]+", " ", text).strip(), {}

def clean_text(text):
    return re.sub(r"\n{3,}", "\n\n", text).strip(), {}

def run(modality, text):
    return {"audio": clean_audio, "image": clean_ocr}.get(modality, clean_text)(text)
