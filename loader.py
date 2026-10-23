import json
from pathlib import Path

def load(modality, limit=None, offset=0):
    rows = []
    with open(Path("data") / f"{modality}.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit is not None and len(rows) >= limit:
                break
            r = json.loads(line)
            r["json_schema"] = json.loads(r["json_schema"])
            r["ground_truth"] = json.loads(r["ground_truth"])
            r["modality"] = modality
            rows.append(r)
    return rows
