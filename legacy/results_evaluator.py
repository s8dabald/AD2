import json
from pathlib import Path

with (Path(__file__).resolve().parent / "data.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        entry = json.loads(line)

        if entry.get("precision", 0) > 0.9 and entry.get("recall", 0) > 0.9:
            print(entry.get("iteration"), entry.get("precision"), entry.get("recall"))
