# utils/class_loader.py
from pathlib import Path

def load_class_names(path: Path):
    """Read class names from the provided file.
    Handles CSV, TSV, or space‑separated formats used in CUB‑200‑2011.
    Returns a list of clean class strings.
    """
    class_names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                _, name = line.split(",", 1)
            elif "\t" in line:
                _, name = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                name = parts[1] if len(parts) == 2 else parts[0]
            class_names.append(name.replace("_", " "))
    return class_names
