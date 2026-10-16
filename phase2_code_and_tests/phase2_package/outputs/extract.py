
from pathlib import Path
import gzip
import shutil

script_dir = Path(__file__).resolve().parent
folder = script_dir / "phase2"

if not folder.exists():
    raise FileNotFoundError(f"Folder not found: {folder}")

for gz_file in folder.glob("*.csv.gz"):
    csv_file = gz_file.with_suffix("")

    with gzip.open(gz_file, "rb") as source:
        with csv_file.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    print(f"Extracted: {csv_file.name}")

print("Extraction completed.")
