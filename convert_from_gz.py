from pathlib import Path
import gzip, shutil

# Pfad als Path-Objekt anlegen
source = Path(r"C:\Users\s4totrin\Downloads\Sepsis Cases - Event Log_1_all\Sepsis Cases - Event Log.xes.gz")

# Ziel: gleiche Datei ohne .gz → ergibt .xes
target = source.with_suffix("")

with gzip.open(source, "rb") as f_in:
    with open(target, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Extracted to:", target)
