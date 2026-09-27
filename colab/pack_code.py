"""Zip the `chronocell` package for upload to Colab:  python colab/pack_code.py"""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "colab" / "chronocell_code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((root / "chronocell").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(root))
    z.write(root / "requirements.txt", "requirements.txt")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
