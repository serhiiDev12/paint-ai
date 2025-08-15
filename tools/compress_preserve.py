from pathlib import Path
from PIL import Image

COMPRESSED_DIR = Path("compressed")
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]

def compress_preserve(input_dir, output_dir=COMPRESSED_DIR, quality=85):
    """
    Converts and compresses images to JPG format while preserving resolution.

    Args:
        input_dir (Path): Folder with original images
        output_dir (Path): Folder to store compressed JPGs
        quality (int): JPEG quality (lower = more compression)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_dir.glob("*"):
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        try:
            img = Image.open(file_path).convert("RGB")
            out_path = output_dir / (file_path.stem + ".jpg")
            img.save(out_path, format="JPEG", quality=quality)
            print(f"[✓] Compressed: {file_path.name} → {out_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to compress {file_path.name}: {e}")