from pathlib import Path
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def main():
    pdf_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_paths = []
    for index in range(len(pdf)):
        bitmap = pdf[index].render(scale=2.0)
        image = bitmap.to_pil().convert("RGB")
        path = out_dir / f"page-{index + 1:02d}.png"
        image.save(path)
        page_paths.append(path)

    thumb_width = 850
    margin = 35
    label_height = 42
    for start in range(0, len(page_paths), 4):
        selected = page_paths[start : start + 4]
        thumbs = []
        for path in selected:
            image = Image.open(path).convert("RGB")
            scale = thumb_width / image.width
            thumb = image.resize((thumb_width, int(image.height * scale)), Image.Resampling.LANCZOS)
            thumbs.append(thumb)
        cell_height = max(image.height for image in thumbs) + label_height
        sheet = Image.new("RGB", (margin * 3 + thumb_width * 2, margin * 3 + cell_height * 2), "#D9DEE5")
        draw = ImageDraw.Draw(sheet)
        for offset, image in enumerate(thumbs):
            row, col = divmod(offset, 2)
            x = margin + col * (thumb_width + margin)
            y = margin + row * (cell_height + margin)
            page_number = start + offset + 1
            draw.text((x, y), f"Page {page_number}", fill="#203864")
            sheet.paste(image, (x, y + label_height))
        sheet.save(out_dir / f"contact-{start + 1:02d}-{start + len(selected):02d}.jpg", quality=90)

    print(f"pages={len(page_paths)} contacts={(len(page_paths) + 3) // 4}")


if __name__ == "__main__":
    main()
