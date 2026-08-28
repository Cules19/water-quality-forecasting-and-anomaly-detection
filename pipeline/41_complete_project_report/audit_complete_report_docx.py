from pathlib import Path
import sys
import zipfile

from docx import Document
from docx.oxml.ns import qn


def main():
    path = Path(sys.argv[1])
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        document_xml = archive.read("word/document.xml").decode("utf-8")

    doc = Document(path)
    section = doc.sections[0]
    normal = doc.styles["Normal"]
    failures = []

    if bad_member is not None:
        failures.append(f"bad_zip_member={bad_member}")
    if round(section.top_margin.inches, 3) != 1.0 or round(section.left_margin.inches, 3) != 1.0:
        failures.append("page_margins")
    if round(section.header_distance.inches, 3) != 0.492 or round(section.footer_distance.inches, 3) != 0.492:
        failures.append("header_footer_distance")
    if normal.font.name != "Calibri" or round(normal.font.size.pt, 1) != 11.0:
        failures.append("normal_font")
    if round(normal.paragraph_format.space_after.pt, 1) != 8.0:
        failures.append("normal_spacing")
    if abs(float(normal.paragraph_format.line_spacing) - 1.333) > 0.002:
        failures.append("normal_line_spacing")

    for index, table in enumerate(doc.tables, start=1):
        tbl_pr = table._tbl.tblPr
        tbl_w = tbl_pr.find(qn("w:tblW"))
        tbl_ind = tbl_pr.find(qn("w:tblInd"))
        if tbl_w is None or tbl_w.get(qn("w:type")) != "dxa" or int(tbl_w.get(qn("w:w"))) != 9360:
            failures.append(f"table_{index}_width")
        if tbl_ind is None or int(tbl_ind.get(qn("w:w"))) != 120:
            failures.append(f"table_{index}_indent")
        grid = table._tbl.tblGrid
        grid_widths = [int(col.get(qn("w:w"))) for col in grid.gridCol_lst]
        if sum(grid_widths) != 9360:
            failures.append(f"table_{index}_grid")

    image_count = len(doc.inline_shapes)
    described = 0
    for inline in doc.inline_shapes:
        descr = inline._inline.docPr.get("descr")
        if descr:
            described += 1
    if described != image_count:
        failures.append("image_alt_text")

    forbidden = ["PLACEHOLDER", "{{", "}}", "file-citation"]
    for token in forbidden:
        if token in document_xml:
            failures.append(f"forbidden_token={token}")

    print(f"zip_ok={bad_member is None}")
    print(f"paragraphs={len(doc.paragraphs)} tables={len(doc.tables)} images={image_count} images_with_alt={described}")
    print(f"failures={len(failures)}")
    for failure in failures:
        print(failure)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
