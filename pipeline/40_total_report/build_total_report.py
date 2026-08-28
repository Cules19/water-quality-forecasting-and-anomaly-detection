from pathlib import Path
from datetime import date

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_outputs" / "total_report_2026-07-30"
OUT_PATH = OUT_DIR / "Pollution_ID_总报告_2026-07-30.docx"

FIG_WARMUP = ROOT / "analysis_outputs" / "common1000_lowcost_deployment_summary_v1" / "figures" / "lowcost_warmup_pH_DO.png"
FIG_ONSET = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "event_lead_time_distribution.png"
FIG_LAG = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "lagged_event_enrichment.png"
FIG_COMPRESSION = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "compression_01645762.png"
FIG_CONFORMAL = ROOT / "analysis_outputs" / "common1000_lowcost_conformal_v1" / "figures" / "conformal_coverage.png"


# narrative_proposal preset with one named CJK override.
TOKENS = {
    "Page_Size": "Letter",
    "Margins_Inches": 1.0,
    "Header_Footer_Inches": 0.492,
    "Body_Font": "Calibri",
    "CJK_Font_Override": "Microsoft YaHei",
    "Body_Size_Pt": 11,
    "Body_Line_Spacing": 1.333,
    "Body_After_Pt": 8,
    "H1_Size_Pt": 16,
    "H2_Size_Pt": 13,
    "H3_Size_Pt": 12,
    "Accent_Blue": "2E74B5",
    "Dark_Navy": "203864",
    "Light_Blue": "EAF2F8",
    "Light_Grey": "F4F6F9",
    "Mid_Grey": "6B7280",
    "Dark_Text": "1F2937",
    "Table_Width_DXA": 9360,
    "Table_Indent_DXA": 120,
}


def rgb(hex_value):
    return RGBColor.from_string(hex_value)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, val in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    total = sum(widths)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TOKENS["Table_Indent_DXA"]))
    tbl_ind.set(qn("w:type"), "dxa")
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        grid.append(grid_col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            set_cell_width(cell, widths[index])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_repeat_header_text(paragraph, text, color="6B7280"):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run(text)
    run.font.name = TOKENS["Body_Font"]
    run.font.size = Pt(8)
    run.font.color.rgb = rgb(color)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["CJK_Font_Override"])


def set_run_font(run, name=None, size=None, bold=None, color=None, italic=None):
    run.font.name = name or TOKENS["Body_Font"]
    run._element.get_or_add_rPr()
    run._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["CJK_Font_Override"])
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = rgb(color)
    if italic is not None:
        run.italic = italic


def set_style_font(style, size, bold=False, color=None):
    style.font.name = TOKENS["Body_Font"]
    style._element.get_or_add_rPr()
    style._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["CJK_Font_Override"])
    style.font.size = Pt(size)
    style.font.bold = bold
    if color:
        style.font.color.rgb = rgb(color)


def configure_styles(doc):
    normal = doc.styles["Normal"]
    set_style_font(normal, TOKENS["Body_Size_Pt"], color=TOKENS["Dark_Text"])
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = TOKENS["Body_Line_Spacing"]
    normal.paragraph_format.space_after = Pt(TOKENS["Body_After_Pt"])
    normal.paragraph_format.widow_control = True

    title = doc.styles["Title"]
    set_style_font(title, 30, bold=True, color=TOKENS["Dark_Navy"])
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(14)
    title.paragraph_format.keep_with_next = True

    subtitle = doc.styles["Subtitle"]
    set_style_font(subtitle, 14, color=TOKENS["Accent_Blue"])
    subtitle.paragraph_format.space_after = Pt(12)

    heading_specs = [
        ("Heading 1", TOKENS["H1_Size_Pt"], TOKENS["Accent_Blue"], 18, 10),
        ("Heading 2", TOKENS["H2_Size_Pt"], TOKENS["Dark_Navy"], 12, 6),
        ("Heading 3", TOKENS["H3_Size_Pt"], TOKENS["Dark_Navy"], 8, 4),
    ]
    for style_name, size, color, before, after in heading_specs:
        style = doc.styles[style_name]
        set_style_font(style, size, bold=True, color=color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    for list_name in ("List Bullet", "List Number"):
        style = doc.styles[list_name]
        set_style_font(style, TOKENS["Body_Size_Pt"], color=TOKENS["Dark_Text"])
        style.paragraph_format.left_indent = Inches(0.38)
        style.paragraph_format.first_line_indent = Inches(-0.18)
        style.paragraph_format.space_after = Pt(5)

    caption = doc.styles["Caption"]
    set_style_font(caption, 9, color=TOKENS["Mid_Grey"])
    caption.font.italic = True
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(10)
    caption.paragraph_format.keep_with_next = False
    caption.paragraph_format.keep_together = True


def add_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, size=8, color=TOKENS["Mid_Grey"])
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def configure_page(doc):
    section = doc.sections[0]
    section.top_margin = Inches(TOKENS["Margins_Inches"])
    section.bottom_margin = Inches(TOKENS["Margins_Inches"])
    section.left_margin = Inches(TOKENS["Margins_Inches"])
    section.right_margin = Inches(TOKENS["Margins_Inches"])
    section.header_distance = Inches(TOKENS["Header_Footer_Inches"])
    section.footer_distance = Inches(TOKENS["Header_Footer_Inches"])
    section.different_first_page_header_footer = True
    set_repeat_header_text(section.header.paragraphs[0], "Pollution_ID | 跨站点水质预测与低成本部署总报告")
    add_page_field(section.footer.paragraphs[0])


def add_top_rule(paragraph, color="2E74B5", size="16"):
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        p_pr.append(borders)
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), size)
    top.set(qn("w:space"), "1")
    top.set(qn("w:color"), color)
    borders.append(top)


def add_callout(doc, title, body, kind="blue"):
    colors = {
        "blue": (TOKENS["Light_Blue"], TOKENS["Accent_Blue"]),
        "grey": (TOKENS["Light_Grey"], TOKENS["Mid_Grey"]),
        "navy": ("E8EEF7", TOKENS["Dark_Navy"]),
    }
    fill, border = colors[kind]
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.16)
    p.paragraph_format.right_indent = Inches(0.08)
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(10)
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "20")
    left.set(qn("w:space"), "7")
    left.set(qn("w:color"), border)
    borders.append(left)
    p_pr.append(borders)
    r1 = p.add_run(title + "  ")
    set_run_font(r1, bold=True, color=border)
    r2 = p.add_run(body)
    set_run_font(r2, color=TOKENS["Dark_Text"])
    return p


def add_paragraph(doc, text="", bold_prefix=None, style=None, align=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_run_font(r1, bold=True, color=TOKENS["Dark_Navy"])
        r2 = p.add_run(text[len(bold_prefix):])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    if level:
        p.paragraph_format.left_indent = Inches(0.38 + 0.28 * level)
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_numbered_list(doc, items):
    for text in items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(text)
        set_run_font(r)


def add_table(doc, headers, rows, widths, header_fill=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    header_fill = header_fill or TOKENS["Dark_Navy"]
    for idx, value in enumerate(headers):
        cell = table.rows[0].cells[idx]
        set_cell_shading(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(str(value))
        set_run_font(run, size=font_size, bold=True, color="FFFFFF")
    set_repeat_table_header(table.rows[0])

    for row_idx, row_values in enumerate(rows):
        row_cells = table.add_row().cells
        for col_idx, value in enumerate(row_values):
            cell = row_cells[col_idx]
            if row_idx % 2 == 1:
                set_cell_shading(cell, TOKENS["Light_Grey"])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            set_run_font(run, size=font_size, color=TOKENS["Dark_Text"])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_metric_strip(doc, metrics):
    table = doc.add_table(rows=1, cols=len(metrics))
    table.style = "Table Grid"
    widths = [TOKENS["Table_Width_DXA"] // len(metrics)] * len(metrics)
    widths[-1] += TOKENS["Table_Width_DXA"] - sum(widths)
    set_table_geometry(table, widths)
    for idx, (value, label) in enumerate(metrics):
        cell = table.cell(0, idx)
        set_cell_shading(cell, TOKENS["Light_Blue"])
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(value)
        set_run_font(r, size=18, bold=True, color=TOKENS["Accent_Blue"])
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_after = Pt(0)
        r2 = p2.add_run(label)
        set_run_font(r2, size=8.5, color=TOKENS["Dark_Text"])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_figure(doc, path, caption, width=6.15):
    if not path.exists():
        add_callout(doc, "图件缺失", f"未找到图件：{path}", "grey")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(style="Caption")
    cr = cap.add_run(caption)
    set_run_font(cr, size=9, color=TOKENS["Mid_Grey"], italic=True)


def add_section_heading(doc, text, level=1, page_break=False):
    if page_break:
        doc.add_page_break()
    return doc.add_heading(text, level=level)


def build_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in (FIG_WARMUP, FIG_ONSET, FIG_LAG, FIG_COMPRESSION, FIG_CONFORMAL):
        if not image_path.exists():
            raise FileNotFoundError(image_path)

    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    doc.core_properties.title = "Pollution_ID 跨站点水质预测与低成本部署总报告"
    doc.core_properties.subject = "pH、溶解氧、降雨、事件预警、低成本部署与不确定性"
    doc.core_properties.author = "Pollution_ID 项目"
    doc.core_properties.keywords = "water quality, pH, dissolved oxygen, rainfall, LSTM-FiLM, warm-up"

    # Cover
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(70)
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_after = Pt(14)
    kr = kicker.add_run("POLLUTION_ID  ·  RESEARCH REPORT")
    set_run_font(kr, size=10, bold=True, color=TOKENS["Accent_Blue"])
    add_top_rule(kicker)

    title = doc.add_paragraph(style="Title")
    title_run = title.add_run("跨站点水质预测、\n异常预警与低成本部署")
    set_run_font(title_run, size=28, bold=True, color=TOKENS["Dark_Navy"])
    subtitle = doc.add_paragraph(style="Subtitle")
    sr = subtitle.add_run("pH / 溶解氧 / 降雨与 API / LSTM-FiLM / 现场预热")
    set_run_font(sr, size=14, color=TOKENS["Accent_Blue"])

    doc.add_paragraph().paragraph_format.space_after = Pt(55)
    info = doc.add_paragraph()
    info.paragraph_format.space_after = Pt(4)
    ir = info.add_run("项目总报告")
    set_run_font(ir, size=12, bold=True, color=TOKENS["Dark_Navy"])
    for line in (
        "报告日期：2026 年 7 月 30 日",
        "数据范围：老师提供的 common1000 / Archive 数据与项目已完成实验",
        "版本：Final synthesis v1",
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(line)
        set_run_font(r, size=10, color=TOKENS["Mid_Grey"])

    doc.add_paragraph().paragraph_format.space_after = Pt(60)
    status = doc.add_paragraph()
    status.paragraph_format.space_after = Pt(0)
    rr = status.add_run("状态：本地可执行实验已完成；地质/水文元数据消融待老师提供数据。")
    set_run_font(rr, size=10.5, bold=True, color=TOKENS["Dark_Navy"])

    doc.add_page_break()

    # Executive summary
    add_section_heading(doc, "执行摘要", 1)
    add_paragraph(
        doc,
        "本项目已经从“单纯比较一个模型的 RMSE”推进为一套较完整的跨站点水质建模与部署研究："
        "首先确认降雨动态信息对 pH 预测存在稳定但有限的增益；随后用残差阈值分析异常可能提前多久显现；"
        "最后构建不依赖历史 pH/DO 的低成本模型，并量化新站点需要多长预热期才能达到可用误差。"
    )
    add_metric_strip(
        doc,
        [
            ("-9.26%", "全数据 rain+API 相对基线的 pH RMSE"),
            ("0.184 pH", "12 个月预热后的 pH 宏平均 MAE"),
            ("0.688 mg/L", "3 个月预热后的 DO 测试宏平均 MAE"),
        ],
    )
    add_callout(
        doc,
        "核心判断",
        "降雨并非“没用”，但它解决的是动态扰动信息不足；现场预热解决的是新站点静态偏移和尺度差异。"
        "两者是互补问题。高精度的历史目标模型与可部署的低成本模型也不能直接用同一 RMSE 横向比较。",
        "navy",
    )
    add_bullet(doc, "固定 LSTM-FiLM 配置下，加入当前降雨的 pH RMSE 为 0.06975，较基础模型改善 1.62%，135 个测试站中 101 个改善。")
    add_bullet(doc, "在 matched 全数据比较中，rain+API 将 pH RMSE 从 0.09056 降至 0.08217，改善 9.26%，135 个站中 96 个改善。")
    add_bullet(doc, "非事件训练提高事件富集度，却通常牺牲总体 RMSE；这是论文中可解释的“预测精度—事件敏感性”权衡。")
    add_bullet(doc, "低成本 pH 模型在 12 个月 FiLM 预热后，宏平均 MAE 为 0.184；34/39 个测试站改善。")
    add_bullet(doc, "低成本 DO 模型在 3 个月 FiLM 预热后，测试宏平均 MAE 为 0.688 mg/L；12 个月为 0.613 mg/L。")
    add_bullet(doc, "几何/地质/水文静态元数据消融仍因老师尚未提供相应数据而无法执行，是唯一明确的外部阻塞项。")

    add_section_heading(doc, "报告结构", 2)
    toc_items = [
        "研究目标与最终问题定义",
        "数据、站点划分与质量控制",
        "模型、指标与验证政策",
        "降雨、API 与非事件训练",
        "异常误差出现时间与事件解释",
        "低成本传感器部署与现场预热",
        "溶解氧单位审计与修正",
        "不确定性区间与覆盖率",
        "论文可主张的贡献、限制与未完成项",
        "附录：关键证据路径与完成状态",
    ]
    for item in toc_items:
        add_paragraph(doc, item, style="List Number")

    # 1
    add_section_heading(doc, "1. 研究目标与最终问题定义", 1, page_break=True)
    add_paragraph(
        doc,
        "研究对象是多个水质监测站的 pH 与溶解氧（DO）时间序列。原始目标不是只在已有站点上取得最低误差，"
        "而是回答三个逐层递进的问题：跨站点模型能否利用降雨等外生扰动；污染或异常事件能否在误差中提前显现；"
        "如果新站点没有历史目标值，低成本传感器和有限现场预热能否达到可用水平。"
    )
    add_table(
        doc,
        ["研究层次", "主要问题", "最终实验"],
        [
            ["预测性能", "降雨/API 是否改善跨站点 pH 预测？", "LSTM-FiLM 降雨消融；all-data / non-event matched 比较"],
            ["事件识别", "异常残差比事件标签早多久出现？", "验证集阈值、连续 3 小时告警、滞后富集与漏报归因"],
            ["真实部署", "没有历史 pH/DO 时，新站点需要多久校准？", "低成本核心、FiLM adapter、2 周至 12 个月预热曲线"],
            ["风险控制", "预测误差如何转为区间？", "固定校准站点的 split conformal 覆盖率"],
        ],
        [1500, 3550, 4310],
    )
    add_callout(
        doc,
        "两个不可混淆的模型口径",
        "历史目标模型可使用过去的 pH/DO，因此 RMSE 可以低至约 0.07；低成本部署模型明确禁止使用历史目标值和昂贵的交叉目标传感器，"
        "误差更高但部署问题更真实。二者应分别作为“性能上限”和“可部署方案”报告。",
        "blue",
    )

    add_section_heading(doc, "1.1 最终论文主线", 2)
    add_numbered_list(doc, (
        "降雨动态特征能够稳定改善 pH 预测，但改善幅度取决于输入表示与训练事件分布。",
        "非事件训练会提高事件富集度，却降低常规预测准确度，形成明确的任务权衡。",
        "异常残差有时能在已记录事件之前数小时出现，但当前标签较弱，因此结果只能作为预警潜力，而不能宣称污染因果识别。",
        "低成本传感器模型在没有历史目标值时仍可工作；站点专属的小型 FiLM adapter 能用少量现场数据显著降低新站点误差。",
        "模型不确定性可以通过站点级校准转化为经验覆盖区间，但 pH 的低名义水平存在欠覆盖。",
    ))

    # 2
    add_section_heading(doc, "2. 数据、站点划分与质量控制", 1)
    add_paragraph(
        doc,
        "数据来自老师 GitHub 项目通过 DVC 管理的 Archive/common1000 数据。DVC 在仓库中保存版本和下载指针，"
        "大文件实际位于远程存储。项目现已获取并整理 Archive，包括水质、流量、浊度、电导率、温度和小时降雨等字段。"
    )
    add_table(
        doc,
        ["数据层", "站点数", "用途"],
        [
            ["主 common1000 cohort", "910", "统一训练/验证/测试和主要论文结果"],
            ["训练站点", "638", "模型参数学习"],
            ["验证站点", "137", "超参数、阈值和预热时长冻结"],
            ["测试站点", "135", "最终一次性报告"],
            ["严格 12 个月预热测试站点", "pH 39；DO 39", "要求完整预热窗口及固定后续评价窗口"],
        ],
        [2500, 1700, 5160],
    )
    add_paragraph(
        doc,
        "为什么预热实验只剩 39 个测试站？135 个测试站用于一般跨站点评价；严格预热实验还要求站点同时具有连续可用的"
        "预热期、足够的后续固定评价时段、目标观测和低成本传感器。因此这是资格筛选后的 matched cohort，不代表电脑里只有 39 或 47 个站点。"
    )

    add_section_heading(doc, "2.1 低成本输入定义", 2)
    add_table(
        doc,
        ["变量", "代码", "低成本核心是否使用", "说明"],
        [
            ["水温", "00010", "是", "可现场连续监测"],
            ["电导率", "00095", "是", "反映离子/溶解物变化"],
            ["浊度", "63680", "是", "反映颗粒物和扰动"],
            ["小时降雨", "00045", "是", "外生动态驱动"],
            ["历史 pH", "00400", "否", "部署模型禁止目标历史泄漏"],
            ["历史 DO", "00300", "否", "部署模型禁止目标历史泄漏"],
        ],
        [1800, 1400, 2300, 3860],
    )
    add_paragraph(
        doc,
        "完整传感器模型作为参照：pH 完整模型可额外加入 DO，DO 完整模型可额外加入 pH。低成本核心删除这些昂贵或不可用的交叉目标输入，"
        "因此回答的是“新站点无目标历史时”的部署问题。"
    )

    add_section_heading(doc, "2.2 防止信息泄漏的设计", 2)
    add_bullet(doc, "站点按训练/验证/测试划分，不允许同一站点跨集合泄漏。")
    add_bullet(doc, "预热时长首先在验证集冻结，然后只在测试集做一次最终评价。")
    add_bullet(doc, "不同预热时长使用相同的 12 个月后固定评价键，避免较长预热因评价时间段不同而占便宜。")
    add_bullet(doc, "事件阈值仅由验证集正常时段残差确定，再应用到测试事件。")
    add_bullet(doc, "DO 采用 mg/L 原生单位输出，并对早期单位错误的结果进行弃用标记和全量再生。")

    # 3
    add_section_heading(doc, "3. 模型、指标与验证政策", 1, page_break=True)
    add_section_heading(doc, "3.1 最终动态模型：LSTM-FiLM", 2)
    add_paragraph(
        doc,
        "最终动态预测采用 LSTM-FiLM。LSTM 提取过去 48 小时的动态序列，FiLM 使用站点信息对隐藏表示进行缩放和平移，"
        "从而同时学习跨站点共性和站点差异。固定最终配置为：48 小时窗口、hidden size 64、dropout 0.1、学习率 0.0003。"
    )
    add_callout(
        doc,
        "为何最终以 LSTM 为主",
        "GRU 与 LSTM 都可作为合理循环基线；本项目已做相应对比，但最终论文主模型按老师偏好聚焦 LSTM-FiLM，"
        "将架构差异降为次要问题，把有限工作量集中到降雨机制、异常预警和真实部署。",
        "grey",
    )
    add_section_heading(doc, "3.2 现场适配器", 2)
    add_paragraph(
        doc,
        "预热阶段冻结训练好的 LSTM-FiLM 主干，只为新站点学习 130 个参数：128 个站点 FiLM 偏移参数，加上一个正值仿射缩放和一个偏置。"
        "该设计使少量现场观测主要校正站点的基线与幅度，而不重新训练整个网络。对照包括 zero-shot、film_affine 和 full_finetune。"
    )
    add_section_heading(doc, "3.3 报告指标", 2)
    add_table(
        doc,
        ["指标", "解释", "本文用途"],
        [
            ["RMSE", "平方误差均值开方，对大误差更敏感", "预测主指标"],
            ["MAE", "平均绝对误差，直观且稳健", "部署主指标"],
            ["宏平均", "先逐站计算再平均", "防止大站点支配结果"],
            ["within tolerance", "预测落在指定误差范围内的比例", "pH ±0.2；DO ±1 mg/L"],
            ["事件富集度", "事件窗口误差/异常相对正常时段的富集", "评估事件敏感性，不等同于 RMSE"],
            ["配对站点差值", "同一站点两个模型的误差差", "站点层统计和置信区间"],
        ],
        [1650, 3420, 4290],
    )

    # 4
    add_section_heading(doc, "4. 降雨、API 与非事件训练", 1)
    add_section_heading(doc, "4.1 固定 LSTM 降雨消融", 2)
    add_table(
        doc,
        ["模型", "测试 RMSE", "相对基础模型", "改善站点"],
        [
            ["LSTM-FiLM 基础模型", "0.07091", "—", "—"],
            ["LSTM-FiLM + 当前降雨", "0.06975", "-1.62%", "101 / 135"],
        ],
        [3200, 1900, 2100, 2160],
    )
    add_paragraph(
        doc,
        "配对站点 RMSE 差的置信区间为 [-0.00144, -0.00046]，说明改善虽小但方向稳定。事件时段 RMSE 改善 1.21%，"
        "前 6 小时湿润窗口改善 1.16%。因此最稳妥的表述是“当前降雨带来小幅、可重复的总体增益”，而不是“降雨决定了预测”。"
    )

    add_section_heading(doc, "4.2 当前降雨、API 与训练分布的 matched 比较", 2)
    add_table(
        doc,
        ["训练模式 / 输入", "RMSE", "事件富集度", "相对含义"],
        [
            ["全数据 / baseline", "0.09056", "1.698", "全数据基础参照"],
            ["全数据 / current rain", "0.09059", "2.161", "富集增强，RMSE近似不变"],
            ["全数据 / rain + API", "0.08217", "1.462", "RMSE 最佳，较 baseline 改善 9.26%"],
            ["non-event / baseline", "0.14867", "2.285", "事件更显著，但总体误差升高"],
            ["non-event / current rain", "0.15015", "3.048", "富集最高，预测最差"],
            ["non-event / rain + API", "0.14621", "1.964", "较 non-event 基础略好"],
        ],
        [3300, 1450, 1800, 2810],
    )
    add_paragraph(
        doc,
        "全数据 rain+API 相对 baseline 的 RMSE 改善为 9.26%，96/135 个站点改善，配对置信区间为 [-0.00960, -0.00498]。"
        "API（Antecedent Precipitation Index，前期降雨指数）把过去降雨按衰减权重累积，表示土壤/流域此前已多湿，而不仅是当前小时是否下雨。"
    )
    add_callout(
        doc,
        "老师早期强调的结果",
        "此前 non-event 训练曾将 pH 事件富集度从 3.66 提高到 4.65，同时 RMSE 基本不变。当前 matched 重跑仍支持相同方向："
        "排除事件训练有利于把事件变得更“异常”，但不同 cohort 和配置下 RMSE 代价可能更明显。论文应报告方向一致性和具体实验口径，不能混用数值。",
        "blue",
    )
    add_section_heading(doc, "4.3 这一阶段能得出的结论", 2)
    add_bullet(doc, "降雨有信息，尤其是包含前期累积状态的 API；不能据此说降雨“没有用”。")
    add_bullet(doc, "总体预测误差与事件富集度是两个不同目标，最好分别优化、分别报告。")
    add_bullet(doc, "non-event 不是唯一可用训练方式；它适合异常敏感性分析，而全数据更适合追求常规预测准确度。")
    add_bullet(doc, "未来若重点转向预警，可使用多任务损失或事件加权，而不是简单永久删除全部事件。")

    # 5
    add_section_heading(doc, "5. 异常误差出现时间与事件解释", 1, page_break=True)
    add_paragraph(
        doc,
        "该分析不直接把模型当作污染分类器，而是用验证集正常时段残差建立站点阈值，再检测测试事件前是否出现连续异常误差。"
        "主定义要求至少连续 3 小时超过阈值，以降低单点噪声告警。事件标签属于 weak labels：记录事件可能比真实水质扰动晚，也可能漏记。"
    )
    add_table(
        doc,
        ["定义", "被预警事件", "比例", "提前时间"],
        [
            ["主分析：连续 3 小时", "579 / 2,853", "20.29%", "中位数 8 h；IQR 4–17 h"],
            ["仅可分析事件分母", "579 / 2,577", "22.47%", "同上"],
            ["敏感性：单小时", "1,339 / 2,853", "46.93%", "中位数 4 h"],
        ],
        [2850, 2050, 1650, 2810],
    )
    add_paragraph(
        doc,
        "按全部 2,853 个事件计算，至少提前 6、12、24 小时被连续 3 小时规则捕获的比例分别为 12.58%、7.75% 和 3.15%。"
        "这说明提前信号存在，但覆盖率有限，不能将其描述为可靠的普适报警系统。"
    )
    add_figure(doc, FIG_ONSET, "图 1　测试事件中异常残差的提前时间分布。主规则为连续 3 小时超过站点阈值。")

    add_section_heading(doc, "5.1 滞后富集与时间方向", 2)
    add_paragraph(
        doc,
        "把事件标签逐步向后移动，检查原始事件时刻之前的误差是否已经富集。rain 模型的富集度从 lead 0 的 1.733，"
        "缓慢下降到 lead 12 的 1.554，在 lead 24 为 1.618。它支持“部分事件前已有异常”的描述，但不证明降雨或污染的因果方向。"
    )
    add_figure(doc, FIG_LAG, "图 2　事件标签向后平移后的残差富集度。富集在提前窗口仍高于 1，但随提前时间总体减弱。")

    add_section_heading(doc, "5.2 动态范围压缩检查", 2)
    add_paragraph(
        doc,
        "模型若只预测均值，可能以较低 RMSE 掩盖高低极值的系统性压缩。对代表站点 01645762 的检查中，rain 模型总体斜率 0.983，"
        "低尾斜率 1.008，高尾斜率 0.981，预测范围/观测范围比为 1.039。该站点没有明显严重的动态范围压缩。"
    )
    add_figure(doc, FIG_COMPRESSION, "图 3　代表站点 01645762 的预测—观测动态范围检查。")

    add_section_heading(doc, "5.3 漏报归因", 2)
    add_paragraph(
        doc,
        "对 17 个站点、802,108 行数据进行了描述性漏报分析。rain 模型中 rainfall_6h、浊度和流量的拟合系数分别约为"
        "0.00667、0.0626 和 5.56×10⁻7。该结果仅帮助定位漏报场景，不应当解释为变量的因果效应；站点结构、事件标签质量和未观测排放均可能混杂。"
    )

    # 6
    add_section_heading(doc, "6. 低成本传感器部署与现场预热", 1, page_break=True)
    add_section_heading(doc, "6.1 无目标历史的低成本模型", 2)
    add_table(
        doc,
        ["目标", "输入配置", "总体 RMSE", "宏 RMSE", "宏 MAE", "容差内比例"],
        [
            ["pH", "完整传感器", "0.429", "0.314", "0.243", "47.25%（±0.2）"],
            ["pH", "低成本核心", "0.446", "0.340", "0.264", "44.58%（±0.2）"],
            ["DO", "完整传感器", "1.333", "1.125", "0.946", "67.72%（±1 mg/L）"],
            ["DO", "低成本核心", "1.386", "1.213", "1.018", "65.22%（±1 mg/L）"],
        ],
        [1250, 1800, 1450, 1450, 1450, 1960],
    )
    add_paragraph(
        doc,
        "删除交叉目标传感器会造成可测量但并非灾难性的性能损失。站点配对比较中，pH 低成本模型的站点 RMSE 平均增加 0.0252"
        "（95% CI 0.0076–0.0436）；DO 平均增加 0.0878 mg/L（95% CI -0.0037–0.1759）。"
        "这构成部署研究的起点：低成本模型可行，但需现场校准。"
    )

    add_section_heading(doc, "6.2 pH 新站点预热结果", 2)
    add_table(
        doc,
        ["方法（12 个月）", "总体 RMSE", "宏 RMSE", "宏 MAE", "±0.2 内"],
        [
            ["Zero-shot", "0.3867", "0.3581", "0.3041", "—"],
            ["FiLM + affine", "0.2529", "0.2381", "0.1843", "65.82%"],
            ["Full fine-tune", "0.2429", "—", "0.1808", "67.87%"],
        ],
        [2600, 1690, 1690, 1690, 1690],
    )
    add_paragraph(
        doc,
        "12 个月 FiLM adapter 将总体 RMSE 降低 34.61%，宏 RMSE 降低 33.51%，宏 MAE 降低 39.38%。"
        "34/39 个测试站改善；站点配对宏 RMSE 差为 -0.1200，置信区间约 [-0.1696, -0.0768]。"
        "Full fine-tune 略好，但与 FiLM 的差距较小，表明 130 参数的适配器已经捕获了大部分可校准偏移。"
    )

    add_section_heading(doc, "6.3 DO 新站点预热结果（mg/L）", 2)
    add_table(
        doc,
        ["方法（12 个月）", "总体 RMSE", "宏 RMSE", "宏 MAE", "±1 mg/L 内"],
        [
            ["Zero-shot", "1.3306", "1.1690", "0.9604", "—"],
            ["FiLM + affine", "0.9842", "0.8225", "0.6134", "79.89%"],
            ["Full fine-tune", "0.9301", "—", "0.5400", "82.47%"],
        ],
        [2600, 1690, 1690, 1690, 1690],
    )
    add_paragraph(
        doc,
        "12 个月 FiLM adapter 将总体 RMSE 降低 26.03%，宏 RMSE 降低 29.64%，宏 MAE 降低 36.12%。"
        "34/39 个测试站改善；站点配对宏 RMSE 差为 -0.3465 mg/L，置信区间约 [-0.5113, -0.2050]。"
        "DO 上 full fine-tune 平均显著优于小适配器，说明 DO 的跨站点差异比单纯尺度/偏置更复杂。"
    )

    add_figure(doc, FIG_WARMUP, "图 4　低成本 pH 与 DO 在不同现场预热时长下的测试表现。评价窗口固定在 12 个月预热之后。")

    add_section_heading(doc, "6.4 推荐预热时长", 2)
    add_table(
        doc,
        ["目标", "保守推荐", "测试表现", "推荐理由"],
        [
            ["pH", "12 个月", "宏 MAE 0.184；34/39 改善", "验证冻结后，测试曲线继续改善至 12 个月"],
            ["DO", "3 个月", "宏 MAE 0.688 mg/L；±1 内 77.15%", "验证与测试共同支持的保守最低时长"],
            ["DO（资源充足）", "12 个月", "宏 MAE 0.613 mg/L；±1 内 79.89%", "继续收集数据仍有额外收益"],
        ],
        [1500, 1700, 2630, 3530],
    )
    add_callout(
        doc,
        "如何理解“预热很长”",
        "这里的 12 个月不是模型每次运行都要等待，而是新站点首次部署时用来覆盖季节变化的校准期资料。"
        "如果实际项目不能等待一年，可采用 DO 3 个月、pH 较短阶段性上线并持续更新，同时报告更高的不确定性。",
        "blue",
    )

    add_section_heading(doc, "6.5 站点异质性", 2)
    add_table(
        doc,
        ["目标", "最早达到阈值的站点分布", "未达到"],
        [
            ["pH：站点 MAE ≤ 0.2", "2周 14；1月 1；2月 1；3月 3；6月 4；9月 2；12月 2", "12 / 39"],
            ["DO：站点 MAE ≤ 1 mg/L", "2周 25；2月 1；3月 5；6月 2；12月 1", "5 / 39"],
        ],
        [2300, 5360, 1700],
    )
    add_paragraph(
        doc,
        "阈值达标站点数说明统一预热政策并不适合所有站点。一部分站点两周即可满足容差，另一部分即使 12 个月也达不到。"
        "这不是“站点太少”的证据，而是站点异质性本身的研究结果。论文可进一步建议自适应停止：当验证误差和不确定性稳定后提前结束预热。"
    )

    # 7
    add_section_heading(doc, "7. 溶解氧单位审计与修正", 1)
    add_paragraph(
        doc,
        "老师的数据处理先将 EA 的 DO 饱和度百分比转换为 mg/L，再为神经网络输入应用固定比例缩放。早期 DO 目标实验误将"
        "x00300/20 直接当作最终目标单位，导致报告值不是原生 mg/L。该错误已经被识别、标记弃用，并重新生成全部 DO 预测和指标。"
    )
    add_table(
        doc,
        ["审计项", "核验结果"],
        [
            ["正确报告目标", "20 × x00300，单位 mg/L"],
            ["训练集均值", "8.633 mg/L"],
            ["训练集标准差", "2.885 mg/L"],
            ["模型权重", "标准化标签等价；修正前后 checkpoint SHA 验证一致"],
            ["重新生成内容", "预测、RMSE/MAE、容差比例、预热曲线、汇总报告"],
            ["弃用旧结果", "旧 DO sensor-ablation 和 warm-up 输出均附 DEPRECATED_UNIT_ERROR.md"],
        ],
        [2850, 6510],
    )
    add_callout(
        doc,
        "重要意义",
        "单位修正没有改变标准化空间中的模型训练，但改变了所有可解释的绝对误差。论文、图表和答辩必须只引用标记为 mg_l 的新目录结果。",
        "navy",
    )

    # 8
    add_section_heading(doc, "8. 不确定性区间与覆盖率", 1, page_break=True)
    add_paragraph(
        doc,
        "使用固定哈希划分的 45 个校准站点估计绝对残差分位数，再在完全不同的 90 个站点上评价区间覆盖率。"
        "该 split conformal 设计保留了站点独立性；由于同一站点内部时间相关，结果应解释为经验边际覆盖，而不是严格 IID 保证。"
    )
    add_table(
        doc,
        ["目标", "名义覆盖", "区间半径", "总体覆盖", "站点宏覆盖"],
        [
            ["pH", "80%", "0.388", "67.0%", "80.1%"],
            ["pH", "90%", "0.583", "85.9%", "91.7%"],
            ["pH", "95%", "0.820", "95.9%", "96.8%"],
            ["DO", "80%", "1.832 mg/L", "87.6%", "84.7%"],
            ["DO", "90%", "2.552 mg/L", "95.0%", "90.4%"],
            ["DO", "95%", "3.217 mg/L", "97.5%", "93.6%"],
        ],
        [1550, 1700, 2000, 2000, 2110],
    )
    add_figure(doc, FIG_CONFORMAL, "图 5　低成本 pH 与 DO 的 conformal 名义覆盖率和经验覆盖率。")
    add_paragraph(
        doc,
        "pH 的 80% 名义水平在 pooled 口径下明显欠覆盖，说明误差分布具有强站点异质性或时间聚集。"
        "95% 水平较稳健但区间偏宽。后续可考虑站点条件化、季节条件化或直接训练分位数预测头，以获得更窄、更均衡的区间。"
    )

    # 9
    add_section_heading(doc, "9. 论文可主张的贡献、限制与未完成项", 1)
    add_section_heading(doc, "9.1 可写入论文的主要贡献", 2)
    contributions = [
        "系统区分了常规预测准确度与事件敏感性，并用 all-data / non-event matched 设计量化二者权衡。",
        "证明前期降雨状态（API）比单独当前降雨更有可能提供稳定预测增益。",
        "建立验证集正常残差阈值的异常提前量分析，给出连续告警、单小时敏感性和滞后富集三种互补证据。",
        "构建不使用历史 pH/DO 的真实低成本部署口径，并量化删除昂贵传感器的性能代价。",
        "用仅 130 参数的冻结 FiLM adapter 给出 pH 和 DO 的新站点预热曲线、站点改善比例和保守部署时长。",
        "完成 DO 原生单位审计和 conformal 经验区间，使结果能够被现场人员理解和使用。",
    ]
    add_numbered_list(doc, contributions)

    add_section_heading(doc, "9.2 不能过度主张的内容", 2)
    add_bullet(doc, "不能说降雨对结果“没有用”；不同降雨表示在不同目标上效果不同。")
    add_bullet(doc, "不能说异常残差就是污染事件；现有标签弱，结果是预警潜力而非因果识别。")
    add_bullet(doc, "不能把使用历史目标的 0.07 RMSE 与无历史目标的 0.45 RMSE直接比较并判定后者失败。")
    add_bullet(doc, "不能声称所有站点都能在统一时长达标；pH 有 12/39、DO 有 5/39 个站点在 12 个月内仍未达到指定 MAE 阈值。")
    add_bullet(doc, "不能将 conformal 覆盖解释为严格时序 IID 保证。")

    add_section_heading(doc, "9.3 当前唯一外部阻塞项", 2)
    add_callout(
        doc,
        "待老师提供的数据",
        "地质类型、baseflow index、不透水面比例和许可排放点距离等站点级地质/水文元数据目前不在本地 Archive。"
        "因此无法完成静态元数据消融。数据到位后，可保持现有 train/validation/test split，加入静态特征并做 matched ablation，无需重做全部动态实验。",
        "grey",
    )

    add_section_heading(doc, "9.4 建议的论文 Results 顺序", 2)
    add_numbered_list(doc, (
        "先报告 common1000 数据、站点划分、目标单位和严格无泄漏政策。",
        "报告 LSTM-FiLM 的降雨/API 消融，并把全数据与 non-event 的权衡作为核心结果。",
        "随后报告异常提前量、滞后富集和动态范围检查，强调 weak-label 限制。",
        "将低成本 sensor ablation 作为部署难度基线，再报告 pH/DO 预热曲线和站点异质性。",
        "最后报告 conformal 区间、DO 单位审计、地质数据缺口和未来工作。",
    ))

    # 10
    add_section_heading(doc, "10. 总结与下一步", 1)
    add_paragraph(
        doc,
        "项目目前已经具备一条可以形成论文的结果链：降雨和 API 提供动态环境信息；non-event 训练揭示预测与事件敏感性的权衡；"
        "残差分析给出有限但可量化的提前预警证据；低成本模型把研究转向更真实的新站点部署；现场预热和不确定性区间则把模型输出转化为操作策略。"
    )
    add_paragraph(
        doc,
        "因此下一阶段不应继续无边界地尝试更多相近架构，而应优先完成论文材料整合、统一表格口径、固定主结论和补充地质元数据消融。"
        "若地质数据短期仍未到位，可先完成论文主体，明确将该消融列为待补结果。"
    )
    add_table(
        doc,
        ["优先级", "行动", "完成条件"],
        [
            ["1", "将本报告拆分为论文 Methods / Results / Discussion", "全部数值只来自 frozen final 输出"],
            ["2", "统一主图和表格命名、单位、小数位", "DO 全部使用 mg/L；旧目录不再引用"],
            ["3", "获得地质/水文元数据后做静态消融", "matched cohort + 配对站点 CI"],
            ["4", "准备老师汇报", "10–12 页幻灯片，突出四条主贡献和限制"],
            ["5", "可选增强", "站点自适应预热停止、条件 conformal、事件加权多任务"],
        ],
        [1150, 4100, 4110],
    )

    # Appendix
    add_section_heading(doc, "附录 A　关键证据路径", 1, page_break=True)
    evidence = [
        ("总体完成报告", r"analysis_outputs\common1000_thesis_completion_v1\reports\FINAL_THESIS_COMPLETION_REPORT.md"),
        ("降雨训练模式比较", r"analysis_outputs\ph_common1000_rainfall_training_mode_comparison_v1\FINAL_REPORT.md"),
        ("LSTM 降雨消融", r"analysis_outputs\ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1\full\reports\FINAL_LSTM_RAIN_ABLATION_REPORT.md"),
        ("异常提前量", r"analysis_outputs\ph_common1000_error_onset_final_lstm_rain_ablation_v1\reports\FINAL_REPORT.md"),
        ("低成本核心", r"analysis_outputs\common1000_lowcost_deployment_summary_v1\reports\core_sensor_ablation.csv"),
        ("pH 预热", r"analysis_outputs\ph_common1000_warmup_adaptation_lowcost_test_v1\full\reports\FINAL_WARMUP_TEST_REPORT.md"),
        ("DO 预热（mg/L）", r"analysis_outputs\do_common1000_warmup_adaptation_lowcost_mg_l_test_v1\full\reports\FINAL_WARMUP_TEST_REPORT.md"),
        ("Conformal", r"analysis_outputs\common1000_lowcost_conformal_v1\reports\FINAL_CONFORMAL_REPORT.md"),
    ]
    add_table(doc, ["主题", "相对项目根目录的证据文件"], evidence, [2450, 6910], font_size=8.3)
    add_paragraph(
        doc,
        f"项目根目录：{ROOT}。各报告中的 CSV、JSON、模型 checkpoint、预测文件和站点级图件构成可复核的分析证据。"
    )

    add_section_heading(doc, "附录 B　完成状态清单", 1)
    add_table(
        doc,
        ["工作包", "状态", "备注"],
        [
            ["Archive / DVC 数据获取与降雨字段确认", "完成", "小时降雨 00045 已进入建模数据"],
            ["910 站点统一 cohort 与 split", "完成", "638 / 137 / 135"],
            ["pH baseline、LSTM-FiLM、降雨/API 消融", "完成", "全数据与 non-event 均已 matched"],
            ["异常提前量、滞后富集、动态范围、漏报归因", "完成", "描述性 weak-label 结论"],
            ["低成本 pH/DO 传感器消融", "完成", "无目标历史"],
            ["pH/DO 预热曲线与 78 张站点图", "完成", "39 pH + 39 DO"],
            ["DO 原生 mg/L 单位修正", "完成", "旧结果已弃用"],
            ["Conformal 区间", "完成", "45 校准 / 90 评价站点"],
            ["测试与 QA", "完成", "8 项审计通过；17 项聚焦测试通过"],
            ["地质/水文静态元数据消融", "阻塞", "等待老师提供外部数据"],
        ],
        [3940, 1200, 4220],
        font_size=8.5,
    )

    add_section_heading(doc, "附录 C　缩写", 1)
    add_table(
        doc,
        ["缩写", "含义"],
        [
            ["API", "Antecedent Precipitation Index，前期降雨指数"],
            ["DO", "Dissolved Oxygen，溶解氧；本文报告单位为 mg/L"],
            ["FiLM", "Feature-wise Linear Modulation，特征级线性调制"],
            ["LSTM", "Long Short-Term Memory，长短期记忆网络"],
            ["RMSE", "Root Mean Squared Error，均方根误差"],
            ["MAE", "Mean Absolute Error，平均绝对误差"],
            ["CI", "Confidence Interval，置信区间"],
            ["DVC", "Data Version Control，用于数据版本与远程大文件管理"],
        ],
        [1900, 7460],
    )

    # Footer closing note.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("— End of report —")
    set_run_font(r, size=8.5, color=TOKENS["Mid_Grey"], italic=True)

    doc.save(OUT_PATH)
    print("Report saved successfully.")


if __name__ == "__main__":
    build_report()
