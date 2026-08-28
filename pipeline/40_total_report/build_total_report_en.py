from pathlib import Path
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_total_report as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_outputs" / "total_report_2026-07-30"
OUT_PATH = OUT_DIR / "Pollution_ID_Consolidated_Research_Report_EN_2026-07-30.docx"
WORKFLOW_FIG = OUT_DIR / "study_design_overview_en.png"

FIG_WARMUP = ROOT / "analysis_outputs" / "common1000_lowcost_deployment_summary_v1" / "figures" / "lowcost_warmup_pH_DO.png"
FIG_ONSET = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "event_lead_time_distribution.png"
FIG_LAG = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "lagged_event_enrichment.png"
FIG_COMPRESSION = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "compression_01645762.png"
FIG_CONFORMAL = ROOT / "analysis_outputs" / "common1000_lowcost_conformal_v1" / "figures" / "conformal_coverage.png"


# Resolved design system:
# preset: narrative_proposal
# header pattern: memo_masthead
# named overrides: TechnicalTableHeader (dark text on F4F6F9);
# CoverMetricStrip (light blue fill); WorkflowFigure (Pillow-rendered schematic).


def load_font(size, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def rounded_box(draw, xy, fill, outline, title, body, title_font, body_font):
    draw.rounded_rectangle(xy, radius=26, fill=fill, outline=outline, width=4)
    x1, y1, x2, y2 = xy
    draw.text((x1 + 28, y1 + 22), title, font=title_font, fill="#203864")
    y = y1 + 78
    for line in body:
        draw.text((x1 + 28, y), line, font=body_font, fill="#1F2937")
        y += 38


def draw_arrow(draw, start, end, color="#2E74B5"):
    draw.line([start, end], fill=color, width=8)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - direction * 28, ey - 18), (ex - direction * 28, ey + 18)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 18, ey - direction * 28), (ex + 18, ey - direction * 28)]
    draw.polygon(points, fill=color)


def create_workflow_figure():
    width, height = 2400, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    box_title = load_font(31, bold=True)
    body_font = load_font(26)
    small_font = load_font(24)

    draw.text((90, 55), "Study design and analytical sequence", font=title_font, fill="#203864")
    draw.text(
        (90, 112),
        "All reported results use fixed site splits and final retained outputs.",
        font=small_font,
        fill="#6B7280",
    )

    rounded_box(
        draw,
        (90, 235, 520, 610),
        "#EAF2F8",
        "#2E74B5",
        "Data",
        ["Archive / common1000", "910 sites", "Hourly water quality", "Rainfall and flow"],
        box_title,
        body_font,
    )
    rounded_box(
        draw,
        (675, 235, 1105, 610),
        "#F4F6F9",
        "#6B7280",
        "Site split",
        ["Training: 638", "Validation: 137", "Test: 135", "No site overlap"],
        box_title,
        body_font,
    )
    rounded_box(
        draw,
        (1260, 205, 1750, 485),
        "#EAF2F8",
        "#2E74B5",
        "Dynamic prediction",
        ["LSTM-FiLM", "Rainfall and API", "All-data / non-event"],
        box_title,
        body_font,
    )
    rounded_box(
        draw,
        (1835, 205, 2310, 485),
        "#F4F6F9",
        "#6B7280",
        "Event analysis",
        ["Residual thresholds", "Lead time", "Lag enrichment"],
        box_title,
        body_font,
    )
    rounded_box(
        draw,
        (1260, 640, 1750, 930),
        "#EAF2F8",
        "#2E74B5",
        "Low-cost deployment",
        ["No pH/DO history", "Sensor ablation", "Site warm-up adapter"],
        box_title,
        body_font,
    )
    rounded_box(
        draw,
        (1835, 640, 2310, 930),
        "#F4F6F9",
        "#6B7280",
        "Risk and reporting",
        ["Conformal intervals", "Unit audit", "Deployment guidance"],
        box_title,
        body_font,
    )
    draw_arrow(draw, (520, 423), (675, 423))
    draw_arrow(draw, (1105, 423), (1260, 345))
    draw_arrow(draw, (1750, 345), (1835, 345))
    draw_arrow(draw, (1105, 500), (1260, 785))
    draw_arrow(draw, (1750, 785), (1835, 785))

    image.save(WORKFLOW_FIG, quality=95)


def patch_list_numbering(doc, style_name, left=540, hanging=279, after=80, line=290):
    style = doc.styles[style_name]
    style.paragraph_format.left_indent = Inches(0.375)
    style.paragraph_format.first_line_indent = Inches(-0.194)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.line_spacing = 1.208

    num_id = int(style._element.pPr.numPr.numId.val)
    numbering = doc.part.numbering_part.element
    num_nodes = numbering.xpath(f"./w:num[@w:numId='{num_id}']")
    if not num_nodes:
        return
    abstract_id = num_nodes[0].find(qn("w:abstractNumId")).get(qn("w:val"))
    abstract_nodes = numbering.xpath(f"./w:abstractNum[@w:abstractNumId='{abstract_id}']")
    if not abstract_nodes:
        return
    levels = [
        node
        for node in abstract_nodes[0].findall(qn("w:lvl"))
        if node.get(qn("w:ilvl")) == "0"
    ]
    if not levels:
        return
    level = levels[0]
    p_pr = level.find(qn("w:pPr"))
    if p_pr is None:
        p_pr = OxmlElement("w:pPr")
        level.append(p_pr)
    tabs = p_pr.find(qn("w:tabs"))
    if tabs is None:
        tabs = OxmlElement("w:tabs")
        p_pr.append(tabs)
    for child in list(tabs):
        tabs.remove(child)
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), str(left))
    tabs.append(tab)
    ind = p_pr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        p_pr.append(ind)
    ind.set(qn("w:left"), str(left))
    ind.set(qn("w:hanging"), str(hanging))
    spacing = p_pr.find(qn("w:spacing"))
    if spacing is None:
        spacing = OxmlElement("w:spacing")
        p_pr.append(spacing)
    spacing.set(qn("w:after"), str(after))
    spacing.set(qn("w:line"), str(line))
    spacing.set(qn("w:lineRule"), "auto")


def configure_styles_en(doc):
    base.configure_styles(doc)
    normal = doc.styles["Normal"]
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.333

    title = doc.styles["Title"]
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(8)

    subtitle = doc.styles["Subtitle"]
    subtitle.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    subtitle.paragraph_format.space_after = Pt(18)

    patch_list_numbering(doc, "List Bullet")
    patch_list_numbering(doc, "List Number")


def configure_page_en(doc):
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True
    base.set_repeat_header_text(
        section.header.paragraphs[0],
        "Pollution_ID | Consolidated research report",
        color="6B7280",
    )
    base.add_page_field(section.footer.paragraphs[0])


def add_table_en(doc, headers, rows, widths, font_size=9):
    table = base.add_table(
        doc,
        headers,
        rows,
        widths,
        header_fill="F4F6F9",
        font_size=font_size,
    )
    for cell in table.rows[0].cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                base.set_run_font(run, size=font_size, bold=True, color="203864")
    for row in table.rows[1:]:
        for cell in row.cells:
            base.set_cell_shading(cell, "FFFFFF")
    return table


def add_figure_en(doc, path, caption, width=6.15):
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    shape = run.add_picture(str(path), width=Inches(width))
    inline = shape._inline
    doc_pr = inline.docPr
    doc_pr.set("descr", caption)
    cap = doc.add_paragraph(style="Caption")
    cap_run = cap.add_run(caption)
    base.set_run_font(cap_run, size=9, color="6B7280", italic=True)


def add_source_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    base.set_run_font(r, size=8.5, color="6B7280", italic=True)


def add_bullets(doc, items):
    for item in items:
        base.add_bullet(doc, item)


def build_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    create_workflow_figure()
    for figure in (WORKFLOW_FIG, FIG_WARMUP, FIG_ONSET, FIG_LAG, FIG_COMPRESSION, FIG_CONFORMAL):
        if not figure.exists():
            raise FileNotFoundError(figure)

    doc = Document()
    configure_styles_en(doc)
    configure_page_en(doc)
    doc.core_properties.title = "Cross-site water-quality forecasting and low-cost deployment"
    doc.core_properties.subject = "Rainfall effects, residual event warning, and site adaptation for pH and dissolved oxygen"
    doc.core_properties.author = "Pollution_ID Project"
    doc.core_properties.keywords = "pH, dissolved oxygen, rainfall, API, LSTM-FiLM, site adaptation"

    # Memo-masthead cover
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(70)
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_after = Pt(15)
    kr = kicker.add_run("POLLUTION_ID  |  CONSOLIDATED RESEARCH REPORT")
    base.set_run_font(kr, size=10, bold=True, color="2E74B5")
    base.add_top_rule(kicker, color="2E74B5", size="16")

    title = doc.add_paragraph(style="Title")
    title_run = title.add_run("Cross-site water-quality forecasting\nand low-cost deployment")
    base.set_run_font(title_run, size=28, bold=True, color="203864")

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle_run = subtitle.add_run(
        "Rainfall effects, residual-based event warning, and site adaptation for pH and dissolved oxygen"
    )
    base.set_run_font(subtitle_run, size=14, color="2E74B5", italic=True)

    metadata = [
        ("Report", "Final retained results"),
        ("Data", "Supervisor-provided Archive/common1000 cohort"),
        ("Date", "30 July 2026"),
        ("Version", "Final synthesis v1"),
    ]
    for label, value in metadata:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r1 = p.add_run(f"{label}: ")
        base.set_run_font(r1, size=10.5, bold=True, color="203864")
        r2 = p.add_run(value)
        base.set_run_font(r2, size=10.5, color="6B7280")

    doc.add_paragraph().paragraph_format.space_after = Pt(62)
    status = doc.add_paragraph()
    sr = status.add_run(
        "Status: all analyses that can be completed with the local data are final. "
        "The geology/hydrology ablation remains pending because the required metadata have not yet been supplied."
    )
    base.set_run_font(sr, size=10.2, bold=True, color="203864")
    doc.add_page_break()

    # Abstract
    base.add_section_heading(doc, "Abstract", 1)
    base.add_paragraph(
        doc,
        "This study evaluates cross-site forecasting of pH and dissolved oxygen (DO) under two distinct settings. "
        "The first uses historical target observations to estimate the predictive value of rainfall and antecedent wetness. "
        "The second removes historical pH and DO, retaining only low-cost sensor inputs, and asks how much site-specific "
        "warm-up data are required at a new monitoring location. A frozen LSTM-FiLM model provided the final dynamic "
        "specification. Residual thresholds, calibrated only on validation-site normal periods, were used to examine "
        "whether model error increased before registered events."
    )
    base.add_paragraph(
        doc,
        "Rainfall produced small but reproducible gains in the frozen LSTM analysis. In a matched all-data comparison, "
        "rainfall plus an antecedent precipitation index (API) reduced pH RMSE by 9.26%. Non-event training increased "
        "event contrast but generally worsened overall prediction, indicating a trade-off between calibration and event "
        "sensitivity. Under a three-hour persistence rule, 20.3% of registered events were preceded by an alert, with a "
        "median lead time of 8 h. For deployment without target history, a 130-parameter site adapter reduced pH macro "
        "MAE to 0.184 after 12 months and DO macro MAE to 0.688 mg/L after 3 months. These results support a deployment "
        "strategy in which environmental drivers and site adaptation address different sources of cross-site error."
    )
    base.add_metric_strip(
        doc,
        [
            ("-9.26%", "pH RMSE with rainfall + API"),
            ("0.184", "pH macro MAE after 12 months"),
            ("0.688 mg/L", "DO macro MAE after 3 months"),
        ],
    )
    base.set_repeat_table_header(doc.tables[-1].rows[0])
    base.add_callout(
        doc,
        "Central interpretation",
        "Rainfall improves representation of short-term environmental forcing. Site warm-up corrects persistent "
        "differences between locations. The two mechanisms are complementary and should not be evaluated as substitutes.",
        "navy",
    )

    base.add_section_heading(doc, "Report structure", 2)
    contents = [
        "Research questions and analytical framework",
        "Data and experimental design",
        "Models and evaluation",
        "Rainfall, API, and non-event training",
        "Residual-based event warning",
        "Low-cost deployment and site warm-up",
        "DO unit audit",
        "Predictive uncertainty",
        "Discussion, limitations, and conclusions",
        "Appendices: evidence files and completion status",
    ]
    for item in contents:
        base.add_paragraph(doc, item, style="List Number")

    # 1
    base.add_section_heading(doc, "1. Research questions and analytical framework", 1)
    base.add_paragraph(
        doc,
        "The project began as a comparison of forecasting models, but the final analysis addresses three connected "
        "questions. First, does rainfall provide predictive information that is not already contained in the water-quality "
        "history? Second, do residuals become abnormal before registered pollution or disturbance events? Third, can a "
        "model be deployed at a new site when historical pH or DO is unavailable?"
    )
    add_table_en(
        doc,
        ["Analysis", "Research question", "Final design"],
        [
            ["Dynamic forecasting", "Do rainfall and API improve cross-site pH prediction?", "Frozen LSTM-FiLM ablation; matched all-data and non-event training"],
            ["Event warning", "How early do abnormal residuals appear?", "Validation-site thresholds; three-hour persistence; lag analysis"],
            ["Deployment", "How much local calibration is required without target history?", "Low-cost inputs; frozen site adapter; 2 weeks to 12 months"],
            ["Uncertainty", "How should point errors be translated into operational ranges?", "Site-separated split conformal calibration"],
        ],
        [1700, 3400, 4260],
    )
    base.add_callout(
        doc,
        "Two model settings",
        "The historical-target model is a forecasting benchmark and reaches RMSE values near 0.07. The low-cost model "
        "deliberately excludes target history and cross-target sensors. Its larger error reflects a more difficult and "
        "more realistic deployment task; the two error scales are not directly comparable.",
        "blue",
    )
    add_figure_en(
        doc,
        WORKFLOW_FIG,
        "Figure 1. Study design and analytical sequence. Site splits were fixed before model comparison, event analysis, and deployment evaluation.",
        width=6.25,
    )

    base.add_section_heading(doc, "1.1 Retained contributions", 2)
    add_bullets(
        doc,
        [
            "A matched analysis separates predictive accuracy from event contrast under all-data and non-event training.",
            "API is evaluated as a compact representation of antecedent catchment wetness rather than current rainfall alone.",
            "Event-warning evidence is reported with persistence and single-hour definitions, lead-time distributions, and lagged enrichment.",
            "A no-target-history deployment model quantifies the cost of removing expensive sensors and the benefit of site adaptation.",
            "DO results are reported in native mg/L after a complete unit audit, and empirical prediction intervals are evaluated on held-out sites.",
        ],
    )

    # 2
    base.add_section_heading(doc, "2. Data and experimental design", 1)
    base.add_paragraph(
        doc,
        "The analysis uses the supervisor-provided Archive/common1000 data managed through DVC. DVC stores file versions "
        "and remote pointers in the repository while the large data objects remain in external storage. The local archive "
        "contains water temperature, conductivity, turbidity, flow, water-quality targets, and hourly rainfall."
    )
    add_table_en(
        doc,
        ["Cohort", "Sites", "Role"],
        [
            ["Primary common1000 cohort", "910", "Common basis for the retained experiments"],
            ["Training", "638", "Model estimation"],
            ["Validation", "137", "Hyperparameter, threshold, and warm-up decisions"],
            ["Test", "135", "Final held-out reporting"],
            ["Strict 12-month warm-up test cohort", "39 pH; 39 DO", "Complete warm-up and fixed post-warm-up evaluation windows"],
        ],
        [2950, 1600, 4810],
    )
    base.add_paragraph(
        doc,
        "The warm-up cohort is smaller because eligibility requires sufficient low-cost inputs, a complete calibration "
        "window, observed targets, and a common evaluation period after the longest warm-up duration. This matched cohort "
        "is a design constraint, not evidence that only 39 sites are available."
    )

    base.add_section_heading(doc, "2.1 Low-cost input definition", 2)
    add_table_en(
        doc,
        ["Variable", "Code", "Low-cost core", "Rationale"],
        [
            ["Water temperature", "00010", "Yes", "Routine continuous measurement"],
            ["Specific conductance", "00095", "Yes", "Proxy for ionic and dissolved constituents"],
            ["Turbidity", "63680", "Yes", "Response to particles and short-term disturbance"],
            ["Hourly rainfall", "00045", "Yes", "External hydrological forcing"],
            ["Historical pH", "00400", "No", "Excluded to prevent target-history dependence"],
            ["Historical DO", "00300", "No", "Excluded to prevent target-history dependence"],
        ],
        [2200, 1150, 1800, 4210],
    )
    base.add_paragraph(
        doc,
        "The full-sensor pH comparator additionally uses DO, and the full-sensor DO comparator additionally uses pH. "
        "The low-cost core removes these cross-target measurements as well as the historical target itself."
    )

    base.add_section_heading(doc, "2.2 Leakage controls", 2)
    add_bullets(
        doc,
        [
            "Sites, rather than rows, were assigned to training, validation, and test sets.",
            "Warm-up duration was selected on validation sites before the test analysis.",
            "All warm-up durations were scored on the same post-12-month evaluation keys.",
            "Event thresholds were estimated from validation-site normal periods only.",
            "DO outputs were converted back to native mg/L before interpretation.",
        ],
    )

    # 3
    base.add_section_heading(doc, "3. Models and evaluation", 1, page_break=True)
    base.add_section_heading(doc, "3.1 Dynamic model", 2)
    base.add_paragraph(
        doc,
        "The final forecasting model is an LSTM-FiLM network. The LSTM represents a 48 h history of dynamic covariates, "
        "while feature-wise linear modulation allows the hidden representation to vary by site. The frozen final "
        "configuration uses 64 hidden units, dropout of 0.1, and a learning rate of 0.0003. GRU experiments were treated "
        "as secondary architecture checks; the final analysis focuses on LSTM-FiLM in line with the supervisory direction."
    )
    base.add_section_heading(doc, "3.2 Site adaptation", 2)
    base.add_paragraph(
        doc,
        "For a new site, the recurrent backbone is frozen and only 130 parameters are estimated: 128 site-specific FiLM "
        "offsets and a positive affine scale and bias. This design targets stable site-level differences without retraining "
        "the full network. Zero-shot prediction, the frozen adapter, and full fine-tuning are compared at each warm-up duration."
    )
    base.add_section_heading(doc, "3.3 Outcomes and statistical reporting", 2)
    add_table_en(
        doc,
        ["Measure", "Definition", "Use"],
        [
            ["RMSE", "Square root of mean squared error", "Primary forecasting metric; sensitive to large errors"],
            ["MAE", "Mean absolute error", "Primary deployment metric"],
            ["Macro average", "Metric calculated by site and then averaged", "Prevents large sites from dominating"],
            ["Tolerance coverage", "Fraction within a prespecified absolute error", "pH +/-0.2; DO +/-1 mg/L"],
            ["Event enrichment", "Event-period abnormality relative to normal periods", "Event sensitivity, distinct from RMSE"],
            ["Paired site difference", "Within-site difference between two models", "Confidence intervals and direction of change"],
        ],
        [1800, 3450, 4110],
    )

    # 4
    base.add_section_heading(doc, "4. Rainfall, API, and non-event training", 1)
    base.add_section_heading(doc, "4.1 Frozen LSTM rainfall ablation", 2)
    add_table_en(
        doc,
        ["Specification", "Test RMSE", "Relative change", "Sites improved"],
        [
            ["LSTM-FiLM baseline", "0.07091", "-", "-"],
            ["LSTM-FiLM + current rainfall", "0.06975", "-1.62%", "101 / 135"],
        ],
        [3300, 1900, 2060, 2100],
    )
    base.add_paragraph(
        doc,
        "The paired site-level RMSE difference had a 95% confidence interval of -0.00144 to -0.00046. Event-period RMSE "
        "fell by 1.21%, and RMSE during the preceding 6 h wet window fell by 1.16%. Current rainfall therefore provides a "
        "small but consistent gain under the frozen specification."
    )

    base.add_section_heading(doc, "4.2 Matched comparison of rainfall representation and training distribution", 2)
    add_table_en(
        doc,
        ["Training / input", "RMSE", "Event enrichment", "Interpretation"],
        [
            ["All data / baseline", "0.09056", "1.698", "Prediction reference"],
            ["All data / current rainfall", "0.09059", "2.161", "Higher event contrast; unchanged RMSE"],
            ["All data / rainfall + API", "0.08217", "1.462", "Best RMSE; 9.26% below baseline"],
            ["Non-event / baseline", "0.14867", "2.285", "Higher contrast; poorer calibration"],
            ["Non-event / current rainfall", "0.15015", "3.048", "Highest contrast; poorest RMSE"],
            ["Non-event / rainfall + API", "0.14621", "1.964", "Modest gain within non-event training"],
        ],
        [3150, 1350, 1900, 2960],
    )
    base.add_paragraph(
        doc,
        "Under all-data training, rainfall plus API reduced RMSE by 9.26% relative to the matched baseline. Ninety-six of "
        "135 test sites improved, and the paired confidence interval was -0.00960 to -0.00498. API accumulates earlier "
        "rainfall with temporal decay and therefore represents antecedent wetness rather than rainfall at a single hour."
    )
    base.add_callout(
        doc,
        "Training-distribution trade-off",
        "Non-event training makes registered events appear more unusual, but it does not necessarily improve ordinary "
        "forecasting. Accuracy and event contrast should be reported as separate objectives.",
        "blue",
    )
    base.add_paragraph(
        doc,
        "An earlier supervisor-highlighted run increased pH event enrichment from 3.66 to 4.65 with little change in RMSE. "
        "Because that run used a different cohort or configuration, it is retained as directional evidence rather than "
        "combined numerically with the final matched comparison."
    )

    # 5
    base.add_section_heading(doc, "5. Residual-based event warning", 1, page_break=True)
    base.add_paragraph(
        doc,
        "Site-specific residual thresholds were estimated from normal periods at validation sites and applied to test "
        "events. The primary rule required the threshold to be exceeded for three consecutive hours. This persistence "
        "condition limits isolated false alarms. Event records are treated as weak labels because the recorded time may "
        "lag the physical disturbance and some disturbances may be unregistered."
    )
    add_table_en(
        doc,
        ["Definition", "Events alerted", "Rate", "Lead time"],
        [
            ["Primary: three consecutive hours", "579 / 2,853", "20.29%", "Median 8 h; IQR 4-17 h"],
            ["Among analysable events", "579 / 2,577", "22.47%", "Same"],
            ["Sensitivity: one abnormal hour", "1,339 / 2,853", "46.93%", "Median 4 h"],
        ],
        [3150, 2050, 1500, 2660],
    )
    base.add_paragraph(
        doc,
        "Using all 2,853 events as the denominator, 12.58% were alerted at least 6 h in advance, 7.75% at least 12 h in "
        "advance, and 3.15% at least 24 h in advance. The residual signal has practical lead time in a subset of events, "
        "but its sensitivity is too low to support a claim of general-purpose event detection."
    )
    add_figure_en(
        doc,
        FIG_ONSET,
        "Figure 2. Distribution of lead time among registered events with a pre-event alert. The primary definition requires three consecutive abnormal hours.",
    )

    base.add_section_heading(doc, "5.1 Lagged enrichment", 2)
    base.add_paragraph(
        doc,
        "Event labels were shifted forward to test whether residual enrichment was already present before the recorded "
        "event time. For the rainfall model, enrichment declined from 1.733 at lead 0 to 1.554 at lead 12 h and was 1.618 "
        "at lead 24 h. The pattern is consistent with early disturbance in some events, but it does not establish causality."
    )
    add_figure_en(
        doc,
        FIG_LAG,
        "Figure 3. Residual enrichment after shifting event labels forward in time. Enrichment remains above one in the pre-event window but generally weakens with lead time.",
    )

    base.add_section_heading(doc, "5.2 Dynamic-range compression", 2)
    compression_intro = base.add_paragraph(
        doc,
        "A model can obtain low average error while attenuating high and low extremes. At representative site 01645762, "
        "the rainfall model had an overall slope of 0.983, lower-tail slope of 1.008, upper-tail slope of 0.981, and a "
        "predicted-to-observed range ratio of 1.039. There was no evidence of severe range compression at this site."
    )
    compression_intro.paragraph_format.keep_with_next = True
    add_figure_en(
        doc,
        FIG_COMPRESSION,
        "Figure 4. Prediction-observation range check at representative held-out site 01645762.",
    )

    base.add_section_heading(doc, "5.3 Descriptive analysis of missed events", 2)
    base.add_paragraph(
        doc,
        "Missed-event attribution covered 17 sites and 802,108 rows. In the rainfall model, fitted coefficients for "
        "6 h rainfall, turbidity, and flow were approximately 0.00667, 0.0626, and 5.56 x 10^-7. These estimates are "
        "descriptive. Site structure, label quality, and unobserved discharges prevent a causal interpretation."
    )

    # 6
    base.add_section_heading(doc, "6. Low-cost deployment and site warm-up", 1)
    base.add_section_heading(doc, "6.1 Performance without target history", 2)
    add_table_en(
        doc,
        ["Target", "Input set", "Pooled RMSE", "Macro RMSE", "Macro MAE", "Within tolerance"],
        [
            ["pH", "Full sensors", "0.429", "0.314", "0.243", "47.25% (+/-0.2)"],
            ["pH", "Low-cost core", "0.446", "0.340", "0.264", "44.58% (+/-0.2)"],
            ["DO", "Full sensors", "1.333", "1.125", "0.946", "67.72% (+/-1 mg/L)"],
            ["DO", "Low-cost core", "1.386", "1.213", "1.018", "65.22% (+/-1 mg/L)"],
        ],
        [1100, 1750, 1500, 1500, 1500, 2010],
    )
    lowcost_intro = base.add_paragraph(
        doc,
        "Removing cross-target sensors produced a measurable but moderate loss of accuracy. The mean site-level increase "
        "in RMSE was 0.0252 for pH (95% CI 0.0076 to 0.0436) and 0.0878 mg/L for DO (95% CI -0.0037 to 0.1759). "
        "The low-cost core is therefore a viable starting point, but site calibration remains important."
    )
    lowcost_intro.paragraph_format.keep_together = True

    base.add_section_heading(doc, "6.2 pH warm-up", 2)
    add_table_en(
        doc,
        ["Method at 12 months", "Pooled RMSE", "Macro RMSE", "Macro MAE", "Within +/-0.2"],
        [
            ["Zero-shot", "0.3867", "0.3581", "0.3041", "-"],
            ["FiLM + affine", "0.2529", "0.2381", "0.1843", "65.82%"],
            ["Full fine-tuning", "0.2429", "-", "0.1808", "67.87%"],
        ],
        [2800, 1640, 1640, 1640, 1640],
    )
    base.add_paragraph(
        doc,
        "The 12-month adapter reduced pooled RMSE by 34.61%, macro RMSE by 33.51%, and macro MAE by 39.38% relative to "
        "zero-shot prediction. Thirty-four of 39 sites improved. The paired mean macro-RMSE difference was -0.1200, with "
        "an approximate confidence interval of -0.1696 to -0.0768. Full fine-tuning was only slightly better."
    )

    base.add_section_heading(doc, "6.3 DO warm-up", 2)
    add_table_en(
        doc,
        ["Method at 12 months", "Pooled RMSE", "Macro RMSE", "Macro MAE", "Within +/-1 mg/L"],
        [
            ["Zero-shot", "1.3306", "1.1690", "0.9604", "-"],
            ["FiLM + affine", "0.9842", "0.8225", "0.6134", "79.89%"],
            ["Full fine-tuning", "0.9301", "-", "0.5400", "82.47%"],
        ],
        [2800, 1640, 1640, 1640, 1640],
    )
    base.add_paragraph(
        doc,
        "At 12 months, the adapter reduced pooled RMSE by 26.03%, macro RMSE by 29.64%, and macro MAE by 36.12%. "
        "Thirty-four of 39 sites improved, with a paired mean macro-RMSE difference of -0.3465 mg/L and an approximate "
        "confidence interval of -0.5113 to -0.2050. Unlike pH, full fine-tuning remained materially better for DO."
    )
    add_figure_en(
        doc,
        FIG_WARMUP,
        "Figure 5. Low-cost pH and DO performance across site warm-up durations. Every duration is evaluated on the same post-12-month keys.",
    )

    base.add_section_heading(doc, "6.4 Recommended warm-up periods", 2)
    add_table_en(
        doc,
        ["Target", "Conservative period", "Test performance", "Basis"],
        [
            ["pH", "12 months", "Macro MAE 0.184; 34/39 sites improved", "Validation-frozen choice; continued test improvement"],
            ["DO", "3 months", "Macro MAE 0.688 mg/L; 77.15% within +/-1", "Shortest period supported by validation and test results"],
            ["DO, if resources permit", "12 months", "Macro MAE 0.613 mg/L; 79.89% within +/-1", "Additional data still improve accuracy"],
        ],
        [1300, 1700, 3100, 3260],
    )
    base.add_paragraph(
        doc,
        "Warm-up is an initial commissioning period, not a delay required for every prediction. A site can be operated "
        "earlier with wider uncertainty bounds and an adapter that is updated as observations accumulate."
    )

    base.add_section_heading(doc, "6.5 Site heterogeneity", 2)
    add_table_en(
        doc,
        ["Target criterion", "Earliest duration among sites that reached the criterion", "Not reached"],
        [
            ["pH: site MAE <= 0.2", "2 wk 14; 1 mo 1; 2 mo 1; 3 mo 3; 6 mo 4; 9 mo 2; 12 mo 2", "12 / 39"],
            ["DO: site MAE <= 1 mg/L", "2 wk 25; 2 mo 1; 3 mo 5; 6 mo 2; 12 mo 1", "5 / 39"],
        ],
        [2350, 5250, 1760],
    )
    base.add_paragraph(
        doc,
        "A single commissioning period is inefficient. Some sites meet the operational threshold after two weeks, whereas "
        "others remain above it after 12 months. A practical extension is an adaptive stopping rule based on stable "
        "validation error and calibrated uncertainty."
    )

    # 7
    base.add_section_heading(doc, "7. Dissolved-oxygen unit audit", 1)
    base.add_paragraph(
        doc,
        "The data preparation converts Environment Agency DO saturation percentages to mg/L and then applies a fixed "
        "scaling for model input. An early experiment mistakenly treated x00300/20 as the reported target. That output "
        "was not in native mg/L. The error has been marked as deprecated, and all predictions, metrics, plots, and reports "
        "have been regenerated using 20 x x00300 as the reporting target."
    )
    add_table_en(
        doc,
        ["Audit item", "Verified result"],
        [
            ["Reporting target", "20 x x00300, in mg/L"],
            ["Training mean", "8.633 mg/L"],
            ["Training standard deviation", "2.885 mg/L"],
            ["Model weights", "Standardised labels are equivalent; checkpoint SHA values are unchanged"],
            ["Regenerated outputs", "Predictions, error metrics, tolerance coverage, warm-up curves, and summary reports"],
            ["Deprecated outputs", "Earlier DO sensor-ablation and warm-up directories contain DEPRECATED_UNIT_ERROR.md"],
        ],
        [2700, 6660],
    )
    base.add_callout(
        doc,
        "Reporting rule",
        "Only directories marked mg_l should be cited in the thesis, figures, or presentation. The unit correction did not "
        "change optimisation in standardised space, but it changed every interpretable absolute error.",
        "navy",
    )

    # 8
    base.add_section_heading(doc, "8. Predictive uncertainty", 1)
    base.add_paragraph(
        doc,
        "Split conformal intervals were calibrated on 45 sites selected by a fixed hash and evaluated on 90 different "
        "sites. The site-separated design avoids reusing calibration locations. Because observations within a site are "
        "serially dependent, coverage is reported as empirical marginal coverage rather than a strict IID guarantee."
    )
    add_table_en(
        doc,
        ["Target", "Nominal level", "Half-width", "Pooled coverage", "Macro-site coverage"],
        [
            ["pH", "80%", "0.388", "67.0%", "80.1%"],
            ["pH", "90%", "0.583", "85.9%", "91.7%"],
            ["pH", "95%", "0.820", "95.9%", "96.8%"],
            ["DO", "80%", "1.832 mg/L", "87.6%", "84.7%"],
            ["DO", "90%", "2.552 mg/L", "95.0%", "90.4%"],
            ["DO", "95%", "3.217 mg/L", "97.5%", "93.6%"],
        ],
        [1450, 1750, 1950, 2100, 2110],
    )
    add_figure_en(
        doc,
        FIG_CONFORMAL,
        "Figure 6. Nominal and empirical conformal coverage for low-cost pH and DO models.",
    )
    base.add_paragraph(
        doc,
        "The pooled pH interval under-covers at the 80% level, indicating site heterogeneity or temporal clustering in the "
        "residual distribution. Coverage is reliable at 95%, but the interval is broad. Site- or season-conditional "
        "calibration may produce narrower and more even intervals."
    )

    # 9
    base.add_section_heading(doc, "9. Discussion", 1)
    base.add_section_heading(doc, "9.1 Interpretation of the rainfall results", 2)
    base.add_paragraph(
        doc,
        "The rainfall analyses reject a simple conclusion that precipitation is either essential or irrelevant. Current "
        "rainfall has a small stable effect under the frozen model, while API produces a larger gain in the matched "
        "all-data comparison. This distinction is physically plausible: API summarises catchment memory, which cannot be "
        "represented by a single hourly rainfall measurement."
    )
    base.add_paragraph(
        doc,
        "Non-event training alters the objective rather than merely improving the same model. By learning normal dynamics "
        "without registered events, the model makes event windows more anomalous. The resulting increase in enrichment "
        "can be accompanied by worse overall calibration. A thesis should therefore present RMSE and event enrichment as "
        "separate outcomes."
    )

    base.add_section_heading(doc, "9.2 Interpretation of the deployment results", 2)
    base.add_paragraph(
        doc,
        "The low-cost experiments isolate the practical cost of deploying a model without historical pH or DO. The zero-shot "
        "model retains useful cross-site structure, and a small site adapter recovers a substantial share of the lost "
        "accuracy. For pH, the adapter performs close to full fine-tuning; for DO, full fine-tuning remains better, suggesting "
        "that DO differences involve more than a site-specific scale and offset."
    )

    base.add_section_heading(doc, "9.3 Claims supported by the evidence", 2)
    add_bullets(
        doc,
        [
            "Rainfall contains predictive information for pH, particularly when antecedent wetness is represented.",
            "Prediction accuracy and event contrast are distinct objectives and respond differently to non-event training.",
            "Residuals provide useful lead time for a subset of registered events, but current sensitivity is limited.",
            "Low-cost deployment without target history is feasible, and site adaptation materially improves both pH and DO.",
            "Site heterogeneity requires site-aware commissioning and uncertainty reporting.",
        ],
    )

    base.add_section_heading(doc, "9.4 Claims not supported", 2)
    add_bullets(
        doc,
        [
            "The analysis does not show that rainfall is unimportant.",
            "An abnormal residual is not equivalent to a confirmed pollution event.",
            "The historical-target RMSE near 0.07 cannot be compared directly with the no-target-history RMSE near 0.45.",
            "A single warm-up duration is not sufficient for every site.",
            "The conformal results do not provide a strict guarantee under arbitrary temporal dependence.",
        ],
    )

    base.add_section_heading(doc, "9.5 Remaining external dependency", 2)
    base.add_callout(
        doc,
        "Pending static-metadata ablation",
        "Geology class, baseflow index, impervious area, and distance to consented discharge points are not present in the "
        "local Archive. Once supplied, they can be added to the existing fixed site split and evaluated through a matched "
        "ablation. The dynamic experiments do not need to be repeated.",
        "grey",
    )

    # 10
    base.add_section_heading(doc, "10. Conclusions and recommended reporting order", 1, page_break=True)
    base.add_paragraph(
        doc,
        "The retained results form a coherent progression from model performance to deployment. Rainfall and API improve "
        "the representation of short-term forcing; non-event training reveals a trade-off between calibration and event "
        "contrast; residuals provide limited but measurable warning before some registered events; and low-cost site "
        "adaptation converts a cross-site model into a practical commissioning strategy. Unit auditing and conformal "
        "intervals make the final outputs interpretable outside the modelling pipeline."
    )
    add_table_en(
        doc,
        ["Order", "Thesis results section", "Purpose"],
        [
            ["1", "Data, site split, target units, and leakage controls", "Define the common evaluation basis"],
            ["2", "LSTM-FiLM rainfall and API ablation", "Establish predictive effects"],
            ["3", "All-data versus non-event training", "Separate accuracy from event sensitivity"],
            ["4", "Residual lead time and lagged enrichment", "Assess warning potential and limitations"],
            ["5", "Low-cost sensor ablation and warm-up", "Move from forecasting to deployment"],
            ["6", "Conformal intervals and unit audit", "Address uncertainty and interpretability"],
            ["7", "Static-metadata ablation when data arrive", "Test the remaining cross-site hypothesis"],
        ],
        [950, 4400, 4010],
    )
    base.add_paragraph(
        doc,
        "Further architecture search is unlikely to add as much value as consistent reporting, matched comparisons, and "
        "completion of the static-metadata analysis. The next practical step is to convert the retained results into the "
        "thesis Methods, Results, and Discussion chapters using the fixed values in this report."
    )

    # Appendices
    base.add_section_heading(doc, "Appendix A. Evidence files", 1, page_break=True)
    evidence = [
        ("Master completion report", r"analysis_outputs\common1000_thesis_completion_v1\reports\FINAL_THESIS_COMPLETION_REPORT.md"),
        ("Rainfall training-mode comparison", r"analysis_outputs\ph_common1000_rainfall_training_mode_comparison_v1\FINAL_REPORT.md"),
        ("Frozen LSTM rainfall ablation", r"analysis_outputs\ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1\full\reports\FINAL_LSTM_RAIN_ABLATION_REPORT.md"),
        ("Residual event-onset analysis", r"analysis_outputs\ph_common1000_error_onset_final_lstm_rain_ablation_v1\reports\FINAL_REPORT.md"),
        ("Low-cost sensor core", r"analysis_outputs\common1000_lowcost_deployment_summary_v1\reports\core_sensor_ablation.csv"),
        ("pH warm-up", r"analysis_outputs\ph_common1000_warmup_adaptation_lowcost_test_v1\full\reports\FINAL_WARMUP_TEST_REPORT.md"),
        ("DO warm-up in mg/L", r"analysis_outputs\do_common1000_warmup_adaptation_lowcost_mg_l_test_v1\full\reports\FINAL_WARMUP_TEST_REPORT.md"),
        ("Conformal intervals", r"analysis_outputs\common1000_lowcost_conformal_v1\reports\FINAL_CONFORMAL_REPORT.md"),
    ]
    add_table_en(doc, ["Topic", "Path relative to the project root"], evidence, [2500, 6860], font_size=8.2)
    add_source_note(
        doc,
        f"Project root: {ROOT}. The associated CSV, JSON, checkpoint, prediction, and site-level figure files provide the reproducible evidence base.",
    )

    base.add_section_heading(doc, "Appendix B. Completion status", 1)
    add_table_en(
        doc,
        ["Work package", "Status", "Evidence"],
        [
            ["Archive/DVC retrieval and rainfall identification", "Complete", "Hourly rainfall 00045 included in modelling data"],
            ["Common 910-site cohort and fixed split", "Complete", "638 training; 137 validation; 135 test"],
            ["pH baseline, LSTM-FiLM, rainfall/API ablation", "Complete", "Matched all-data and non-event analyses"],
            ["Residual lead time, lag enrichment, range, missed-event analysis", "Complete", "Weak-label interpretation retained"],
            ["Low-cost pH/DO sensor ablation", "Complete", "No target history"],
            ["pH/DO warm-up curves and site figures", "Complete", "39 pH + 39 DO sites"],
            ["Native mg/L DO correction", "Complete", "Earlier outputs deprecated"],
            ["Conformal intervals", "Complete", "45 calibration + 90 evaluation sites"],
            ["Quality assurance", "Complete", "8 audits and 17 focused tests passed"],
            ["Geology/hydrology static-metadata ablation", "Blocked", "Awaiting external data from the supervisor"],
        ],
        [4300, 1300, 3760],
        font_size=8.3,
    )

    base.add_section_heading(doc, "Appendix C. Abbreviations", 1)
    abbreviations = [
        ("API", "Antecedent Precipitation Index"),
        ("DO", "dissolved oxygen, reported here in mg/L"),
        ("FiLM", "feature-wise linear modulation"),
        ("LSTM", "long short-term memory network"),
        ("RMSE", "root mean squared error"),
        ("MAE", "mean absolute error"),
        ("CI", "confidence interval"),
        ("DVC", "Data Version Control"),
    ]
    for start in range(0, len(abbreviations), 4):
        group = abbreviations[start : start + 4]
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        for idx, (term, definition) in enumerate(group):
            if idx:
                separator = p.add_run(";  ")
                base.set_run_font(separator, size=9)
            label = p.add_run(f"{term}: ")
            base.set_run_font(label, size=9, bold=True, color="203864")
            value = p.add_run(definition)
            base.set_run_font(value, size=9, color="1F2937")

    doc.save(OUT_PATH)
    print("English report saved successfully.")


if __name__ == "__main__":
    build_report()
