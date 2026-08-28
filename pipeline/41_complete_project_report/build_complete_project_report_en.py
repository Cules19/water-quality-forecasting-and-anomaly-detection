from pathlib import Path
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = ROOT / "pipeline" / "40_total_report"
sys.path.insert(0, str(HELPER_DIR))
import build_total_report_en as prev  # noqa: E402


base = prev.base
OUT_DIR = ROOT / "analysis_outputs" / "complete_project_report_2026-08-01"
OUT_PATH = OUT_DIR / "Pollution_ID_Complete_Project_Report_EN_2026-08-01.docx"
PROGRESSION_FIG = OUT_DIR / "research_progression_en.png"
BASELINE_FIG = OUT_DIR / "initial_non_neural_baselines_en.png"

FIG_WARMUP = ROOT / "analysis_outputs" / "common1000_lowcost_deployment_summary_v1" / "figures" / "lowcost_warmup_pH_DO.png"
FIG_ONSET = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "event_lead_time_distribution.png"
FIG_LAG = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "lagged_event_enrichment.png"
FIG_COMPRESSION = ROOT / "analysis_outputs" / "ph_common1000_error_onset_final_lstm_rain_ablation_v1" / "figures" / "compression_01645762.png"
FIG_CONFORMAL = ROOT / "analysis_outputs" / "common1000_lowcost_conformal_v1" / "figures" / "conformal_coverage.png"


# Resolved design system:
# preset: narrative_proposal
# first-page pattern: memo_masthead
# named overrides: dark-blue technical report cover, pale-blue metric strip,
# fixed-width technical tables, and Pillow-rendered research-summary figures.


def load_font(size, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_lines(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_stage(draw, xy, number, title, body, fonts, accent=False):
    x1, y1, x2, y2 = xy
    fill = "#EAF2F8" if accent else "#F4F6F9"
    outline = "#2E74B5" if accent else "#9CA3AF"
    draw.rounded_rectangle(xy, radius=24, fill=fill, outline=outline, width=4)
    draw.ellipse((x1 + 24, y1 + 22, x1 + 82, y1 + 80), fill="#2E74B5")
    num_bbox = draw.textbbox((0, 0), str(number), font=fonts[0])
    nx = x1 + 53 - (num_bbox[2] - num_bbox[0]) / 2
    ny = y1 + 51 - (num_bbox[3] - num_bbox[1]) / 2 - 3
    draw.text((nx, ny), str(number), font=fonts[0], fill="white")
    draw.text((x1 + 102, y1 + 24), title, font=fonts[1], fill="#203864")
    y = y1 + 92
    for line in wrap_lines(draw, body, fonts[2], x2 - x1 - 50):
        draw.text((x1 + 25, y), line, font=fonts[2], fill="#1F2937")
        y += 34


def create_progression_figure():
    width, height = 2400, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(25)
    number_font = load_font(28, bold=True)
    stage_title_font = load_font(29, bold=True)
    body_font = load_font(24)
    draw.text((90, 55), "Research progression from benchmark to deployment", font=title_font, fill="#203864")
    draw.text(
        (90, 115),
        "Each stage answered a different question and fixed the design used by the next stage.",
        font=subtitle_font,
        fill="#6B7280",
    )
    boxes = [
        (1, "Data audit", "Archive and DVC retrieval; rainfall availability; target-unit checks", True),
        (2, "Non-neural baselines", "Site and seasonal means, persistence, Ridge, and HGB on common keys", False),
        (3, "Recurrent models", "GRU/LSTM, concatenation and FiLM, site-count scaling, held-out sites", True),
        (4, "Frozen LSTM", "Validation-only selection of 48 h input and learning rate 0.0003", False),
        (5, "Rainfall and API", "Current rain, 6 h accumulation, antecedent wetness, all-data/non-event", True),
        (6, "Event analysis", "Residual thresholds, lead time, lagged enrichment, misses and compression", False),
        (7, "Low-cost deployment", "No target history; sensor ablation; 2-week to 12-month site warm-up", True),
        (8, "Uncertainty and QA", "DO mg/L correction, conformal intervals, audits and retained-result register", False),
    ]
    x_positions = [90, 670, 1250, 1830]
    y_positions = [220, 820]
    idx = 0
    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            number, title, body, accent = boxes[idx]
            draw_stage(draw, (x, y, x + 490, y + 430), number, title, body, (number_font, stage_title_font, body_font), accent)
            idx += 1
    for col in range(3):
        y = 435
        x1 = x_positions[col] + 490
        x2 = x_positions[col + 1]
        draw.line((x1 + 10, y, x2 - 18, y), fill="#2E74B5", width=7)
        draw.polygon([(x2 - 18, y), (x2 - 46, y - 16), (x2 - 46, y + 16)], fill="#2E74B5")
    draw.line((2280, 650, 2280, 765), fill="#2E74B5", width=7)
    draw.line((2280, 765, 120, 765), fill="#2E74B5", width=7)
    draw.line((120, 765, 120, 820), fill="#2E74B5", width=7)
    draw.polygon([(120, 820), (104, 792), (136, 792)], fill="#2E74B5")
    for col in range(3):
        y = 1035
        x1 = x_positions[col] + 490
        x2 = x_positions[col + 1]
        draw.line((x1 + 10, y, x2 - 18, y), fill="#2E74B5", width=7)
        draw.polygon([(x2 - 18, y), (x2 - 46, y - 16), (x2 - 46, y + 16)], fill="#2E74B5")
    image.save(PROGRESSION_FIG, quality=95)


def create_baseline_figure():
    models = [
        ("1 h persistence", 0.049504, "#2E74B5"),
        ("Seasonal mean", 0.314067, "#6B7280"),
        ("HGB + metadata", 0.314212, "#2E74B5"),
        ("Site mean", 0.316512, "#6B7280"),
        ("Ridge + metadata", 0.454810, "#2E74B5"),
        ("HGB, no metadata", 0.494165, "#6B7280"),
        ("Ridge, no metadata", 0.578203, "#6B7280"),
    ]
    width, height = 2100, 1180
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(40, bold=True)
    body_font = load_font(28)
    value_font = load_font(27, bold=True)
    note_font = load_font(23)
    draw.text((85, 50), "Initial non-neural pH baselines on matched test keys", font=title_font, fill="#203864")
    draw.text((85, 108), "Test RMSE in native pH units; lower is better", font=note_font, fill="#6B7280")
    x0 = 620
    max_width = 1240
    scale_max = 0.60
    for i, (name, value, color) in enumerate(models):
        y = 205 + i * 125
        draw.text((85, y + 17), name, font=body_font, fill="#1F2937")
        draw.rounded_rectangle((x0, y, x0 + max_width, y + 64), radius=14, fill="#EEF1F5")
        bar_width = int(max_width * value / scale_max)
        draw.rounded_rectangle((x0, y, x0 + bar_width, y + 64), radius=14, fill=color)
        draw.text((x0 + bar_width + 18, y + 15), f"{value:.3f}", font=value_font, fill="#203864")
    draw.text(
        (85, 1100),
        "Persistence is a forecasting-only reference because it uses the previous pH observation.",
        font=note_font,
        fill="#6B7280",
    )
    image.save(BASELINE_FIG, quality=95)


def add_table(doc, headers, rows, widths, font_size=9):
    table = prev.add_table_en(doc, headers, rows, widths, font_size=font_size)
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        if tr_pr.find(qn("w:cantSplit")) is None:
            tr_pr.append(OxmlElement("w:cantSplit"))
    return table


def add_figure(doc, path, caption, width=6.15):
    return prev.add_figure_en(doc, path, caption, width=width)


def add_bullets(doc, items):
    return prev.add_bullets(doc, items)


def add_source(doc, text):
    return prev.add_source_note(doc, text)


def configure_document(doc):
    prev.configure_styles_en(doc)
    prev.configure_page_en(doc)
    section = doc.sections[0]
    header_paragraph = section.header.paragraphs[0]
    for run in list(header_paragraph.runs):
        header_paragraph._p.remove(run._r)
    base.set_repeat_header_text(
        header_paragraph,
        "Pollution_ID | Complete project report",
        color="6B7280",
    )


def add_cover(doc):
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(62)
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_after = Pt(15)
    kr = kicker.add_run("POLLUTION_ID  |  COMPLETE PROJECT REPORT")
    base.set_run_font(kr, size=10, bold=True, color="2E74B5")
    base.add_top_rule(kicker, color="2E74B5", size="16")

    title = doc.add_paragraph(style="Title")
    tr = title.add_run("Cross-site water-quality modelling:\nfrom baseline evaluation to field deployment")
    base.set_run_font(tr, size=27, bold=True, color="203864")

    subtitle = doc.add_paragraph(style="Subtitle")
    sr = subtitle.add_run(
        "pH and dissolved oxygen forecasting, rainfall effects, event-oriented residual analysis, and low-cost site adaptation"
    )
    base.set_run_font(sr, size=13.5, color="2E74B5", italic=True)

    metadata = [
        ("Scope", "Complete retained research record, from initial non-neural baselines to final uncertainty analysis"),
        ("Data", "Supervisor-provided Archive/common1000 resources and derived hourly rainfall products"),
        ("Date", "1 August 2026"),
        ("Version", "Complete synthesis v1"),
    ]
    for label, value in metadata:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r1 = p.add_run(f"{label}: ")
        base.set_run_font(r1, size=10.2, bold=True, color="203864")
        r2 = p.add_run(value)
        base.set_run_font(r2, size=10.2, color="6B7280")

    doc.add_paragraph().paragraph_format.space_after = Pt(46)
    status = doc.add_paragraph()
    rr = status.add_run(
        "Status: all analyses supported by the local data are complete and retained. The small geology/hydrology metadata ablation remains pending because the required supervisor-provided table is not yet available locally."
    )
    base.set_run_font(rr, size=10.2, bold=True, color="203864")
    doc.add_page_break()


def build_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    create_progression_figure()
    create_baseline_figure()
    figures = (PROGRESSION_FIG, BASELINE_FIG, FIG_WARMUP, FIG_ONSET, FIG_LAG, FIG_COMPRESSION, FIG_CONFORMAL)
    for figure in figures:
        if not figure.exists():
            raise FileNotFoundError(figure)

    doc = Document()
    configure_document(doc)
    doc.core_properties.title = "Cross-site water-quality modelling: from baseline evaluation to field deployment"
    doc.core_properties.subject = "Complete retained Pollution_ID research record"
    doc.core_properties.author = "Pollution_ID Project"
    doc.core_properties.keywords = "pH, dissolved oxygen, rainfall, API, GRU, LSTM, FiLM, site adaptation, conformal prediction"
    add_cover(doc)

    # Abstract and navigation
    base.add_section_heading(doc, "Abstract", 1)
    base.add_paragraph(
        doc,
        "This report consolidates the complete retained analytical record of the Pollution_ID project. The work began with non-neural pH benchmarks, including site and seasonal means, persistence, Ridge regression, and histogram-based gradient boosting (HGB). It then progressed through controlled GRU/LSTM comparisons, feature-wise linear modulation (FiLM), site-count scaling, rainfall and antecedent precipitation experiments, residual-based event analysis, and a deployment study in which historical pH and dissolved oxygen (DO) were deliberately removed. All major comparisons use held-out sites and matched target timestamps; the final 910-site cohort is divided into 638 training, 137 validation, and 135 test sites."
    )
    base.add_paragraph(
        doc,
        "The early benchmarks showed that a one-hour persistence reference was difficult to beat when the previous pH observation was available, while metadata substantially improved both Ridge and HGB. In the final rainfall experiments, current hourly rainfall reduced frozen LSTM-FiLM test RMSE by 1.62%, whereas a matched rainfall-plus-API specification reduced RMSE by 9.26% relative to its baseline. Non-event training consistently increased registered-event enrichment but usually worsened ordinary forecasting error, demonstrating that prediction accuracy and event detectability are separate objectives. Under a three-hour persistence rule, 579 of 2,853 registered events were preceded by an alert, with a median lead time of 8 h among alerted events."
    )
    base.add_paragraph(
        doc,
        "For deployment without target history, the low-cost core used water temperature, specific conductance, turbidity, and hourly rainfall. A frozen recurrent backbone with a 130-parameter site adapter reduced pH macro-site MAE to 0.184 after 12 months and DO macro-site MAE to 0.688 mg/L after 3 months. A complete DO unit audit regenerated all retained results in native mg/L. Site-separated conformal calibration provided empirical prediction intervals, although lower nominal pH intervals under-covered. Together, the results support a thesis centred on the distinction between forecasting accuracy, event-oriented residual information, and practical transfer to new monitoring sites."
    )
    base.add_metric_strip(
        doc,
        [
            ("1,009 → 910", "discovery cohort to strict final cohort"),
            ("-9.26%", "best matched rainfall/API pH RMSE change"),
            ("130", "parameters in the deployment adapter"),
        ],
    )
    base.set_repeat_table_header(doc.tables[-1].rows[0])
    base.add_callout(
        doc,
        "Central finding",
        "Environmental forcing and site adaptation solve different problems. Rainfall and antecedent wetness explain short-term disturbance, whereas warm-up data correct persistent differences at a new site. Neither should be judged solely by whether it changes the same aggregate RMSE.",
        "navy",
    )

    base.add_section_heading(doc, "Report structure", 2)
    contents = [
        "Research development and scope",
        "Data acquisition, rainfall coverage, and cohort construction",
        "Initial non-neural baselines",
        "Recurrent model and FiLM development",
        "Final 910-site architecture selection and frozen LSTM specification",
        "Rainfall, API, and training-distribution experiments",
        "Residual-based event analysis",
        "Low-cost sensor deployment and site warm-up",
        "Dissolved-oxygen unit audit",
        "Predictive uncertainty",
        "Integrated interpretation, limitations, and conclusions",
        "Appendices: chronology, evidence paths, and retained-result status",
    ]
    for item in contents:
        base.add_paragraph(doc, item, style="List Number")

    # 1. Scope and progression
    base.add_section_heading(doc, "1. Research development and scope", 1, page_break=True)
    base.add_paragraph(
        doc,
        "The project was developed sequentially. The first question was whether increasingly flexible models improved one-hour pH prediction on a common set of endpoints. The analysis then moved to site transfer, recurrent architectures, and metadata conditioning. Once dynamic rainfall became available from the supervisor's Archive, the focus shifted from architecture search to environmental forcing, event-oriented residual behaviour, and field deployment. The final work therefore forms a connected research programme rather than a collection of unrelated model trials."
    )
    add_figure(
        doc,
        PROGRESSION_FIG,
        "Figure 1. Analytical progression from data audit and non-neural benchmarks to the final deployment and uncertainty analyses.",
        width=6.25,
    )

    base.add_section_heading(doc, "1.1 Final research questions", 2)
    add_bullets(
        doc,
        [
            "How much predictive value is added by current rainfall and antecedent wetness?",
            "How does removing registered events from training change ordinary error and event-period residual contrast?",
            "How early do persistent abnormal residuals occur before registered events, and what limits their interpretation?",
            "What is the accuracy cost of removing historical pH/DO and expensive cross-target sensors?",
            "How much local warm-up data are required to adapt a frozen cross-site model to a new site?",
            "How should DO units and predictive uncertainty be reported for operational use?",
        ],
    )

    # 2. Data
    base.add_section_heading(doc, "2. Data acquisition, rainfall coverage, and cohort construction", 1)
    base.add_paragraph(
        doc,
        "The repository uses Data Version Control (DVC): Git records code, configuration, and data pointers, while large data objects are retrieved from remote storage. The supervisor-provided Archive contains processed water-quality records, reference metadata, and precipitation products. This distinction matters because the visible GitHub repository alone does not contain every large data file. The local project now includes the required hourly rainfall products."
    )
    add_table(
        doc,
        ["Data layer", "Coverage", "Role in the project"],
        [
            ["Archive rainfall inventory", "1,750 / 1,750 sites prepared", "All available precipitation sites; 25.60 million hourly values and no missing rainfall within the generated sidecars"],
            ["Strict pH modelling cohort", "910 / 910 sites ready", "Final common-key recurrent, rainfall, event, and deployment experiments"],
            ["Final site split", "638 train / 137 validation / 135 test", "No site overlap; validation decisions frozen before test reporting"],
            ["Initial discovery cohort", "1,009 sites", "Early baseline, architecture, scaling, and 30-minute sensitivity work"],
            ["Strict warm-up test cohort", "39 pH / 39 DO sites", "Complete 12-month warm-up plus a common post-warm-up evaluation period"],
        ],
        [2450, 2050, 4860],
        font_size=8.5,
    )
    base.add_paragraph(
        doc,
        "For the 910-site pH cohort, all sites were ready for the common-key rainfall experiment. Across 18.43 million model hours, 16.82 million aligned rainfall observations were present. Within each source-overlap period, coverage was complete. Missing hours outside source overlap were retained as explicit coverage limitations rather than fabricated observations. The full 1,750-site rainfall inventory drew from CAMELSH (258 sites), Met Office products (1,421), and ERA5-seamless data (71)."
    )
    add_source(doc, "Evidence: all_precip_sites_dynamic_rainfall_v1/full/reports/summary.json and ph_common1000_dynamic_rainfall_v1/full/reports/summary.json.")

    base.add_section_heading(doc, "2.1 Cohort evolution", 2)
    base.add_paragraph(
        doc,
        "The apparent change from 1,009 to 910 sites reflects a deliberate tightening of eligibility, not data loss. The 1,009-site discovery cohort supported broad baseline and scaling experiments. The final 910-site cohort required a consistent feature schema, teacher-aligned DO conversion, causal 48-hour windows, dynamic rainfall, and matched target keys. Target-specific held-out and warm-up studies are smaller because they impose additional continuity and observation requirements."
    )
    add_table(
        doc,
        ["Cohort", "What it established", "Why it differs"],
        [
            ["1,009-site discovery", "Baseline ranking, metadata value, FiLM/concatenation tests, site-count scaling", "Broad common1000 inclusion"],
            ["153-site pH held-out evaluation", "Early site-disjoint history versus exogenous-only comparison", "Only pH test sites with valid windows"],
            ["910-site strict final cohort", "Teacher-aligned recurrent, rainfall, API, event, and deployment analyses", "Consistent final features and units"],
            ["39-site strict warm-up test", "Matched 2-week to 12-month calibration curves", "Requires complete calibration and fixed future evaluation"],
        ],
        [2450, 3900, 3010],
        font_size=8.6,
    )

    base.add_section_heading(doc, "2.2 Variables and causal feature construction", 2)
    add_table(
        doc,
        ["Variable", "Code / definition", "Use"],
        [
            ["Water temperature", "00010", "Dynamic predictor and low-cost core"],
            ["Specific conductance", "00095", "Dynamic predictor and low-cost core"],
            ["Dissolved oxygen", "00300", "pH predictor in full-sensor settings; DO target after native-unit correction"],
            ["Turbidity", "63680", "Dynamic predictor and low-cost core"],
            ["pH", "00400", "pH target; excluded from low-cost target history"],
            ["Hourly rainfall", "00045", "Current exogenous forcing"],
            ["Six-hour rainfall sum", "causal rolling total", "Recent accumulated rainfall"],
            ["API", "normalised EWMA; 48 h half-life", "Antecedent wetness/catchment memory"],
        ],
        [2300, 2800, 4260],
        font_size=8.5,
    )
    base.add_paragraph(
        doc,
        "The API is the antecedent precipitation index, not a software interface. It is computed causally from present and earlier rainfall, with exponentially decaying weights. The retained teacher-Archive specification uses a 48-hour half-life. A zero rainfall value is treated as an observed dry hour, during which the index decays normally."
    )

    base.add_section_heading(doc, "2.3 Leakage and evaluation controls", 2)
    add_bullets(
        doc,
        [
            "Sites, not individual rows, define the final training, validation, and test split.",
            "Sequence windows remain within site boundaries and use only causal history.",
            "Feature transforms and metadata imputation are fitted on training sites only.",
            "Hyperparameters, alert thresholds, and warm-up duration decisions use validation sites before the test set is opened.",
            "Matched comparisons use identical held-out sites and target timestamps.",
            "Event indicators are evaluation labels and are never predictors.",
            "Every warm-up duration is evaluated on the same post-12-month keys.",
        ],
    )

    # 3. Initial baselines
    base.add_section_heading(doc, "3. Initial non-neural baselines", 1)
    base.add_paragraph(
        doc,
        "The first complete comparison used the 1,009-site discovery cohort and 2,663,890 common test windows. Its purpose was to establish transparent references before investing in recurrent neural networks. The baselines included site and seasonal means, one-hour persistence, linear Ridge models, and HGB models, with and without static metadata. Event-removed variants changed training rows only; validation and test keys remained fixed."
    )
    add_table(
        doc,
        ["Non-neural model", "Training", "RMSE", "MAE", "R²"],
        [
            ["1 h persistence", "all", "0.04950", "0.02134", "0.99247"],
            ["Seasonal mean", "all", "0.31407", "0.20105", "0.69699"],
            ["HGB + metadata", "all", "0.31421", "0.20192", "0.69671"],
            ["Site mean", "all", "0.31651", "0.20936", "0.69225"],
            ["HGB + metadata", "non-event", "0.32920", "0.20724", "0.66708"],
            ["Ridge + metadata", "non-event", "0.45407", "0.31199", "0.36662"],
            ["Ridge + metadata", "all", "0.45481", "0.30965", "0.36456"],
            ["HGB, no metadata", "all", "0.49417", "0.37296", "0.24983"],
            ["HGB, no metadata", "non-event", "0.49646", "0.37381", "0.24285"],
            ["Ridge, no metadata", "all", "0.57820", "0.44536", "-0.02701"],
            ["Ridge, no metadata", "non-event", "0.57935", "0.44749", "-0.03107"],
        ],
        [2900, 1450, 1500, 1500, 2010],
        font_size=8.2,
    )
    add_figure(
        doc,
        BASELINE_FIG,
        "Figure 2. Initial non-neural pH baseline RMSE on matched test endpoints. The one-hour persistence reference uses the previous pH observation and is therefore not a low-cost deployment model.",
        width=6.25,
    )
    base.add_callout(
        doc,
        "Interpretation of persistence",
        "The previous-hour pH value is an extremely strong predictor because pH is highly autocorrelated. Persistence therefore defines a forecasting reference, not a deployable no-target-history baseline. Later low-cost models answer a harder question and should not be judged against 0.0495 as if they used the same information.",
        "blue",
    )

    base.add_section_heading(doc, "3.1 What the initial baselines established", 2)
    add_bullets(
        doc,
        [
            "Static metadata reduced Ridge RMSE from 0.5782 to 0.4548 and HGB RMSE from 0.4942 to 0.3142.",
            "HGB captured nonlinear relationships that Ridge did not, but its metadata-enhanced accuracy remained close to the seasonal and site-mean references.",
            "Removing registered event targets did not improve the non-neural forecasting baselines.",
            "A low aggregate RMSE did not imply reliable transfer at every site; the early LSTM reference had positive per-site R² at only 300 of 1,003 evaluated sites.",
        ],
    )
    base.add_paragraph(
        doc,
        "A separate early HGB development run used all available validation and test rows rather than the sequence endpoint keys. In that setting, metadata reduced test RMSE from 0.598 to 0.477. This result is retained as evidence that metadata were informative, but the later common-key table above is the appropriate basis for model ranking."
    )
    add_source(doc, "Evidence: ph_common1000_baseline_v1/full/reports/FINAL_REPORT.md and ph_common1000_hgb_v1/full/reports/FINAL_REPORT.md.")

    # 4. Recurrent development
    base.add_section_heading(doc, "4. Recurrent model and FiLM development", 1)
    base.add_paragraph(
        doc,
        "The next stage introduced recurrent models to represent temporal dependence. GRU and LSTM encoders were combined either with direct metadata concatenation or with FiLM. FiLM produces metadata-dependent scaling and offset terms for recurrent hidden states, allowing the shared temporal representation to vary systematically between sites without learning a site identifier embedding."
    )

    base.add_section_heading(doc, "4.1 N = 200 architecture ablation", 2)
    add_table(
        doc,
        ["Model", "Test RMSE", "MAE", "R²", "Selection role"],
        [
            ["LSTM + concatenation", "0.21786", "0.15654", "0.83773", "Comparator"],
            ["GRU + concatenation", "0.22117", "0.15902", "0.83277", "Comparator"],
            ["GRU-FiLM", "0.22325", "0.16097", "0.82961", "FiLM winner by validation RMSE"],
            ["LSTM-FiLM", "0.22773", "0.16488", "0.82270", "FiLM comparator"],
        ],
        [2600, 1600, 1450, 1450, 2260],
        font_size=8.6,
    )
    base.add_paragraph(
        doc,
        "Although LSTM concatenation had the lowest test RMSE in this small ablation, the architecture decision within the prespecified FiLM family was made on validation data: GRU-FiLM validation RMSE was 0.20155 compared with 0.20370 for LSTM-FiLM. This distinction preserves the separation between model selection and test evaluation."
    )

    base.add_section_heading(doc, "4.2 Site-count scaling", 2)
    add_table(
        doc,
        ["Training cohort", "Common evaluation cohort", "Test RMSE", "R²", "Interpretation"],
        [
            ["200 sites", "first 200 test sites", "0.22325", "0.82961", "Reference"],
            ["500 sites", "same first 200", "0.22984", "0.81939", "No gain at this intermediate scale"],
            ["1,009 sites", "same first 200", "0.21621", "0.84019", "Best fixed-site result"],
            ["50 sites", "first 50 test sites", "0.26496", "0.73300", "Small-cohort reference"],
            ["1,009 sites", "same first 50", "0.21471", "0.82468", "19.0% lower RMSE than 50-site model"],
        ],
        [1700, 2300, 1500, 1400, 2460],
        font_size=8.5,
    )
    base.add_paragraph(
        doc,
        "Scaling was evaluated on fixed test subsets to separate the effect of more training sites from changing test-cohort difficulty. The full 1,009-site model performed best on both fixed comparisons, although the 500-site model was not monotonically better than the 200-site model. This supported continued site scaling while cautioning against a simple power-law claim."
    )

    base.add_section_heading(doc, "4.3 Early all-data versus non-event recurrent result", 2)
    add_table(
        doc,
        ["GRU-FiLM, 1,009 sites", "RMSE", "Absolute top-5% enrichment", "Negative-tail enrichment"],
        [
            ["All-data training", "0.25909", "3.912", "3.873"],
            ["Non-event training", "0.27175", "6.641", "8.623"],
        ],
        [2950, 1700, 2450, 2260],
        font_size=8.6,
    )
    base.add_paragraph(
        doc,
        "This was the first clear evidence that non-event training changed the purpose of the residuals: ordinary RMSE worsened, but registered-event enrichment increased sharply. A supervisor-highlighted later run showed the same direction, with enrichment increasing from 3.66 to 4.65 while RMSE was nearly unchanged. Because the configurations differ, the numbers are reported separately rather than pooled."
    )

    base.add_section_heading(doc, "4.4 Site-disjoint transfer and target history", 2)
    add_table(
        doc,
        ["pH model on 153 held-out sites", "Target history", "Metadata", "RMSE", "Macro RMSE"],
        [
            ["GRU-FiLM, non-event", "Yes", "No coords", "0.14082", "0.08655"],
            ["GRU-FiLM, all data", "Yes", "No coords", "0.14167", "0.09683"],
            ["LSTM-FiLM, all data", "Yes", "No coords", "0.15444", "0.11893"],
            ["GRU-FiLM, exogenous", "No", "No coords", "0.39996", "0.32876"],
            ["LSTM-FiLM, exogenous", "No", "No coords", "0.41348", "0.35507"],
            ["GRU, exogenous", "No", "None", "0.47400", "0.38135"],
        ],
        [2850, 1400, 1950, 1400, 1760],
        font_size=7.7,
    )
    base.add_paragraph(
        doc,
        "The large gap between target-history and exogenous-only models quantified the cost of realistic deployment. Metadata-conditioned exogenous models were materially better than a no-metadata GRU, but historical pH still dominated short-horizon accuracy. This finding motivated the later warm-up experiment rather than further attempts to conceal the deployment gap. Coordinates were not retained as a primary metadata input because they did not improve transfer consistently."
    )

    base.add_section_heading(doc, "4.5 Resolution sensitivity", 2)
    base.add_paragraph(
        doc,
        "A 30-minute sensitivity experiment was completed after auditing all 1,009 sites. Only 776 sites had continuous 48-step test windows at this resolution, and the test RMSE was 0.2303. The result was retained as a secondary sensitivity check. Hourly resolution remained the primary basis because it preserved broader coverage and aligned with the rainfall and event analyses."
    )
    add_source(doc, "Evidence: ph_common1000_film_scaling_v1 and ph_supervisor_tasks_common1000_v1/reports/FINAL_COMPLETION_REPORT.md.")

    # 5. Final recurrent comparison and LSTM freeze
    base.add_section_heading(doc, "5. Final 910-site architecture selection and frozen LSTM specification", 1, page_break=True)
    base.add_paragraph(
        doc,
        "The strict final architecture comparison used the 910-site cohort, disjoint site splits, 48-hour causal windows, target history, identical FiLM metadata, common target keys, and the same training budget. Four one-seed models tested GRU and LSTM with and without current rainfall."
    )
    add_table(
        doc,
        ["Model", "Rainfall", "RMSE", "MAE", "NSE", "Macro RMSE", "Parameters"],
        [
            ["GRU-FiLM", "Yes", "0.07153", "0.02450", "0.98495", "0.05032", "28,545"],
            ["GRU-FiLM", "No", "0.08110", "0.03435", "0.98065", "0.05786", "28,161"],
            ["LSTM-FiLM", "No", "0.08155", "0.03422", "0.98043", "0.05773", "33,409"],
            ["LSTM-FiLM", "Yes", "0.09300", "0.03873", "0.97455", "0.05900", "33,921"],
        ],
        [1800, 1150, 1250, 1250, 1250, 1450, 1210],
        font_size=8.1,
    )
    base.add_paragraph(
        doc,
        "Within GRU, rainfall reduced overall RMSE by 11.80% and improved 122 of 135 sites; the site-block mean RMSE-difference interval was -0.00921 to -0.00601. Without rainfall, GRU and LSTM were effectively comparable: the overall difference was 0.56%, and the paired interval included zero. The rainfall LSTM was unstable in this single run, whereas the rainfall GRU was better at 119 of 135 sites. These one-seed results justified a controlled check but do not establish a general architecture ranking."
    )
    base.add_callout(
        doc,
        "Why LSTM remained the final thesis model",
        "Architecture search was not the main research contribution, and the supervisory direction favoured LSTM. The LSTM configuration was therefore tuned on validation data, frozen, and evaluated once on the test set. The earlier one-seed comparison remains a transparent sensitivity result rather than being omitted.",
        "blue",
    )

    base.add_section_heading(doc, "5.1 Validation-only tuning", 2)
    add_table(
        doc,
        ["Decision", "Candidates", "Selected", "Validation evidence"],
        [
            ["Input history", "24, 48, 96 h", "48 h", "RMSE 0.05054; 2.05% and 2.59% lower than 24 h and 96 h"],
            ["Learning rate", "0.0001, 0.0003, 0.001", "0.0003", "RMSE 0.05071; test split remained sealed"],
            ["Hidden size / layers", "prespecified", "64 / 1", "Fixed before final test"],
            ["Dropout", "prespecified", "0.1", "Fixed before final test"],
        ],
        [1900, 2500, 1450, 3510],
        font_size=8.5,
    )
    base.add_paragraph(
        doc,
        "The final current-rainfall LSTM reached test RMSE 0.06975 compared with 0.07091 for the matched no-rain baseline, a 1.62% reduction. Rainfall improved 101 of 135 sites, and the site-level mean RMSE-difference interval was -0.00144 to -0.00046. The gain was smaller than in the GRU comparison but was reproducible under the pre-specified frozen LSTM."
    )
    add_source(doc, "Evidence: ph_common1000_recurrent_comparison_teacher_aligned_do_converted_v1 and the LSTM window, learning-rate, and final-rainfall reports.")

    # 6. Rainfall and non-event
    base.add_section_heading(doc, "6. Rainfall, API, and training-distribution experiments", 1)
    base.add_section_heading(doc, "6.1 Frozen current-rainfall ablation", 2)
    add_table(
        doc,
        ["Stratum", "Baseline RMSE", "Rainfall RMSE", "Relative change", "Sites favouring rainfall"],
        [
            ["All test hours", "0.07091", "0.06975", "-1.62%", "101 / 135"],
            ["Registered event hours", "0.17488", "0.17277", "-1.21%", "15 / 33"],
            ["Non-event hours", "0.06027", "0.05917", "-1.82%", "104 / 135"],
            ["Current-rain hours", "0.07607", "0.07504", "-1.36%", "87 / 135"],
            ["Rain in previous 6 h", "0.07753", "0.07663", "-1.16%", "97 / 135"],
        ],
        [2000, 1700, 1700, 1700, 2260],
        font_size=8.5,
    )
    base.add_paragraph(
        doc,
        "Current rainfall produced a small improvement across the full test set. The event-only site interval was wide because only 33 sites contained registered event hours, so the event-stratum result is not a strong stand-alone claim. The more stable conclusion is that current rainfall added modest information across sites under the frozen model."
    )

    base.add_section_heading(doc, "6.2 Matched rainfall/API feature study", 2)
    add_table(
        doc,
        ["Training / feature set", "RMSE", "Macro RMSE", "Macro NSE", "Top-5% event enrichment"],
        [
            ["All data / baseline", "0.09056", "0.06898", "0.87483", "1.698"],
            ["All data / current rainfall", "0.09059", "0.06505", "0.90706", "2.161"],
            ["All data / rainfall + API + 6 h", "0.08217", "0.06174", "0.91237", "1.462"],
            ["Non-event / baseline", "0.14867", "0.07337", "0.90824", "2.285"],
            ["Non-event / current rainfall", "0.15015", "0.07495", "0.90589", "3.048"],
            ["Non-event / rainfall + API + 6 h", "0.14621", "0.07378", "0.90810", "1.964"],
        ],
        [2950, 1250, 1500, 1450, 2210],
        font_size=8.2,
    )
    base.add_paragraph(
        doc,
        "Under all-data training, the extended rainfall representation reduced RMSE by 9.26% relative to the matched baseline, improved 96 of 135 sites, and produced a paired mean site-RMSE interval of -0.00960 to -0.00498. Current rainfall alone left pooled RMSE nearly unchanged in this configuration but improved macro-site measures and event enrichment. API and the six-hour total supplied catchment memory that a single rainfall hour could not represent."
    )
    base.add_callout(
        doc,
        "Accuracy and detectability are separate objectives",
        "All-data training gave substantially lower forecasting RMSE. Non-event training increased registered-event enrichment for every matched feature set. The correct interpretation is a trade-off between calibration and anomaly contrast, not that one training mode is universally superior.",
        "navy",
    )

    base.add_section_heading(doc, "6.3 Reconciliation of the rainfall results", 2)
    base.add_paragraph(
        doc,
        "The 1.62% and 9.26% improvements are not competing estimates from the same experiment. The first is the final frozen LSTM ablation of current rainfall only. The second comes from a matched feature study that adds current rainfall, a six-hour accumulation, and the 48-hour API under a different frozen configuration. Both support a positive role for precipitation, while showing that the representation of antecedent wetness matters more than the presence of a single rainfall channel."
    )

    # 7. Event analysis
    base.add_section_heading(doc, "7. Residual-based event analysis", 1)
    base.add_paragraph(
        doc,
        "Registered events were treated as weak labels. An unregistered hour is not necessarily a true negative, and the recorded event time may lag the physical disturbance. Site-specific residual thresholds were therefore estimated from validation-site normal periods and evaluated through enrichment and lead time rather than ordinary classification accuracy. The primary alert required three consecutive abnormal hours."
    )
    add_table(
        doc,
        ["Alert definition", "Alerted events", "Rate", "Median lead time", "Interpretation"],
        [
            ["Three consecutive abnormal hours", "579 / 2,853", "20.29%", "8 h (IQR 4–17)", "Primary conservative rule"],
            ["Among analysable events", "579 / 2,577", "22.47%", "8 h", "Excludes events without sufficient pre-window"],
            ["One abnormal hour", "1,339 / 2,853", "46.93%", "4 h", "Sensitivity analysis; more isolated alarms"],
        ],
        [2600, 1900, 1250, 1800, 1810],
        font_size=8.4,
    )
    base.add_paragraph(
        doc,
        "Using all 2,853 registered events as the denominator, 12.58% were alerted at least 6 h early, 7.75% at least 12 h early, and 3.15% at least 24 h early. The result supports measurable warning potential for a subset of events, but the sensitivity is too low for a claim of general-purpose pollution detection."
    )
    add_figure(
        doc,
        FIG_ONSET,
        "Figure 3. Lead-time distribution for registered events preceded by a three-hour persistent abnormal residual.",
    )

    base.add_section_heading(doc, "7.1 Lagged event enrichment", 2)
    add_table(
        doc,
        ["Event-window lead", "Rainfall-model enrichment"],
        [
            ["0 h", "1.733"],
            ["1 h", "1.732"],
            ["3 h", "1.664"],
            ["6 h", "1.587"],
            ["12 h", "1.553"],
            ["24 h", "1.618"],
        ],
        [4680, 4680],
        font_size=8.6,
    )
    base.add_paragraph(
        doc,
        "Enrichment was greatest at the registered event time and one hour earlier, then generally weakened with lead. Values remained above one through 24 h, consistent with early disturbance in some windows. This pattern is descriptive and does not prove that rainfall or a registered discharge caused the residual."
    )
    add_figure(
        doc,
        FIG_LAG,
        "Figure 4. Residual enrichment after shifting registered event windows forward in time.",
    )

    base.add_section_heading(doc, "7.2 Extreme-range compression", 2)
    base.add_paragraph(
        doc,
        "At the supervisor-specified site 01645762, the rainfall model had an overall prediction-on-observation slope of 0.983, lower-tail slope of 1.008, upper-tail slope of 0.981, and a predicted-to-observed 5%–95% range ratio of 1.039. These values provide little evidence of severe compression in the final model at this site, even though regression toward the mean remains a general risk under squared-error training."
    )
    add_figure(
        doc,
        FIG_COMPRESSION,
        "Figure 5. Observed–predicted range and tail-slope check at held-out site 01645762.",
    )

    base.add_section_heading(doc, "7.3 Missed-event attribution", 2)
    base.add_paragraph(
        doc,
        "The missed-event analysis covered 17 sites and 802,108 rows. In a descriptive residual model, the fitted coefficients for six-hour rainfall, turbidity, and flow were approximately 0.00667, 0.0626, and 5.56 × 10⁻⁷. These estimates identify associations worth inspecting, but they are not causal effects because sites differ structurally, event labels are incomplete, and important discharge or catchment variables may be unobserved."
    )
    add_source(doc, "Evidence: ph_common1000_error_onset_final_lstm_rain_ablation_v1/reports/FINAL_REPORT.md.")

    # 8. Low-cost deployment
    base.add_section_heading(doc, "8. Low-cost sensor deployment and site warm-up", 1)
    base.add_paragraph(
        doc,
        "The deployment experiment deliberately removes historical pH and DO. The low-cost core contains only water temperature (00010), specific conductance (00095), turbidity (63680), and hourly rainfall (00045). The full-sensor pH comparator adds DO; the full-sensor DO comparator adds pH. Both use identical held-out target timestamps, so the difference estimates the cost of removing the expensive cross-target measurement."
    )

    base.add_section_heading(doc, "8.1 Full-sensor versus low-cost core", 2)
    add_table(
        doc,
        ["Target", "Input set", "Pooled RMSE", "Pooled MAE", "Macro RMSE", "Macro MAE", "Tolerance coverage"],
        [
            ["pH", "Full sensors", "0.4291", "0.2835", "0.3145", "0.2433", "47.25% within ±0.2"],
            ["pH", "Low-cost", "0.4457", "0.3032", "0.3397", "0.2641", "44.58% within ±0.2"],
            ["DO", "Full sensors", "1.3329", "0.9203", "1.1248", "0.9465", "67.72% within ±1 mg/L"],
            ["DO", "Low-cost", "1.3857", "0.9747", "1.2126", "1.0175", "65.22% within ±1 mg/L"],
        ],
        [900, 1300, 1300, 1300, 1300, 1300, 1960],
        font_size=7.8,
    )
    base.add_paragraph(
        doc,
        "Removing the cross-target sensor increased mean site RMSE by 0.0252 pH (95% interval 0.0076–0.0436; 53 sites favoured low-cost and 82 favoured full sensors). For DO, the mean increase was 0.0878 mg/L (interval -0.0037–0.1759; 47 versus 88 sites), so the average DO degradation was not clearly different from zero. The low-cost core was therefore retained as the deployment basis."
    )

    base.add_section_heading(doc, "8.2 Frozen site adapter", 2)
    base.add_paragraph(
        doc,
        "At each new site, the recurrent backbone remains frozen. Only 130 parameters are estimated: 128 site-specific FiLM offsets plus a positive affine scale and bias. This approach is compared with zero-shot prediction and full fine-tuning over warm-up periods of 2 weeks, 1, 2, 3, 6, 9, and 12 months. It matches the operational scenario in which an expensive sonde is installed temporarily, then removed while low-cost sensors continue to operate."
    )

    base.add_section_heading(doc, "8.3 pH warm-up", 2)
    add_table(
        doc,
        ["12-month method", "Pooled RMSE", "Macro RMSE", "Macro MAE", "Within ±0.2"],
        [
            ["Zero-shot", "0.3867", "0.3581", "0.3041", "—"],
            ["FiLM + affine adapter", "0.2529", "0.2381", "0.1843", "65.82%"],
            ["Full fine-tuning", "0.2429", "—", "0.1808", "67.87%"],
        ],
        [2700, 1700, 1700, 1700, 1560],
        font_size=8.6,
    )
    base.add_paragraph(
        doc,
        "At 12 months, the adapter reduced pooled RMSE by 34.61%, macro RMSE by 33.51%, and macro MAE by 39.38% relative to zero-shot prediction. Thirty-four of 39 test sites improved, and the paired mean site-RMSE interval was entirely below zero. Nine-month macro MAE was 0.2034, whereas 12-month macro MAE was 0.1843. The conservative pH recommendation is therefore 12 months."
    )

    base.add_section_heading(doc, "8.4 Dissolved-oxygen warm-up", 2)
    add_table(
        doc,
        ["12-month method", "Pooled RMSE", "Macro RMSE", "Macro MAE", "Within ±1 mg/L"],
        [
            ["Zero-shot", "1.3306", "1.1690", "0.9604", "—"],
            ["FiLM + affine adapter", "0.9842", "0.8225", "0.6134", "79.89%"],
            ["Full fine-tuning", "0.9301", "—", "0.5400", "82.47%"],
        ],
        [2700, 1700, 1700, 1700, 1560],
        font_size=8.6,
    )
    base.add_paragraph(
        doc,
        "At 12 months, the DO adapter reduced pooled RMSE by 26.03%, macro RMSE by 29.64%, and macro MAE by 36.12%. Thirty-four of 39 sites improved. Validation macro MAE first fell below 1 mg/L at 3 months (0.9768), and the corresponding held-out test macro MAE was 0.6880 mg/L with 77.15% of predictions within ±1 mg/L. Three months is therefore the conservative duration supported by both validation and test, although 12 months gives a further accuracy gain."
    )
    add_figure(
        doc,
        FIG_WARMUP,
        "Figure 6. Low-cost pH and DO performance across site warm-up durations. All durations use the same post-12-month evaluation keys.",
    )

    base.add_section_heading(doc, "8.5 Site heterogeneity and operational thresholds", 2)
    add_table(
        doc,
        ["Target-level criterion", "Earliest duration among sites that reached it", "Not reached"],
        [
            ["pH site MAE ≤ 0.2", "2 wk 14; 1 mo 1; 2 mo 1; 3 mo 3; 6 mo 4; 9 mo 2; 12 mo 2", "12 / 39"],
            ["DO site MAE ≤ 1 mg/L", "2 wk 25; 2 mo 1; 3 mo 5; 6 mo 2; 12 mo 1", "5 / 39"],
        ],
        [2300, 5250, 1810],
        font_size=8.3,
    )
    base.add_paragraph(
        doc,
        "Twenty-seven of 39 pH sites and 34 of 39 DO sites reached the site-level target at some tested duration. A single commissioning duration is therefore inefficient: some sites are usable after two weeks, whereas others remain outside tolerance after one year. A future deployment protocol should use site-specific stopping based on stable error and calibrated uncertainty."
    )

    # 9 DO audit
    base.add_section_heading(doc, "9. Dissolved-oxygen unit audit", 1)
    base.add_paragraph(
        doc,
        "The preparation layer converts Environment Agency percentage saturation to concentration in mg/L and then divides concentration by 20 for fixed model input scaling. An early DO target implementation mistakenly reported the scaled input, x00300/20, as the target. That output was not in native mg/L and has been explicitly deprecated. The corrected reporting target is 20 × x00300."
    )
    add_table(
        doc,
        ["Audit item", "Verified result"],
        [
            ["Native reporting target", "20 × x00300, in mg/L"],
            ["Training target mean", "8.6332 mg/L"],
            ["Training target standard deviation", "2.8847 mg/L"],
            ["Checkpoint effect", "Standardised labels and checkpoint weights are unchanged; SHA values match"],
            ["Regenerated artefacts", "Predictions, absolute errors, tolerance coverage, sensor ablation, warm-up curves, figures, and reports"],
            ["Deprecated artefacts", "Earlier non-mg_l sensor-ablation and warm-up directories carry DEPRECATED_UNIT_ERROR.md"],
        ],
        [2800, 6560],
        font_size=8.5,
    )
    base.add_callout(
        doc,
        "Reporting rule",
        "Only outputs whose path includes mg_l should be cited for dissolved oxygen. The correction did not require re-optimising standardised checkpoints, but it changed every interpretable absolute-error value and tolerance statement.",
        "navy",
    )

    # 10 uncertainty
    base.add_section_heading(doc, "10. Predictive uncertainty", 1, page_break=True)
    base.add_paragraph(
        doc,
        "The final low-cost point models were supplemented with split-conformal intervals. A fixed site hash divided the 135 held-out sites into 45 calibration sites and 90 separate evaluation sites. The point models were not refitted. Because hourly observations within each site are serially dependent, coverage is interpreted as empirical marginal coverage rather than a strict independent-and-identically-distributed guarantee."
    )
    add_table(
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
        [1450, 1750, 1900, 2100, 2160],
        font_size=8.4,
    )
    add_figure(
        doc,
        FIG_CONFORMAL,
        "Figure 7. Nominal and empirical split-conformal coverage for the low-cost pH and DO models.",
    )
    base.add_paragraph(
        doc,
        "The 80% and 90% pooled pH intervals under-covered, indicating heterogeneity across sites or clustering of residuals in time. The 95% intervals achieved or exceeded nominal pooled coverage for both targets but were wide. Conditional calibration by site type, season, or residual scale is the most direct next step if narrower intervals are needed."
    )

    # 11 integrated discussion
    base.add_section_heading(doc, "11. Integrated interpretation", 1)
    base.add_section_heading(doc, "11.1 What changed from the first baseline to the final study", 2)
    add_table(
        doc,
        ["Early view", "Evidence added later", "Final interpretation"],
        [
            ["Lowest RMSE defines the best model", "Persistence wins when past pH is available; no-history deployment is much harder", "Model quality must be judged relative to the information available at deployment"],
            ["Events can be removed to learn normal conditions", "Non-event training increases enrichment but often worsens RMSE", "Accuracy and event contrast are different objectives"],
            ["Rainfall may be unimportant because current rain gives small gains", "API and recent accumulation reduce matched RMSE by 9.26%", "Antecedent wetness carries information beyond current rainfall"],
            ["A larger recurrent model may solve site transfer", "A 130-parameter adapter recovers substantial accuracy", "Local calibration is more valuable than further broad architecture search"],
            ["Point errors are sufficient", "Coverage varies by target and nominal level", "Operational reporting requires uncertainty and unit audits"],
        ],
        [2650, 3500, 3210],
        font_size=8.0,
    )

    base.add_section_heading(doc, "11.2 Supported thesis claims", 2)
    add_bullets(
        doc,
        [
            "Static metadata improve tabular pH baselines, but target history remains the strongest short-horizon predictor.",
            "GRU and LSTM provide comparable no-rain performance under a matched one-seed design; architecture-specific rainfall behaviour is sensitive to optimisation.",
            "Current rainfall provides a small stable gain in the frozen final LSTM, while an antecedent wetness representation provides a larger gain in a separate matched experiment.",
            "Non-event training increases registered-event residual enrichment at the cost of ordinary forecasting accuracy.",
            "Persistent abnormal residuals provide several hours of lead time for a subset of registered events, but current sensitivity is limited.",
            "Low-cost deployment without target history is feasible, and a small frozen-core site adapter materially improves both pH and DO.",
            "Required warm-up duration varies greatly by site; aggregate recommendations should not be interpreted as guarantees for every location.",
            "Native-unit auditing and site-separated empirical intervals are necessary for operational interpretation.",
        ],
    )

    base.add_section_heading(doc, "11.3 Claims not supported", 2)
    add_bullets(
        doc,
        [
            "The results do not show that rainfall is irrelevant or that rainfall alone explains event residuals.",
            "An abnormal residual is not equivalent to a confirmed pollution event.",
            "The one-seed GRU/LSTM comparison does not establish a universal architecture ranking.",
            "The historical-target RMSE near 0.07 is not directly comparable with the low-cost no-target-history RMSE near 0.45.",
            "A three- or twelve-month aggregate warm-up recommendation does not guarantee that every site reaches the stated tolerance.",
            "The conformal analysis does not provide a strict coverage guarantee under arbitrary temporal dependence.",
        ],
    )

    base.add_section_heading(doc, "11.4 Practical deployment interpretation", 2)
    base.add_paragraph(
        doc,
        "A realistic field strategy is to install a multiparameter sonde and the low-cost sensor package together, collect local data during commissioning, fit only the site adapter, and then remove the expensive target sensor once error and uncertainty have stabilised. For pH, 12 months is the conservative aggregate duration supported by the present validation/test design. For DO, 3 months is the shortest aggregate duration supported in both splits, although additional data remain beneficial. Sites that fail the operational threshold should retain the sonde or undergo a more flexible calibration."
    )

    # 12 limitations
    base.add_section_heading(doc, "12. Limitations and remaining work", 1)
    add_table(
        doc,
        ["Limitation", "Consequence", "Response"],
        [
            ["Event register is incomplete", "Unlabelled periods may contain real disturbances", "Use enrichment and lead time; avoid accuracy/specificity claims"],
            ["Architecture comparison uses one seed", "Optimisation variance is unknown", "Treat GRU/LSTM ranking as a sensitivity result"],
            ["Warm-up strict cohort is 39 sites per target", "Deployment results may not represent every hydrological setting", "Report site-level curves and matched-cohort eligibility"],
            ["Conformal rows are temporally dependent", "Formal IID coverage assumptions are weakened", "Report empirical marginal and macro-site coverage"],
            ["Static geology/hydrology table is absent", "The buffering and catchment-response hypothesis is untested", "Run a small matched ablation when the supervisor supplies the data"],
        ],
        [2500, 3400, 3460],
        font_size=8.1,
    )
    base.add_callout(
        doc,
        "Only blocked required analysis",
        "Geology class or alkalinity proxy, baseflow index, imperviousness, and distance to consented discharge points are not present in the local Archive. Once the table is supplied, it can be added to the existing fixed 910-site split. The dynamic rainfall, event, and warm-up experiments do not need to be repeated.",
        "grey",
    )
    base.add_paragraph(
        doc,
        "Positive-unlabelled learning, multi-task learning, and one PatchTST or TCN benchmark remain optional extensions rather than gaps in the completed core analysis. Repeated architecture search, self-supervised reconstruction, and river-network graph models were assigned lower priority because they do not address the main unresolved data limitation."
    )

    # 13 conclusions
    base.add_section_heading(doc, "13. Conclusions", 1)
    base.add_paragraph(
        doc,
        "The complete project develops a coherent path from transparent baselines to an operational deployment question. Initial non-neural benchmarks established the strength of autocorrelation and the value of static metadata. Recurrent and FiLM experiments then separated shared temporal structure from site-specific context and showed that increasing the number of training sites can improve fixed-site transfer. The final rainfall work demonstrated that current precipitation offers a modest stable gain and that antecedent wetness can provide a larger improvement."
    )
    base.add_paragraph(
        doc,
        "The most important conceptual result is that forecasting accuracy, event detectability, and deployment transfer are not the same objective. Non-event training improves event contrast while often worsening RMSE. Residual alerts offer useful lead time for only a subset of registered events. In contrast, low-cost deployment depends mainly on how well a shared model can be calibrated to a new location: a 130-parameter adapter reduced both pH and DO error substantially, with conservative aggregate warm-up periods of 12 months for pH and 3 months for DO."
    )
    base.add_paragraph(
        doc,
        "These findings are sufficient for a defensible thesis contribution when presented with their scope and limitations. The strongest narrative is not that one neural architecture won every comparison. It is that the work identifies what information is required for cross-site water-quality prediction, how residuals behave around registered events, and how a model can be transferred to a new site under realistic sensor constraints."
    )
    base.add_metric_strip(
        doc,
        [
            ("0.184", "12-month pH macro MAE"),
            ("0.688 mg/L", "3-month DO macro MAE"),
            ("8 h", "median lead among alerted events"),
        ],
    )
    base.set_repeat_table_header(doc.tables[-1].rows[0])

    # Appendices
    base.add_section_heading(doc, "Appendix A. Chronological experiment register", 1, page_break=True)
    add_table(
        doc,
        ["Order", "Work package", "Scale", "Retained conclusion"],
        [
            ["1", "Archive, DVC, station and rainfall audit", "1,750 rainfall sites", "Required hourly rainfall was retrieved and prepared"],
            ["2", "Non-neural pH baselines", "1,009 sites", "Persistence dominates with target history; metadata improve Ridge/HGB"],
            ["3", "GRU/LSTM, concatenation and FiLM", "N = 200", "GRU-FiLM selected within FiLM by validation"],
            ["4", "Site-count scaling", "50/200/500/1,009", "Full training scale improves fixed-site evaluation overall"],
            ["5", "All/non-event recurrent comparison", "1,009 sites", "Non-event training increases enrichment but worsens RMSE"],
            ["6", "Site-disjoint transfer", "153 pH test sites", "Target history drives accuracy; metadata aid exogenous-only models"],
            ["7", "30-minute sensitivity", "776 eligible sites", "Secondary RMSE 0.2303; hourly retained as primary"],
            ["8", "Matched GRU/LSTM rainfall comparison", "910 sites", "No-rain architectures comparable; rainfall response seed-sensitive"],
            ["9", "Validation-only LSTM tuning", "137 validation sites", "48 h input and learning rate 0.0003 selected"],
            ["10", "Frozen LSTM rainfall test", "135 test sites", "Current rainfall reduces RMSE by 1.62%"],
            ["11", "Rainfall/API and training-mode study", "135 test sites", "Rainfall + API reduces matched RMSE by 9.26%"],
            ["12", "Residual event-onset analysis", "2,853 events", "579 alerts; median 8 h lead among alerted events"],
            ["13", "Low-cost pH/DO sensor ablation", "135 test sites", "Moderate accuracy cost from removing cross-target sensors"],
            ["14", "pH and DO warm-up", "39 + 39 test sites", "Small site adapter materially improves transfer"],
            ["15", "DO native-unit audit", "All retained DO outputs", "Metrics regenerated in mg/L; earlier outputs deprecated"],
            ["16", "Split-conformal intervals", "45 calibration / 90 evaluation", "Empirical intervals quantified deployment uncertainty"],
        ],
        [650, 3300, 2000, 3410],
        font_size=7.6,
    )

    base.add_section_heading(doc, "Appendix B. Evidence paths", 1)
    evidence = [
        ("Initial fair baselines", r"analysis_outputs\ph_common1000_baseline_v1\full\reports\FINAL_REPORT.md"),
        ("Early HGB development", r"analysis_outputs\ph_common1000_hgb_v1\full\reports\FINAL_REPORT.md"),
        ("FiLM architecture and scaling", r"analysis_outputs\ph_common1000_film_scaling_v1\reports\FINAL_REPORT.md"),
        ("Fixed-common-site scaling", r"analysis_outputs\ph_common1000_film_scaling_v1\reports\COMMON_SITE_SCALING_REPORT.md"),
        ("Site-disjoint recurrent models", r"analysis_outputs\ph_common1000_leave_sites_out_models_v1\full\reports\FINAL_REPORT.md"),
        ("Matched GRU/LSTM comparison", r"analysis_outputs\ph_common1000_recurrent_comparison_teacher_aligned_do_converted_v1\full\reports\RECURRENT_COMPARISON_REPORT.md"),
        ("LSTM input-window selection", r"analysis_outputs\ph_common1000_lstm_film_rain_window_sweep_teacher_aligned_v1\full_validation_only\reports\LSTM_WINDOW_SWEEP_REPORT.md"),
        ("LSTM learning-rate selection", r"analysis_outputs\ph_common1000_lstm_film_rain_lr_sweep_teacher_aligned_v1\full_validation_only\reports\LSTM_LR_SWEEP_REPORT.md"),
        ("Frozen LSTM rainfall ablation", r"analysis_outputs\ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1\full\reports\FINAL_LSTM_RAIN_ABLATION_REPORT.md"),
        ("Rainfall/API training-mode study", r"analysis_outputs\ph_common1000_rainfall_training_mode_comparison_v1\FINAL_REPORT.md"),
        ("Residual event analysis", r"analysis_outputs\ph_common1000_error_onset_final_lstm_rain_ablation_v1\reports\FINAL_REPORT.md"),
        ("Low-cost sensor comparison", r"analysis_outputs\common1000_lowcost_deployment_summary_v1\reports\core_sensor_ablation.csv"),
        ("pH warm-up", r"analysis_outputs\ph_common1000_warmup_adaptation_lowcost_test_v1\full\reports\FINAL_WARMUP_TEST_REPORT.md"),
        ("DO warm-up, native mg/L", r"analysis_outputs\do_common1000_warmup_adaptation_lowcost_mg_l_test_v1\full\reports\FINAL_WARMUP_TEST_REPORT.md"),
        ("Conformal intervals", r"analysis_outputs\common1000_lowcost_conformal_v1\reports\FINAL_CONFORMAL_REPORT.md"),
        ("Master completion record", r"analysis_outputs\common1000_thesis_completion_v1\reports\FINAL_THESIS_COMPLETION_REPORT.md"),
    ]
    add_table(doc, ["Topic", "Path relative to project root"], evidence, [2600, 6760], font_size=7.5)
    add_source(doc, f"Project root: {ROOT}. CSV, JSON, checkpoints, predictions, and site-level figures in the same work-package directories form the reproducible evidence base.")

    base.add_section_heading(doc, "Appendix C. Retained, superseded, and pending outputs", 1)
    add_table(
        doc,
        ["Category", "Outputs", "Reporting rule"],
        [
            ["Retained primary", "910-site frozen LSTM rainfall, rainfall/API, event, low-cost, warm-up, DO mg/L, conformal", "Use for thesis tables and conclusions"],
            ["Retained developmental", "1,009-site baseline, HGB, FiLM scaling, site-disjoint and 30-minute sensitivity", "Use to explain research progression and methodological decisions"],
            ["Superseded but documented", "Earlier one-seed or different-cohort estimates", "Report only with configuration label; never pool with final metrics"],
            ["Deprecated", "DO outputs without the mg_l path marker", "Do not cite; unit error is documented"],
            ["Pending external data", "Geology, baseflow index, imperviousness, discharge-distance ablation", "Run only after supervisor supplies the table"],
        ],
        [2100, 4200, 3060],
        font_size=8.2,
    )

    base.add_section_heading(doc, "Appendix D. Metric interpretation", 1)
    add_table(
        doc,
        ["Metric", "Direction", "Interpretation"],
        [
            ["RMSE", "Lower is better", "Penalises large errors more strongly; reported in native target units"],
            ["MAE", "Lower is better", "Average absolute error; easier to interpret operationally"],
            ["R² / NSE", "Higher is better", "One indicates perfect agreement; zero matches a mean reference; negative values are worse"],
            ["Macro-site metric", "Depends on base metric", "Calculate within each site, then average so large sites do not dominate"],
            ["Tolerance coverage", "Higher is better", "Fraction of predictions within ±0.2 pH or ±1 mg/L DO"],
            ["Event enrichment", "Above one indicates concentration", "Abnormal residuals are more frequent in registered event windows than elsewhere"],
            ["Conformal coverage", "Compare with nominal level", "Empirical fraction of observations inside the prediction interval"],
        ],
        [1900, 1600, 5860],
        font_size=8.4,
    )

    abbreviations = [
        ("API", "antecedent precipitation index"),
        ("CI", "confidence interval"),
        ("DO", "dissolved oxygen"),
        ("DVC", "Data Version Control"),
        ("FiLM", "feature-wise linear modulation"),
        ("GRU", "gated recurrent unit"),
        ("HGB", "histogram-based gradient boosting"),
        ("LSTM", "long short-term memory network"),
        ("MAE", "mean absolute error"),
        ("NSE", "Nash–Sutcliffe efficiency"),
        ("RMSE", "root mean squared error"),
    ]
    base.add_section_heading(doc, "Appendix E. Abbreviations and variable codes", 1)
    add_table(doc, ["Abbreviation", "Meaning"], abbreviations, [2100, 7260], font_size=8.4)
    add_table(
        doc,
        ["Code / notation", "Meaning", "Reporting use"],
        [
            ["00010", "Water temperature", "Low-cost dynamic input"],
            ["00045", "Hourly rainfall", "Low-cost external forcing"],
            ["00095", "Specific conductance", "Low-cost dynamic input"],
            ["00300", "Dissolved oxygen", "DO target in mg/L; full-sensor pH input"],
            ["00400", "pH", "pH target; full-sensor DO input"],
            ["63680", "Turbidity", "Low-cost dynamic input"],
            ["±0.2 / ±1 mg/L", "Operational absolute-error bands", "pH / DO tolerance coverage"],
        ],
        [2100, 3200, 4060],
        font_size=8.3,
    )

    doc.save(OUT_PATH)
    print(str(OUT_PATH))


if __name__ == "__main__":
    build_report()
