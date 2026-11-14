"""
Read-only consolidated data audit for the portfolio project.
Run from backend: python validation/run_data_validation.py

Creates PDF, CSV and JSON evidence under outputs/data_validation/.
Requires pandas, numpy, matplotlib and reportlab.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
RAW = BACKEND / "data" / "raw"
PROC = BACKEND / "data" / "processed"
OUT = BACKEND / "outputs" / "data_validation"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

FILES = {
    "raw_equity": RAW / "equity_daily_returns_nifty50.csv",
    "raw_funds": RAW / "mutual_fund_nav_returns.csv",
    "raw_gold": RAW / "gold_data_unvalidated.csv",
    "raw_gsec": RAW / "india_10y_gsec_yield_changes.csv",
    "stocks_daily": PROC / "stock_daily_returns.csv",
    "funds_daily": PROC / "fund_daily_returns.csv",
    "assets_daily": PROC / "asset_daily_returns.csv",
    "assets_monthly": PROC / "monthly_asset_returns.csv",
    "gsec_daily": PROC / "gsec_yields.csv",
    "gsec_monthly": PROC / "monthly_gsec_features.csv",
    "gold_monthly_pending": PROC / "gold_monthly_returns_pending_verification.csv",
}
rows, details = [], {}

def record(section, check, status, observed, why, action=""):
    rows.append(dict(section=section, check=check, status=status,
                     observed=str(observed), interpretation=why,
                     recommendation=action))

def read(name, path):
    if not path.exists():
        record("Inventory", f"{name} file exists", "FAIL", "Missing",
               "Expected dataset was not found.", "Check path and filename.")
        return None
    try:
        df = pd.read_csv(path)
        details[name] = {
            "file": str(path.relative_to(BACKEND)),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        }
        record("Inventory", f"{name} dimensions", "PASS" if len(df) else "WARN",
               f"{len(df)} rows x {len(df.columns)} columns",
               "Records the dataset size and schema.")
        missing = int(df.isna().sum().sum())
        record("Integrity", f"{name} missing cells", "PASS" if not missing else "WARN",
               missing, "Missing observations can reduce usable history or affect joins.",
               "Determine whether missingness is expected or indicates a data issue.")
        return df
    except Exception as e:
        record("Inventory", f"{name} readable", "FAIL", repr(e),
               "CSV parsing failed.", "Inspect encoding, delimiter and file integrity.")
        return None

def date_checks(name, df, col):
    if col not in df: return
    d = pd.to_datetime(df[col], errors="coerce")
    bad = int(d.isna().sum())
    record("Time series", f"{name} invalid dates", "PASS" if not bad else "FAIL",
           bad, "Invalid dates cannot be reliably ordered or aligned.")
    record("Time series", f"{name} chronological order",
           "PASS" if d.dropna().is_monotonic_increasing else "WARN",
           d.dropna().is_monotonic_increasing,
           "Time order matters for return calculations and chronological validation.",
           "Sort by date before time-series calculations.")
    details[name]["date_min"] = str(d.min()) if d.notna().any() else None
    details[name]["date_max"] = str(d.max()) if d.notna().any() else None

def duplicates(name, df, keys):
    if not all(k in df for k in keys):
        record("Integrity", f"{name} duplicate keys", "NOT TESTED",
               "Required columns missing", "Duplicate-key test requires all key columns.")
        return
    n = int(df.duplicated(keys, keep=False).sum())
    record("Integrity", f"{name} duplicate keys", "PASS" if n == 0 else "WARN",
           n, "Duplicate observations can cause ambiguous or double-counted returns.",
           "Inspect conflicting records; avoid silently dropping duplicates.")

# Raw equity
eq = read("raw_equity", FILES["raw_equity"])
if eq is not None:
    date_checks("raw_equity", eq, "date")
    duplicates("equity date-symbol pairs", eq, ["date", "symbol"])
    for c in ["adjusted_close", "close", "volume"]:
        if c in eq:
            x = pd.to_numeric(eq[c], errors="coerce")
            parse = int((eq[c].notna() & x.isna()).sum())
            invalid = int((x <= 0).sum()) if c != "volume" else int((x < 0).sum())
            record("Integrity", f"Equity {c} numeric parse failures",
                   "PASS" if parse == 0 else "WARN", parse,
                   "Unparseable numeric values can silently reduce usable data.")
            record("Financial", f"Equity {c} invalid values",
                   "PASS" if invalid == 0 else "WARN", invalid,
                   "Prices must be positive; volume should not be negative.")
    if "daily_return_adj_close" in eq:
        r = pd.to_numeric(eq["daily_return_adj_close"], errors="coerce")
        n = int((r.abs() > .20).sum())
        record("Financial", "Equity daily returns above 20% absolute",
               "INVESTIGATE" if n else "PASS", n,
               "Large daily moves may be genuine or caused by corporate actions/data issues.",
               "Review source prices and corporate actions; do not auto-delete.")
    if {"adjusted_close", "close", "adjustment_ratio"}.issubset(eq.columns):
        ratio = pd.to_numeric(eq.adjusted_close, errors="coerce") / pd.to_numeric(eq.close, errors="coerce").replace(0, np.nan)
        stored = pd.to_numeric(eq.adjustment_ratio, errors="coerce")
        n = int(((ratio - stored).abs() > 1e-6).sum())
        record("Financial", "Adjustment-ratio reconciliation",
               "PASS" if n == 0 else "WARN", n,
               "Compares stored adjustment_ratio with adjusted_close / close.",
               "Investigate mismatches and verify provider adjustment methodology.")

# Raw funds
mf = read("raw_funds", FILES["raw_funds"])
if mf is not None:
    date_checks("raw_funds", mf, "date")
    duplicates("fund scheme-date pairs", mf, ["scheme_code", "date"])
    if "nav" in mf:
        nav = pd.to_numeric(mf.nav, errors="coerce")
        n = int((nav <= 0).sum())
        record("Financial", "Mutual fund non-positive NAV",
               "PASS" if n == 0 else "FAIL", n,
               "Conventional fund NAV observations should be positive.")
    if "daily_nav_return" in mf:
        n = int(pd.to_numeric(mf.daily_nav_return, errors="coerce").isna().sum())
        record("Financial", "Mutual fund missing daily returns",
               "WARN" if n else "PASS", n,
               "First observation per scheme commonly has no prior NAV; other gaps need review.",
               "Check whether missing returns are limited to initial scheme observations.")

# Raw gold
gold = read("raw_gold", FILES["raw_gold"])
if gold is not None:
    date_col = "Date" if "Date" in gold else "date"
    date_checks("raw_gold", gold, date_col)
    duplicates("gold dates", gold, [date_col])
    record("Financial", "Gold instrument, market and currency provenance",
           "INVESTIGATE", "Not established by this audit",
           "A price series cannot be interpreted without instrument, market, currency and units.",
           "Verify provenance before including gold in cross-asset analysis.")
    for c in ["Close", "Adj Close", "Open", "High", "Low"]:
        if c in gold:
            x = pd.to_numeric(gold[c], errors="coerce")
            n = int((x <= 0).sum())
            record("Financial", f"Gold {c} non-positive values",
                   "PASS" if n == 0 else "WARN", n,
                   "Non-positive prices are invalid; positive values still require provenance verification.")

# Raw G-Sec
gs = read("raw_gsec", FILES["raw_gsec"])
if gs is not None:
    date_checks("raw_gsec", gs, "date")
    duplicates("G-Sec dates", gs, ["date"])
    if "yield_pct" in gs:
        y = pd.to_numeric(gs.yield_pct, errors="coerce")
        n = int(((y <= 0) | (y > 30)).sum())
        record("Financial", "G-Sec broad yield plausibility screen",
               "PASS" if n == 0 else "INVESTIGATE", n,
               "Broad screening only; units and source conventions must be verified.")
    record("Financial", "G-Sec yield versus bond total return",
           "INVESTIGATE", "Yield series only",
           "Yield changes in basis points are not realized bond investment returns.",
           "Use a verified bond total-return series or clearly label any duration-based approximation.")

# Processed datasets
loaded = {}
for name in ["stocks_daily", "funds_daily", "assets_daily", "assets_monthly",
             "gsec_daily", "gsec_monthly", "gold_monthly_pending"]:
    df = read(name, FILES[name])
    if df is None: continue
    loaded[name] = df
    dcol = "date" if "date" in df else ("month" if "month" in df else None)
    if dcol:
        date_checks(name, df, dcol)
        duplicates(f"{name} duplicate {dcol}", df, [dcol])
    vals = [c for c in df.columns if c != dcol]
    if vals:
        numeric = df[vals].apply(pd.to_numeric, errors="coerce")
        parse = int((df[vals].notna() & numeric.isna()).sum().sum())
        inf = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        record("Integrity", f"{name} numeric parse failures",
               "PASS" if parse == 0 else "WARN", parse,
               "Processed numeric columns should parse consistently.")
        record("Integrity", f"{name} infinite values",
               "PASS" if inf == 0 else "FAIL", inf,
               "Infinite values can break model fitting and covariance estimation.",
               "Trace upstream calculations before applying any replacement.")
        if name in ["stocks_daily", "funds_daily", "assets_daily", "assets_monthly"]:
            threshold = .50 if name == "assets_monthly" else .20
            n = int((numeric.abs() > threshold).sum().sum())
            record("Financial", f"{name} extreme return screen (>{threshold:.0%})",
                   "INVESTIGATE" if n else "PASS", n,
                   "Screening flag only; extreme market returns are not automatically errors.",
                   "Inspect asset/date and reconcile with source prices.")

# Combined daily matrix reconciliation
if all(k in loaded for k in ["stocks_daily", "funds_daily", "assets_daily"]):
    s, f, a = loaded["stocks_daily"], loaded["funds_daily"], loaded["assets_daily"]
    expected = (set(s.columns) | set(f.columns)) - {"date"}
    actual = set(a.columns) - {"date"}
    miss, extra = sorted(expected-actual), sorted(actual-expected)
    record("Pipeline", "Combined daily asset columns",
           "PASS" if not miss and not extra else "WARN",
           f"missing={miss}; extra={extra}",
           "Combined daily matrix should retain the stock and fund asset columns.")
    if all("date" in x for x in [s, f, a]):
        sd = set(pd.to_datetime(s.date, errors="coerce").dropna())
        fd = set(pd.to_datetime(f.date, errors="coerce").dropna())
        ad = set(pd.to_datetime(a.date, errors="coerce").dropna())
        lost = len((sd | fd) - ad)
        record("Pipeline", "Combined daily date coverage",
               "PASS" if lost == 0 else "WARN", lost,
               "Checks whether the combined matrix retains dates from either component dataset.",
               "Confirm whether inner-join date loss is intentional.")

# Monthly coverage and statistical summaries
if "assets_monthly" in loaded:
    m = loaded["assets_monthly"]
    asset_cols = [c for c in m if c != "month"]
    x = m[asset_cols].apply(pd.to_numeric, errors="coerce")
    coverage = x.count().sort_values()
    low = {str(k): int(v) for k, v in coverage[coverage < 60].items()}
    details["monthly_coverage"] = {str(k): int(v) for k, v in coverage.items()}
    record("Time series", "Assets with fewer than 60 monthly returns",
           "WARN" if low else "PASS", low,
           "Short histories can make covariance and simulation estimates unstable.",
           "Use documented minimum-history eligibility; never fill missing returns with zero.")
    flat = x.stack().dropna()
    details["monthly_return_stats"] = {
        "pooled_count": int(len(flat)),
        "mean": float(flat.mean()) if len(flat) else None,
        "std": float(flat.std()) if len(flat) else None,
        "min": float(flat.min()) if len(flat) else None,
        "max": float(flat.max()) if len(flat) else None,
    }
    corr = x.corr(min_periods=30)
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    details["correlation_summary"] = {
        "pair_count": int(len(upper)),
        "min": float(upper.min()) if len(upper) else None,
        "max": float(upper.max()) if len(upper) else None,
        "absolute_correlation_over_0_95": int((upper.abs() > .95).sum()) if len(upper) else 0,
    }
    cov = x.cov(min_periods=30).to_numpy(dtype=float)
    if cov.size and np.isfinite(cov).all():
        eigmin = float(np.linalg.eigvalsh((cov + cov.T) / 2).min())
        record("Statistics", "Monthly covariance PSD screen",
               "PASS" if eigmin >= -1e-8 else "WARN", eigmin,
               "Covariance matrices should be positive semidefinite; pairwise missingness can complicate this.")
    # Coverage chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    coverage.plot(kind="bar", ax=ax)
    ax.set_title("Valid monthly observations by asset")
    ax.set_ylabel("Number of valid monthly returns")
    fig.tight_layout()
    fig.savefig(CHARTS / "monthly_asset_coverage.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    flat.hist(bins=60, ax=ax)
    ax.set_title("Pooled monthly return distribution")
    ax.set_xlabel("Monthly return (decimal)")
    fig.tight_layout()
    fig.savefig(CHARTS / "monthly_return_distribution.png", dpi=150)
    plt.close(fig)

# Explicit inherited limitations from the preliminary audit
limitations = [
    "The equity universe is 50 present-day constituents, not point-in-time historical index membership; survivorship bias may affect historical conclusions.",
    "Adjusted-close returns are preliminary price-return proxies; provider adjustment methodology and dividend/corporate-action treatment require verification.",
    "The G-Sec series contains yields and yield changes, not observed bond total returns.",
    "Gold instrument, market, currency and units remain unverified; its monthly return file is explicitly pending verification.",
    "The preliminary sample starts in September 2016, not 2015.",
    "This audit is data-quality screening, not certification of forecast accuracy, investment suitability or production readiness.",
]

# Persist machine-readable evidence
result_df = pd.DataFrame(rows)
result_df.to_csv(OUT / "validation_results.csv", index=False)
summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "backend": str(BACKEND),
    "dataset_details": details,
    "status_counts": result_df.status.value_counts().to_dict(),
    "results": rows,
    "known_limitations": limitations,
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=7, leading=8))
styles.add(ParagraphStyle(name="SectionX", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
pdf = OUT / "data_validation_report.pdf"
doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
                        topMargin=15*mm, bottomMargin=17*mm)
story = [
    Spacer(1, 25*mm),
    Paragraph("DATA VALIDATION &amp; AUDIT REPORT", styles["Title"]),
    Spacer(1, 5*mm),
    Paragraph("Personal Portfolio and Asset Optimization Platform", styles["Heading2"]),
    Spacer(1, 4*mm),
    Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles["BodyText"]),
    Spacer(1, 8*mm),
    Paragraph("This report records checks executed against the datasets present at runtime. Anomaly flags are not automatically classified as errors.", styles["BodyText"]),
    PageBreak(),
    Paragraph("1. Executive Summary", styles["SectionX"]),
]
counts = result_df.status.value_counts().to_dict()
summary_table = [["Status", "Count"]] + [[str(k), str(v)] for k, v in sorted(counts.items())]
t = Table(summary_table, colWidths=[65*mm, 25*mm])
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#243B5A")),
                       ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                       ("GRID",(0,0),(-1,-1),.3,colors.grey),("PADDING",(0,0),(-1,-1),5)]))
story += [t, Spacer(1, 4*mm),
          Paragraph("PASS = check met its rule. WARN = review needed. FAIL = defined integrity violation. INVESTIGATE = unresolved uncertainty requiring source/context. NOT TESTED = required fields unavailable.", styles["BodyText"]),
          Paragraph("2. Dataset Inventory", styles["SectionX"])]
for name, info in details.items():
    if isinstance(info, dict) and "rows" in info:
        story.append(Paragraph(
            f"<b>{name}</b>: {info['rows']:,} rows; {len(info['columns'])} columns; "
            f"date range {info.get('date_min','not available')} to {info.get('date_max','not available')}.",
            styles["BodyText"]))
story.append(Paragraph("3. Detailed Validation Results", styles["SectionX"]))
for section in result_df.section.drop_duplicates():
    story.append(Paragraph(section, styles["Heading3"]))
    subset = result_df[result_df.section == section]
    data = [["Check", "Status", "Observed", "Interpretation / recommended action"]]
    for _, r in subset.iterrows():
        data.append([
            Paragraph(str(r.check), styles["Tiny"]),
            Paragraph(str(r.status), styles["Tiny"]),
            Paragraph(str(r.observed)[:220], styles["Tiny"]),
            Paragraph((str(r.interpretation) + " " + str(r.recommendation))[:520], styles["Tiny"]),
        ])
    tab = Table(data, colWidths=[37*mm, 22*mm, 30*mm, 83*mm], repeatRows=1)
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#243B5A")),
                             ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("GRID",(0,0),(-1,-1),.25,colors.HexColor("#AAAAAA")),
                             ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),3)]))
    story += [tab, Spacer(1, 3*mm)]
story.append(PageBreak())
story.append(Paragraph("4. Statistical Visualizations", styles["SectionX"]))
for image_name in ["monthly_asset_coverage.png", "monthly_return_distribution.png"]:
    image_path = CHARTS / image_name
    if image_path.exists():
        story += [Paragraph(image_name.replace("_", " ").replace(".png", "").title(), styles["Heading3"]),
                  Image(str(image_path), width=165*mm, height=76*mm), Spacer(1, 3*mm)]
story.append(Paragraph("5. Known Project-Level Limitations", styles["SectionX"]))
for item in limitations:
    story.append(Paragraph("• " + item, styles["BodyText"]))
    story.append(Spacer(1, 2*mm))
story.append(Paragraph("6. Recommended Follow-up", styles["SectionX"]))
for item in [
    "Resolve flagged observations only after checking original source records and corporate actions.",
    "Document provenance, instrument definitions, currencies, units, adjustment conventions and data availability timestamps.",
    "Apply explicit asset-history eligibility rules; do not impute missing returns as zero.",
    "Reconcile raw-to-processed datasets and retain input versions and reproducible build commands.",
    "Complete walk-forward ML validation and portfolio optimizer stress testing as separate workstreams.",
]:
    story.append(Paragraph("• " + item, styles["BodyText"]))
    story.append(Spacer(1, 2*mm))

def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(14*mm, 9*mm, "Internal project testing document")
    canvas.drawRightString(A4[0]-14*mm, 9*mm, f"Page {document.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("\nDATA VALIDATION FINISHED")
print("PDF:", pdf)
print("CSV:", OUT / "validation_results.csv")
print("JSON:", OUT / "validation_summary.json")
print("\nStatus counts:\n", result_df.status.value_counts().to_string())
