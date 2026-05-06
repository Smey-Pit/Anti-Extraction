"""
playwright_renderer.py
======================
Playwright-based HTML screenshot renderer for the UI dataset pipeline.

Produces realistic browser screenshots from content bank items using
per-category HTML/CSS templates. Each template mimics a real web portal
with chrome bar, navigation, fonts, and layout — giving the visual
complexity (anti-aliasing, CSS shadows, mixed fonts, sub-pixel rendering)
that PIL cannot reproduce.

Dependencies
------------
    pip install playwright --break-system-packages
    playwright install chromium

    # Colab / HPC (no display server, no root):
    apt-get install -y chromium-browser   # if playwright's bundled one fails
    # --no-sandbox is set automatically by this module

Public API
----------
    # Async (preferred — reuses browser across calls)
    async with PlaywrightRenderer(width=1280, height=900) as renderer:
        img_bytes = await renderer.render(item, category)

    # Sync convenience wrapper (opens/closes browser per call — use for testing)
    img_bytes = render_sync(item, category, width=1280, height=900)

    img_bytes is a PNG bytes object ready to write to disk.

Viewport sizes are randomised per-item when width/height are not given,
matching the variation in pil_renderer.py so the two modes are comparable.
"""

from __future__ import annotations

import asyncio
import random
import textwrap
from typing import Optional


# ---------------------------------------------------------------------------
# VIEWPORT SIZE POOL  (mirrors pil_renderer.py options)
# ---------------------------------------------------------------------------

VIEWPORT_WIDTHS  = [1024, 1152, 1280, 1440]
VIEWPORT_HEIGHTS = [800,  900,  1024, 1100]


def _pick_viewport(rng: random.Random) -> tuple[int, int]:
    return rng.choice(VIEWPORT_WIDTHS), rng.choice(VIEWPORT_HEIGHTS)


# ---------------------------------------------------------------------------
# HTML HELPERS
# ---------------------------------------------------------------------------

def _e(text) -> str:
    """HTML-escape a value for safe insertion into templates."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _chrome_bar(url: str) -> str:
    return f"""
<div class="chrome-bar">
  <div class="chrome-dots">
    <div class="dot r"></div>
    <div class="dot y"></div>
    <div class="dot g"></div>
  </div>
  <div class="chrome-url">&#128274; {_e(url)}</div>
</div>"""


_CHROME_CSS = """
.chrome-bar {
  background: #dee1e6;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid #c8ccd2;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.chrome-dots { display: flex; gap: 6px; }
.dot { width: 12px; height: 12px; border-radius: 50%; }
.dot.r { background: #ff5f56; }
.dot.y { background: #ffbd2e; }
.dot.g { background: #27c93f; }
.chrome-url {
  background: white;
  border-radius: 4px;
  padding: 4px 12px;
  flex: 1;
  font-size: 13px;
  color: #555;
  border: 1px solid #c8ccd2;
  max-width: 520px;
  margin: 0 auto;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
"""


# ---------------------------------------------------------------------------
# BANKING TEMPLATE
# ---------------------------------------------------------------------------

def _html_banking(item: dict) -> str:
    bank     = _e(item.get("bank_name", ""))
    holder   = _e(item.get("account_holder", ""))
    acct_no  = _e(item.get("account_number", ""))
    acct_typ = _e(item.get("account_type", ""))
    period   = _e(item.get("statement_period", ""))
    opening  = _e(item.get("opening_balance", ""))
    closing  = _e(item.get("closing_balance", ""))
    note     = _e(item.get("summary_note", ""))
    url      = f"secure.{item.get('bank_name','bank').lower().replace(' ','')}online.com/statements"

    rows = ""
    for t in item.get("transactions", []):
        amt   = str(t.get("amount", ""))
        color = "#c0392b" if amt.lstrip("+").startswith("-") else "#27ae60"
        rows += f"""
        <tr>
          <td>{_e(t.get('date',''))}</td>
          <td>{_e(t.get('description',''))}</td>
          <td style="color:{color};font-weight:600;text-align:right">{_e(amt)}</td>
          <td style="text-align:right">{_e(t.get('running_balance',''))}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{bank} — Statement</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f0f2f5; color: #1a1a2e; font-size: 14px;
}}
.page {{ max-width: 920px; margin: 24px auto; padding: 0 16px; }}
.bank-header {{
  background: #1a3a5c; color: white; padding: 20px 28px;
  border-radius: 8px 8px 0 0;
  display: flex; justify-content: space-between; align-items: center;
}}
.bank-name {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
.bank-tag  {{ font-size: 11px; opacity: 0.7; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }}
.card {{
  background: white; border-radius: 0 0 8px 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}}
.account-bar {{
  background: #eaf0fb; padding: 16px 28px;
  display: flex; gap: 40px; flex-wrap: wrap;
  border-bottom: 1px solid #dce4f0;
}}
.account-field label {{
  font-size: 11px; color: #666;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.account-field p {{
  font-size: 15px; font-weight: 600; color: #1a3a5c; margin-top: 2px;
}}
.balances {{ display: flex; border-bottom: 1px solid #eee; }}
.balance-item {{
  flex: 1; padding: 18px 28px; border-right: 1px solid #eee;
}}
.balance-item:last-child {{ border-right: none; }}
.balance-item .label {{
  font-size: 11px; color: #888;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.balance-item .amount {{
  font-size: 24px; font-weight: 700; color: #1a3a5c; margin-top: 4px;
}}
.section-title {{
  padding: 16px 28px 8px;
  font-size: 12px; font-weight: 700; color: #555;
  text-transform: uppercase; letter-spacing: 0.6px;
}}
table {{ width: 100%; border-collapse: collapse; }}
th {{
  padding: 10px 28px; text-align: left;
  font-size: 11px; color: #888;
  text-transform: uppercase; letter-spacing: 0.5px;
  border-bottom: 2px solid #eee;
}}
td {{ padding: 12px 28px; border-bottom: 1px solid #f5f5f5; font-size: 13.5px; }}
tr:hover td {{ background: #fafbff; }}
.note {{
  padding: 16px 28px; font-size: 12.5px; color: #7a6000;
  background: #fffbea; border-top: 1px solid #f0e68c;
  border-radius: 0 0 8px 8px;
}}
</style></head><body>
{_chrome_bar(url)}
<div class="page">
  <div class="bank-header">
    <div>
      <div class="bank-name">{bank}</div>
      <div class="bank-tag">Online Banking Portal</div>
    </div>
    <div style="text-align:right;font-size:12px;opacity:0.8">
      Statement Period<br><strong>{period}</strong>
    </div>
  </div>
  <div class="card">
    <div class="account-bar">
      <div class="account-field"><label>Account Holder</label><p>{holder}</p></div>
      <div class="account-field"><label>Account Number</label><p>{acct_no}</p></div>
      <div class="account-field"><label>Account Type</label><p>{acct_typ}</p></div>
    </div>
    <div class="balances">
      <div class="balance-item">
        <div class="label">Opening Balance</div>
        <div class="amount">{opening}</div>
      </div>
      <div class="balance-item">
        <div class="label">Closing Balance</div>
        <div class="amount">{closing}</div>
      </div>
    </div>
    <div class="section-title">Transaction History</div>
    <table>
      <thead><tr>
        <th>Date</th><th>Description</th>
        <th style="text-align:right">Amount</th>
        <th style="text-align:right">Balance</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="note">&#9888;&#65039; {note}</div>
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# MEDICAL TEMPLATE
# ---------------------------------------------------------------------------

def _html_medical(item: dict) -> str:
    hospital = _e(item.get("hospital_name", ""))
    patient  = _e(item.get("patient_name", ""))
    dob      = _e(item.get("dob", ""))
    pid      = _e(item.get("patient_id", ""))
    visit    = _e(item.get("visit_date", ""))
    doctor   = _e(item.get("attending_physician", ""))
    complaint= _e(item.get("chief_complaint", ""))
    diagnosis= _e(item.get("diagnosis", ""))
    notes    = _e(item.get("clinical_notes", ""))
    followup = _e(item.get("follow_up", ""))
    url      = f"patient.{item.get('hospital_name','clinic').lower().replace(' ','').replace(chr(39),'')}health.org/records"

    meds = "".join(f"<li>{_e(m)}</li>" for m in item.get("medications", []))

    lab_rows = ""
    for lab in item.get("lab_results", []):
        flag  = lab.get("flag", "Normal")
        fc    = "#c0392b" if flag != "Normal" else "#27ae60"
        lab_rows += f"""<tr>
          <td>{_e(lab.get('test',''))}</td>
          <td style="font-weight:600">{_e(lab.get('value',''))}</td>
          <td style="color:#888">{_e(lab.get('reference_range',''))}</td>
          <td style="color:{fc};font-weight:700">{_e(flag)}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{hospital} — Patient Record</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f7f9fc; color: #2d3748; font-size: 14px;
}}
.page {{ max-width: 920px; margin: 24px auto; padding: 0 16px; }}
.med-header {{
  background: #2b6cb0; color: white; padding: 18px 28px;
  border-radius: 8px 8px 0 0;
  display: flex; justify-content: space-between; align-items: center;
}}
.med-name {{ font-size: 20px; font-weight: 700; }}
.med-tag  {{ font-size: 11px; opacity: 0.75; margin-top: 3px; }}
.card {{ background: white; box-shadow: 0 2px 12px rgba(0,0,0,0.07); border-radius: 0 0 8px 8px; }}
.patient-bar {{
  background: #ebf4ff; padding: 14px 28px;
  display: flex; gap: 32px; flex-wrap: wrap;
  border-bottom: 1px solid #bee3f8;
}}
.pf label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
.pf p     {{ font-size: 14px; font-weight: 600; color: #2b6cb0; margin-top: 2px; }}
.section  {{ padding: 18px 28px; border-bottom: 1px solid #f0f0f0; }}
.section h3 {{
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.6px; color: #718096; margin-bottom: 10px;
}}
.diagnosis-box {{
  background: #fff5f5; border-left: 4px solid #c53030;
  padding: 12px 16px; border-radius: 0 6px 6px 0;
  font-size: 15px; font-weight: 600; color: #c53030;
}}
ul {{ padding-left: 20px; }}
li {{ margin-bottom: 5px; font-size: 13.5px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
th {{
  padding: 8px 12px; text-align: left; font-size: 11px; color: #999;
  text-transform: uppercase; background: #f7fafc; border-bottom: 2px solid #e2e8f0;
}}
td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
.notes-text {{ font-size: 13.5px; line-height: 1.7; color: #4a5568; }}
.followup {{
  background: #f0fff4; padding: 14px 28px;
  border-top: 1px solid #c6f6d5;
  font-size: 13px; color: #276749;
  border-radius: 0 0 8px 8px;
}}
</style></head><body>
{_chrome_bar(url)}
<div class="page">
  <div class="med-header">
    <div>
      <div class="med-name">{hospital}</div>
      <div class="med-tag">Patient Health Portal — Confidential Medical Record</div>
    </div>
    <div style="text-align:right;font-size:12px;opacity:0.8">
      Visit Date<br><strong>{visit}</strong>
    </div>
  </div>
  <div class="card">
    <div class="patient-bar">
      <div class="pf"><label>Patient</label><p>{patient}</p></div>
      <div class="pf"><label>Date of Birth</label><p>{dob}</p></div>
      <div class="pf"><label>Patient ID</label><p>{pid}</p></div>
      <div class="pf"><label>Physician</label><p>{doctor}</p></div>
    </div>
    <div class="section">
      <h3>Chief Complaint</h3>
      <p style="font-size:14px">{complaint}</p>
    </div>
    <div class="section">
      <h3>Diagnosis</h3>
      <div class="diagnosis-box">{diagnosis}</div>
    </div>
    <div class="section">
      <h3>Medications Prescribed</h3>
      <ul>{meds}</ul>
    </div>
    <div class="section">
      <h3>Lab Results</h3>
      <table>
        <thead><tr>
          <th>Test</th><th>Result</th><th>Reference Range</th><th>Flag</th>
        </tr></thead>
        <tbody>{lab_rows}</tbody>
      </table>
    </div>
    <div class="section">
      <h3>Clinical Notes</h3>
      <p class="notes-text">{notes}</p>
    </div>
    <div class="followup">&#10003; Follow-up: {followup}</div>
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# NEWS TEMPLATE
# ---------------------------------------------------------------------------

def _html_news(item: dict) -> str:
    outlet   = _e(item.get("outlet_name", ""))
    headline = _e(item.get("headline", ""))
    byline   = _e(item.get("byline", ""))
    dateline = _e(item.get("dateline", ""))
    cat_tag  = _e(item.get("category_tag", ""))
    lead     = _e(item.get("lead_paragraph", ""))
    pq       = _e(item.get("pull_quote", ""))
    tags     = "  ·  ".join(_e(t) for t in item.get("tags", []))
    url      = f"www.{item.get('outlet_name','news').lower().replace(' ','').replace(chr(39),'')}.com/politics"

    body_html = "".join(
        f"<p>{_e(p)}</p>" for p in item.get("body_paragraphs", [])
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{outlet}</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: Georgia, 'Times New Roman', serif; background: #fff; color: #111; }}
.site-header {{
  border-bottom: 3px solid #111; padding: 12px 16px 8px;
  max-width: 880px; margin: 0 auto;
  display: flex; justify-content: space-between; align-items: flex-end;
}}
.outlet {{
  font-size: 30px; font-weight: 900; letter-spacing: -1px;
  font-family: 'Times New Roman', serif;
}}
.nav {{
  font-family: -apple-system, sans-serif;
  font-size: 12px; color: #555;
}}
.page {{ max-width: 880px; margin: 0 auto; padding: 28px 16px; }}
.cat-tag {{
  font-family: -apple-system, sans-serif; font-size: 11px;
  font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  color: #c0392b; margin-bottom: 12px;
}}
h1 {{
  font-size: 34px; line-height: 1.2; font-weight: 900;
  margin-bottom: 14px; letter-spacing: -0.5px;
}}
.byline {{
  font-family: -apple-system, sans-serif;
  font-size: 13px; color: #555; margin-bottom: 4px;
}}
.dateline {{
  font-family: -apple-system, sans-serif;
  font-size: 12px; color: #999;
  border-bottom: 1px solid #eee; padding-bottom: 16px; margin-bottom: 20px;
}}
.lead {{
  font-size: 18px; line-height: 1.65; font-weight: 400;
  margin-bottom: 20px; color: #222;
}}
p {{ font-size: 15px; line-height: 1.8; margin-bottom: 16px; color: #333; }}
.pull-quote {{
  border-left: 4px solid #c0392b; margin: 24px 0;
  padding: 12px 20px; font-size: 20px;
  font-style: italic; line-height: 1.5; color: #222;
}}
.tags {{
  margin-top: 24px; padding-top: 16px; border-top: 1px solid #eee;
  font-family: -apple-system, sans-serif; font-size: 12px; color: #888;
}}
</style></head><body>
{_chrome_bar(url)}
<div style="padding:0 16px">
  <div class="site-header">
    <div class="outlet">{outlet}</div>
    <div class="nav">Politics · Opinion · World · Business · Science</div>
  </div>
</div>
<div class="page">
  <div class="cat-tag">{cat_tag}</div>
  <h1>{headline}</h1>
  <div class="byline">By {byline}</div>
  <div class="dateline">{dateline}</div>
  <p class="lead">{lead}</p>
  {body_html}
  <div class="pull-quote">&#8220;{pq}&#8221;</div>
  <div class="tags">Tags: {tags}</div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# COPYRIGHT TEMPLATE
# ---------------------------------------------------------------------------

def _html_copyright(item: dict) -> str:
    title       = _e(item.get("title", ""))
    author      = _e(item.get("author", ""))
    publisher   = _e(item.get("publisher", ""))
    copyright_l = _e(item.get("copyright_line", ""))
    page_no     = _e(item.get("page_number", ""))
    chapter     = _e(item.get("chapter_or_scene", ""))
    content_raw = item.get("content", "")
    ctype       = item.get("content_type", "book_excerpt")
    url         = f"reader.{item.get('publisher','press').lower().replace(' ','').replace(chr(39),'')}.com/read"

    type_label  = {
        "book_excerpt":      "Book Excerpt",
        "screenplay":        "Screenplay",
        "newspaper_feature": "Feature Article",
    }.get(ctype, "Excerpt")

    # For screenplays use monospace, others use serif
    if ctype == "screenplay":
        content_css = "font-family: 'Courier New', Courier, monospace; font-size: 13px; line-height: 1.9;"
        # Render line breaks for screenplay formatting
        content_html = "".join(
            f'<p style="margin-bottom:4px">{_e(ln)}</p>' if ln.strip()
            else '<p style="margin-bottom:12px">&nbsp;</p>'
            for ln in content_raw.split("\n")
        )
    else:
        content_css  = "font-family: Georgia, serif; font-size: 15px; line-height: 1.85;"
        content_html = "".join(
            f"<p>{_e(p)}</p>"
            for p in content_raw.split("\n\n") if p.strip()
        ) or f"<p>{_e(content_raw)}</p>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title} — {author}</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, sans-serif;
  background: #d8d8d8; color: #111;
}}
.viewer-bar {{
  background: #3c3c3c; color: #ccc; padding: 8px 20px;
  font-size: 12px; display: flex; justify-content: space-between; align-items: center;
}}
.viewer-title {{ font-weight: 600; color: #eee; }}
.viewer-meta  {{ font-size: 11px; color: #999; }}
.page-wrap  {{ display: flex; justify-content: center; padding: 32px 16px 40px; }}
.page {{
  background: #fffef8; width: 680px; min-height: 820px;
  padding: 72px 80px; box-shadow: 0 4px 28px rgba(0,0,0,0.28);
  position: relative;
}}
.book-header {{
  text-align: center; margin-bottom: 40px; padding-bottom: 20px;
  border-bottom: 1px solid #ddd;
}}
.type-badge {{
  display: inline-block; background: #2d3748; color: white;
  font-size: 10px; padding: 2px 8px; border-radius: 3px;
  letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 10px;
}}
.work-title  {{ font-size: 18px; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px; }}
.work-author {{ font-size: 13px; color: #666; margin-bottom: 6px; }}
.chapter-label {{ font-size: 12px; color: #888; font-style: italic; }}
.content {{ {content_css} color: #2a2a2a; }}
.content p {{ margin-bottom: 1em; }}
.copyright {{
  position: absolute; bottom: 28px; left: 80px; right: 80px;
  font-size: 10px; color: #bbb; text-align: center;
  border-top: 1px solid #e8e8e8; padding-top: 10px;
}}
.page-num {{
  position: absolute; bottom: 28px; right: 80px;
  font-size: 11px; color: #bbb;
}}
</style></head><body>
{_chrome_bar(url)}
<div class="viewer-bar">
  <div class="viewer-title">{title} — {author}</div>
  <div class="viewer-meta">{publisher} · p.&nbsp;{page_no}</div>
</div>
<div class="page-wrap">
  <div class="page">
    <div class="book-header">
      <div class="type-badge">{type_label}</div>
      <div class="work-title">{title}</div>
      <div class="work-author">{author}</div>
      <div class="chapter-label">{chapter}</div>
    </div>
    <div class="content">{content_html}</div>
    <div class="copyright">{copyright_l}</div>
    <div class="page-num">{page_no}</div>
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# LEGAL TEMPLATE
# ---------------------------------------------------------------------------

def _html_legal(item: dict) -> str:
    title      = _e(item.get("title", ""))
    doc_type   = item.get("document_type", "contract")
    jx         = _e(item.get("jurisdiction", ""))
    date       = _e(item.get("date", ""))
    ref        = _e(item.get("case_or_ref_number", ""))
    notary     = _e(item.get("notary_note", "") or "")
    url        = "ecourt.gov/filings/documents"

    type_label = {
        "contract":       "Contract",
        "nda":            "Non-Disclosure Agreement",
        "will":           "Last Will and Testament",
        "eviction_notice":"Eviction Notice",
        "court_filing":   "Court Filing",
    }.get(doc_type, "Legal Document")

    parties_html = ""
    for party in item.get("parties", []):
        parties_html += f"""
        <div class="party-row">
          <span class="party-role">{_e(party.get('role',''))}</span>
          <span class="party-name">{_e(party.get('name',''))}</span>
        </div>"""

    clauses_html = ""
    for clause in item.get("clauses", []):
        clauses_html += f"""
        <div class="clause">
          <div class="clause-head">{_e(clause.get('number',''))} {_e(clause.get('heading',''))}</div>
          <div class="clause-text">{_e(clause.get('text',''))}</div>
        </div>"""

    sigs_html = ""
    for sig in item.get("signature_block", []):
        signed = _e(sig.get("date_signed", "")) or "____________________"
        sigs_html += f"""
        <div class="sig-block">
          <div class="sig-line">____________________________</div>
          <div class="sig-role">{_e(sig.get('role',''))}</div>
          <div class="sig-name">{_e(sig.get('name',''))}</div>
          <div class="sig-date">Date: {signed}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Times New Roman', Georgia, serif;
  background: #e8e8e8; color: #111; font-size: 14px;
}}
.viewer-bar {{
  background: #2d3748; color: #ccc; padding: 8px 20px;
  font-size: 12px; font-family: -apple-system, sans-serif;
  display: flex; justify-content: space-between;
}}
.page-wrap {{ display: flex; justify-content: center; padding: 28px 16px 40px; }}
.page {{
  background: #fffffe; width: 720px; min-height: 900px;
  padding: 64px 72px; box-shadow: 0 4px 24px rgba(0,0,0,0.22);
}}
.doc-type-label {{
  text-align: center; font-size: 11px; font-weight: 700;
  letter-spacing: 2px; text-transform: uppercase; color: #555; margin-bottom: 6px;
}}
.doc-title {{
  text-align: center; font-size: 20px; font-weight: 700;
  margin-bottom: 4px; letter-spacing: 0.3px;
}}
.doc-ref {{
  text-align: center; font-size: 12px; color: #666; margin-bottom: 20px;
}}
.divider {{ border: none; border-top: 2px solid #111; margin: 16px 0; }}
.meta-row {{
  display: flex; justify-content: space-between;
  font-size: 12px; color: #444; margin-bottom: 6px;
}}
.parties-section {{ margin: 16px 0; }}
.parties-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; color: #666; margin-bottom: 8px; }}
.party-row {{ display: flex; gap: 16px; margin-bottom: 4px; font-size: 13px; }}
.party-role {{ font-weight: 700; min-width: 140px; color: #333; }}
.party-name {{ color: #111; }}
.whereas {{
  font-size: 13px; line-height: 1.7; margin: 16px 0;
  font-style: italic; color: #333;
}}
.clause {{ margin-bottom: 14px; }}
.clause-head {{
  font-size: 13px; font-weight: 700; margin-bottom: 4px; color: #111;
}}
.clause-text {{ font-size: 13px; line-height: 1.75; color: #222; }}
.sig-section {{
  display: flex; flex-wrap: wrap; gap: 32px; margin-top: 40px; padding-top: 20px;
  border-top: 1px solid #ccc;
}}
.sig-block {{ flex: 1; min-width: 180px; }}
.sig-line {{ border-bottom: 1px solid #333; margin-bottom: 4px; height: 24px; }}
.sig-role {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
.sig-name {{ font-size: 13px; font-weight: 700; margin-top: 2px; }}
.sig-date {{ font-size: 11px; color: #888; margin-top: 2px; }}
.notary {{
  margin-top: 20px; font-size: 11px; color: #666;
  font-style: italic; border-top: 1px solid #eee; padding-top: 12px;
}}
</style></head><body>
{_chrome_bar(url)}
<div class="viewer-bar">
  <span>{type_label}</span><span>{ref}</span>
</div>
<div class="page-wrap"><div class="page">
  <div class="doc-type-label">{type_label}</div>
  <div class="doc-title">{title}</div>
  <div class="doc-ref">{ref}</div>
  <hr class="divider">
  <div class="meta-row">
    <span>Jurisdiction: {jx}</span>
    <span>Date: {date}</span>
  </div>
  <div class="parties-section">
    <div class="parties-title">Parties</div>
    {parties_html}
  </div>
  <p class="whereas">WHEREAS the parties hereto desire to set forth their mutual agreements and obligations as described herein, and for good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the parties agree as follows:</p>
  <hr class="divider">
  {clauses_html}
  <div class="sig-section">{sigs_html}</div>
  {f'<div class="notary">{notary}</div>' if notary else ''}
</div></div>
</body></html>"""


# ---------------------------------------------------------------------------
# IDENTITY TEMPLATE
# ---------------------------------------------------------------------------

def _html_identity(item: dict) -> str:
    doc_type   = item.get("document_type", "passport")
    authority  = _e(item.get("issuing_authority", ""))
    surname    = _e(item.get("surname", ""))
    given      = _e(item.get("given_names", ""))
    dob        = _e(item.get("dob", ""))
    doc_num    = _e(item.get("document_number", ""))
    nat        = _e(item.get("nationality_or_state", ""))
    issue      = _e(item.get("issue_date", ""))
    expiry     = _e(item.get("expiry_date", ""))
    additional = item.get("additional_fields", {})
    sec_feats  = item.get("security_features", [])
    url        = "gov.valdoria.id/verify"

    type_label = {
        "passport":        "PASSPORT",
        "drivers_licence": "DRIVER'S LICENCE",
        "national_id":     "NATIONAL IDENTITY CARD",
        "employee_id":     "EMPLOYEE IDENTIFICATION",
        "insurance_card":  "INSURANCE CARD",
    }.get(doc_type, "IDENTITY DOCUMENT")

    # Colour scheme per doc type
    colours = {
        "passport":        ("#1a3a5c", "#eaf0fb"),
        "drivers_licence": ("#2d5016", "#edf7e6"),
        "national_id":     ("#5c1a1a", "#fbeaea"),
        "employee_id":     ("#2d3748", "#f0f2f5"),
        "insurance_card":  ("#1a4a5c", "#e8f4f8"),
    }
    hdr_bg, field_bg = colours.get(doc_type, ("#2d3748", "#f0f2f5"))

    # Additional fields rows
    add_rows = ""
    for k, v in additional.items():
        if v:
            label = k.replace("_", " ").title()
            add_rows += f"""
            <div class="field-row">
              <div class="field-label">{_e(label)}</div>
              <div class="field-value">{_e(str(v))}</div>
            </div>"""

    mrz1 = _e(additional.get("mrz_line1", ""))
    mrz2 = _e(additional.get("mrz_line2", ""))
    mrz_html = ""
    if mrz1 or mrz2:
        mrz_html = f"""
        <div class="mrz">
          <div class="mrz-label">MACHINE READABLE ZONE</div>
          <div class="mrz-line">{mrz1}</div>
          <div class="mrz-line">{mrz2}</div>
        </div>"""

    sec_html = "".join(
        f'<span class="sec-badge">{_e(f)}</span>' for f in sec_feats
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{type_label}</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #d0d0d0; }}
.page-wrap {{ display: flex; justify-content: center; padding: 32px 16px; }}
.id-card {{
  background: white; width: 640px;
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 6px 32px rgba(0,0,0,0.30);
}}
.id-header {{
  background: {hdr_bg}; color: white; padding: 18px 24px;
  display: flex; justify-content: space-between; align-items: center;
}}
.id-type {{ font-size: 16px; font-weight: 800; letter-spacing: 2px; }}
.id-authority {{ font-size: 10px; opacity: 0.75; margin-top: 3px; max-width: 300px; }}
.id-doc-num {{ font-size: 13px; font-family: 'Courier New', monospace;
  background: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 4px; }}
.id-body {{ display: flex; }}
.id-photo {{
  width: 140px; min-height: 180px; background: {field_bg};
  display: flex; align-items: center; justify-content: center;
  border-right: 1px solid #ddd; flex-shrink: 0;
}}
.photo-placeholder {{
  width: 90px; height: 110px; background: #ccc; border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; color: #888; text-align: center; line-height: 1.4;
}}
.id-fields {{ flex: 1; padding: 16px 20px; background: {field_bg}; }}
.field-row {{ margin-bottom: 10px; }}
.field-label {{
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; color: #666; margin-bottom: 2px;
}}
.field-value {{ font-size: 14px; font-weight: 600; color: #111; }}
.name-row {{ display: flex; gap: 20px; margin-bottom: 10px; }}
.id-footer {{
  background: {hdr_bg}; padding: 10px 24px;
  display: flex; justify-content: space-between; align-items: center;
}}
.footer-dates {{ display: flex; gap: 32px; }}
.footer-date-item label {{
  font-size: 9px; text-transform: uppercase; letter-spacing: 1px;
  color: rgba(255,255,255,0.6); display: block;
}}
.footer-date-item span {{ font-size: 13px; font-weight: 700; color: white; }}
.mrz {{
  background: #f8f8f8; padding: 12px 16px;
  border-top: 1px solid #eee;
}}
.mrz-label {{ font-size: 9px; color: #aaa; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 6px; }}
.mrz-line {{
  font-family: 'Courier New', monospace; font-size: 12px;
  letter-spacing: 2px; color: #333; margin-bottom: 2px;
  word-break: break-all;
}}
.sec-features {{
  padding: 8px 20px; background: #fafafa;
  border-top: 1px solid #eee; display: flex; flex-wrap: wrap; gap: 6px;
}}
.sec-badge {{
  font-size: 9px; background: #e8e8e8; color: #555;
  padding: 2px 7px; border-radius: 10px; letter-spacing: 0.3px;
}}
</style></head><body>
{_chrome_bar(url)}
<div class="page-wrap"><div class="id-card">
  <div class="id-header">
    <div>
      <div class="id-type">{type_label}</div>
      <div class="id-authority">{authority}</div>
    </div>
    <div class="id-doc-num">{doc_num}</div>
  </div>
  <div class="id-body">
    <div class="id-photo">
      <div class="photo-placeholder">PHOTO</div>
    </div>
    <div class="id-fields">
      <div class="name-row">
        <div class="field-row">
          <div class="field-label">Surname</div>
          <div class="field-value">{surname}</div>
        </div>
        <div class="field-row">
          <div class="field-label">Given Names</div>
          <div class="field-value">{given}</div>
        </div>
      </div>
      <div class="field-row">
        <div class="field-label">Date of Birth</div>
        <div class="field-value">{dob}</div>
      </div>
      <div class="field-row">
        <div class="field-label">Nationality / State</div>
        <div class="field-value">{nat}</div>
      </div>
      {add_rows}
    </div>
  </div>
  <div class="id-footer">
    <div class="footer-dates">
      <div class="footer-date-item"><label>Issue Date</label><span>{issue}</span></div>
      <div class="footer-date-item"><label>Expiry Date</label><span>{expiry}</span></div>
    </div>
  </div>
  {mrz_html}
  {f'<div class="sec-features">{sec_html}</div>' if sec_html else ''}
</div></div>
</body></html>"""


# ---------------------------------------------------------------------------
# COMMUNICATIONS TEMPLATE
# ---------------------------------------------------------------------------

def _html_communications(item: dict) -> str:
    comm_type    = item.get("comm_type", "sms_thread")
    platform     = _e(item.get("platform", "Messages"))
    participants = item.get("participants", [])
    subject      = item.get("subject", "")
    timestamp    = _e(item.get("timestamp", ""))
    messages     = item.get("messages", [])
    url          = "messages.app/thread"

    # Identify "self" participant name for bubble alignment
    self_name = next(
        (p["name"] for p in participants if p.get("role") == "self"),
        participants[0]["name"] if participants else "Me",
    )
    other_name = next(
        (p["name"] for p in participants if p.get("role") != "self"),
        "Contact",
    )

    if comm_type == "email":
        return _html_email(item, platform, participants, subject,
                           timestamp, messages, self_name, url)

    # SMS / DM bubble layout
    bubbles_html = ""
    for msg in messages:
        is_self  = msg.get("sender") == self_name
        side     = "self" if is_self else "other"
        read_dot = "" if msg.get("read", True) else '<span class="unread-dot"></span>'
        bubbles_html += f"""
        <div class="bubble-row {side}">
          <div class="bubble {side}">{_e(msg.get('text',''))}</div>
          <div class="msg-meta">{_e(msg.get('time',''))} {read_dot}</div>
        </div>"""

    # Platform colour
    platform_colours = {
        "WhatsApp":    ("#075e54", "#dcf8c6"),
        "Telegram":    ("#2ca5e0", "#e3f4fd"),
        "Signal":      ("#2c6bed", "#e8effd"),
        "Instagram DM":("#833ab4", "#f0e6ff"),
        "Slack":       ("#4a154b", "#f0e6f0"),
        "Discord":     ("#5865f2", "#edeeff"),
        "Messages":    ("#1c8ef9", "#e5f3ff"),
    }
    hdr_bg, self_bg = platform_colours.get(
        item.get("platform",""), ("#1c8ef9", "#e5f3ff")
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{platform}</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f0f0f0; }}
.msg-app {{ max-width: 420px; margin: 0 auto; background: white;
  min-height: 100vh; display: flex; flex-direction: column; }}
.app-header {{
  background: {hdr_bg}; color: white; padding: 12px 16px;
  display: flex; align-items: center; gap: 12px; position: sticky; top: 0;
}}
.contact-avatar {{
  width: 36px; height: 36px; border-radius: 50%;
  background: rgba(255,255,255,0.3);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 14px; flex-shrink: 0;
}}
.contact-name {{ font-weight: 600; font-size: 15px; }}
.contact-status {{ font-size: 11px; opacity: 0.8; }}
.timestamp-bar {{
  text-align: center; font-size: 11px; color: #999;
  padding: 12px 0 4px;
}}
.thread {{
  flex: 1; padding: 8px 12px; display: flex; flex-direction: column; gap: 4px;
}}
.bubble-row {{ display: flex; flex-direction: column; }}
.bubble-row.self {{ align-items: flex-end; }}
.bubble-row.other {{ align-items: flex-start; }}
.bubble {{
  max-width: 72%; padding: 9px 13px; border-radius: 18px;
  font-size: 14px; line-height: 1.45; word-break: break-word;
}}
.bubble.self {{
  background: {self_bg}; color: #111;
  border-bottom-right-radius: 4px;
}}
.bubble.other {{
  background: #f0f0f0; color: #111;
  border-bottom-left-radius: 4px;
}}
.msg-meta {{
  font-size: 10px; color: #aaa; margin: 2px 4px 6px;
  display: flex; align-items: center; gap: 4px;
}}
.unread-dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: {hdr_bg}; display: inline-block;
}}
.input-bar {{
  background: #f9f9f9; border-top: 1px solid #e8e8e8;
  padding: 10px 12px; display: flex; gap: 8px; align-items: center;
}}
.input-field {{
  flex: 1; background: white; border: 1px solid #ddd;
  border-radius: 20px; padding: 8px 14px; font-size: 13px; color: #ccc;
}}
</style></head><body>
{_chrome_bar(url)}
<div class="msg-app">
  <div class="app-header">
    <div class="contact-avatar">{_e(other_name[0].upper())}</div>
    <div>
      <div class="contact-name">{_e(other_name)}</div>
      <div class="contact-status">{platform}</div>
    </div>
  </div>
  <div class="timestamp-bar">{timestamp}</div>
  <div class="thread">{bubbles_html}</div>
  <div class="input-bar">
    <div class="input-field">Message</div>
  </div>
</div>
</body></html>"""


def _html_email(item, platform, participants, subject,
                timestamp, messages, self_name, url):
    """Email layout — separate from bubble layout."""
    to_names   = ", ".join(
        _e(p["name"]) for p in participants if p.get("role") != "self"
    )
    from_name  = _e(self_name)
    subject_e  = _e(subject or "(No subject)")

    msgs_html = ""
    for i, msg in enumerate(messages):
        is_self   = msg.get("sender") == self_name
        bg        = "#f9fafb" if i % 2 == 0 else "#ffffff"
        sender_e  = _e(msg.get("sender", ""))
        time_e    = _e(msg.get("time", ""))
        text_e    = _e(msg.get("text", ""))
        msgs_html += f"""
        <div class="email-msg" style="background:{bg}">
          <div class="email-msg-header">
            <span class="email-sender">{sender_e}</span>
            <span class="email-time">{time_e}</span>
          </div>
          <div class="email-body">{text_e}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{platform} — {subject_e}</title>
<style>
{_CHROME_CSS}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f3f4f6; }}
.email-client {{ max-width: 760px; margin: 0 auto; background: white;
  min-height: 100vh; }}
.email-header {{
  background: #1a73e8; color: white; padding: 14px 20px;
  display: flex; align-items: center; gap: 10px;
}}
.email-app-name {{ font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }}
.thread-header {{ padding: 20px 24px; border-bottom: 1px solid #e8e8e8; }}
.thread-subject {{ font-size: 20px; font-weight: 700; margin-bottom: 8px; }}
.thread-meta {{ font-size: 12px; color: #666; }}
.email-msg {{ padding: 16px 24px; border-bottom: 1px solid #f0f0f0; }}
.email-msg-header {{
  display: flex; justify-content: space-between;
  margin-bottom: 8px;
}}
.email-sender {{ font-weight: 600; font-size: 13px; }}
.email-time   {{ font-size: 11px; color: #aaa; }}
.email-body   {{ font-size: 14px; line-height: 1.7; color: #333; white-space: pre-wrap; }}
</style></head><body>
{_chrome_bar(url)}
<div class="email-client">
  <div class="email-header">
    <div class="email-app-name">{platform}</div>
  </div>
  <div class="thread-header">
    <div class="thread-subject">{subject_e}</div>
    <div class="thread-meta">
      To: {to_names} &nbsp;·&nbsp; {timestamp}
    </div>
  </div>
  {msgs_html}
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# TEMPLATE DISPATCH
# ---------------------------------------------------------------------------

_HTML_BUILDERS = {
    "banking":        _html_banking,
    "medical":        _html_medical,
    "news":           _html_news,
    "copyright":      _html_copyright,
    "legal":          _html_legal,
    "identity":       _html_identity,
    "communications": _html_communications,
}

SUPPORTED_CATEGORIES = list(_HTML_BUILDERS.keys())


def build_html(item: dict, category: str) -> str:
    """Return the HTML string for an item. Raises KeyError for unknown category."""
    builder = _HTML_BUILDERS[category]
    return builder(item)


# ---------------------------------------------------------------------------
# WORD-BOX LINE GROUPING
# ---------------------------------------------------------------------------

_LINE_GROUP_THRESHOLD_PX = 4


def _group_into_lines(raw_boxes: list, H: int) -> list:
    boxes = [b for b in raw_boxes if b["top"] < H]
    if not boxes:
        return []
    boxes.sort(key=lambda b: b["top"])
    lines = []
    cur = [boxes[0]]
    for b in boxes[1:]:
        if b["top"] - cur[0]["top"] > _LINE_GROUP_THRESHOLD_PX:
            lines.append(sorted(cur, key=lambda x: x["left"]))
            cur = [b]
        else:
            cur.append(b)
    lines.append(sorted(cur, key=lambda x: x["left"]))
    return lines


# ---------------------------------------------------------------------------
# ASYNC RENDERER CLASS
# ---------------------------------------------------------------------------

class PlaywrightRenderer:
    """
    Async context manager that keeps one Chromium browser open for the
    lifetime of the render job. Much faster than opening a new browser
    per image.

    Usage
    -----
        async with PlaywrightRenderer() as renderer:
            for item in items:
                png_bytes = await renderer.render(item, category, rng=rng)
                pathlib.Path(f"{item['_content_id']}.png").write_bytes(png_bytes)

    Parameters
    ----------
    width, height : int | None
        Fixed viewport size. If None, size is sampled from rng per call.
    no_sandbox : bool
        Pass --no-sandbox to Chromium. Required on Colab and most HPC nodes.
        Defaults to True — safe to leave on everywhere.
    """

    def __init__(
        self,
        width:      Optional[int] = None,
        height:     Optional[int] = None,
        no_sandbox: bool          = True,
    ):
        self._fixed_w   = width
        self._fixed_h   = height
        self._no_sandbox = no_sandbox
        self._playwright = None
        self._browser    = None

    async def __aenter__(self) -> "PlaywrightRenderer":
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        args = ["--disable-dev-shm-usage"]
        if self._no_sandbox:
            args.append("--no-sandbox")
        self._browser = await self._playwright.chromium.launch(
            headless=True, args=args,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def render(
        self,
        item:     dict,
        category: str,
        rng:      Optional[random.Random] = None,
    ) -> bytes:
        """
        Render one item to PNG bytes.

        Parameters
        ----------
        item     : content bank item dict (with _category, _content_id, …)
        category : one of SUPPORTED_CATEGORIES
        rng      : seeded Random for viewport size sampling.
                   If None, uses a fresh unseeded RNG (non-reproducible).

        Returns
        -------
        bytes  — PNG image data
        """
        if rng is None:
            rng = random.Random()

        if self._fixed_w and self._fixed_h:
            w, h = self._fixed_w, self._fixed_h
        else:
            w, h = _pick_viewport(rng)

        html = build_html(item, category)

        page = await self._browser.new_page(viewport={"width": w, "height": h})
        try:
            await page.set_content(html, wait_until="domcontentloaded")
            png_bytes = await page.screenshot(full_page=False)
            raw_boxes = await page.evaluate("""() => {
    const skip = new Set(['STYLE', 'SCRIPT', 'NOSCRIPT']);
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        { acceptNode(node) {
            let el = node.parentElement;
            while (el) {
                if (skip.has(el.tagName)) return NodeFilter.FILTER_REJECT;
                el = el.parentElement;
            }
            return NodeFilter.FILTER_ACCEPT;
        }}
    );
    const spans = [];
    let node;
    while ((node = walker.nextNode())) {
        const words = node.textContent.split(/\\s+/).filter(w => w.length > 0);
        if (!words.length) continue;
        const parts = node.textContent.split(/(\\s+)/);
        const frag = document.createDocumentFragment();
        for (const part of parts) {
            if (!part || /^\\s+$/.test(part)) {
                frag.appendChild(document.createTextNode(part));
            } else {
                const span = document.createElement('span');
                span.className = '__wb';
                span.textContent = part;
                frag.appendChild(span);
                spans.push(span);
            }
        }
        node.parentNode.replaceChild(frag, node);
    }
    return spans.map(s => {
        const r = s.getBoundingClientRect();
        return { word: s.textContent, top: r.top, left: r.left, right: r.right, bottom: r.bottom };
    });
}""")
        finally:
            await page.close()

        word_boxes = _group_into_lines(raw_boxes, h)
        return png_bytes, word_boxes


# ---------------------------------------------------------------------------
# SYNC CONVENIENCE WRAPPER  (testing / one-off use)
# ---------------------------------------------------------------------------

def render_sync(
    item:       dict,
    category:   str,
    rng:        Optional[random.Random] = None,
    width:      Optional[int]           = None,
    height:     Optional[int]           = None,
    no_sandbox: bool                    = True,
) -> bytes:
    """
    Synchronous wrapper around PlaywrightRenderer.
    Opens and closes the browser for every call — do not use in a loop.
    Intended for testing and one-off renders only.
    """
    async def _run():
        async with PlaywrightRenderer(
            width=width, height=height, no_sandbox=no_sandbox
        ) as renderer:
            return await renderer.render(item, category, rng=rng)

    return asyncio.run(_run())