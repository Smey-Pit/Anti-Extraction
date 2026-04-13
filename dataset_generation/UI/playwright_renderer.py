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
# TEMPLATE DISPATCH
# ---------------------------------------------------------------------------

_HTML_BUILDERS = {
    "banking":   _html_banking,
    "medical":   _html_medical,
    "news":      _html_news,
    "copyright": _html_copyright,
}

SUPPORTED_CATEGORIES = list(_HTML_BUILDERS.keys())


def build_html(item: dict, category: str) -> str:
    """Return the HTML string for an item. Raises KeyError for unknown category."""
    builder = _HTML_BUILDERS[category]
    return builder(item)


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
        finally:
            await page.close()

        return png_bytes


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