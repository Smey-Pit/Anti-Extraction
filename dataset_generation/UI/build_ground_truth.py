"""
build_ground_truth.py
=====================
Converts labels_pil.jsonl (produced by render_images.py) into per-image
ground-truth JSON files consumed by vlm_suppress.metrics.

The GT schema maps each domain's raw_content fields to typed Entity objects
and extracts content_blocks from the raw_content prose fields.

Output layout (mirrors the image directory):
    data/ui_dataset/
    ├── images/pil/
    │   ├── banking_0000.png
    │   └── ...
    └── ground_truth/pil/
        ├── banking_0000.json
        └── ...

A manifest CSV is also written to data/ui_dataset/manifest.csv.

Usage
-----
    python build_ground_truth.py \\
        --labels data/ui_dataset/labels_pil.jsonl \\
        --out-dir data/ui_dataset/ground_truth/pil

    # Dry-run: print first GT of each domain without writing
    python build_ground_truth.py --labels ... --out-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific entity extractors
#
# Each function receives raw_content dict (from the labels file) and
# returns (entities, content_blocks) matching the GT schema.
#
# Entity format: {"label": str, "type": str, "value": str}
# Content block: {"label": str, "text": str}
# ─────────────────────────────────────────────────────────────────────────────

def _e(label: str, type_: str, value, question: str = "") -> dict | None:
    """Build one entity dict. Returns None if value is missing/empty."""
    v = str(value).strip() if value is not None else ""
    if not v:
        return None
    d = {"label": label, "type": type_, "value": v}
    if question:
        d["question"] = question
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Canonical question registry
#
# One question per entity label, per domain. Used for Binding Accuracy
# (Threat 1): the eval harness issues answer_query(image, question) and
# scores the response with the entity's type-appropriate comparator.
#
# Design rules:
#   - Singular fields: "What is the X?" → model should return the value directly
#   - List fields: "List all X" → model returns a list, we search for each value
#   - Questions are domain-agnostic where possible (patient name vs account holder)
#   - Keep questions short and unambiguous
# ─────────────────────────────────────────────────────────────────────────────

FIELD_QUESTIONS: dict[str, dict[str, str]] = {
    "banking": {
        "bank_name":        "What is the name of the bank?",
        "account_holder":   "What is the account holder name?",
        "account_number":   "What is the account number?",
        "account_type":     "What is the account type?",
        "statement_period": "What is the statement period?",
        "opening_balance":  "What is the opening balance?",
        "closing_balance":  "What is the closing balance?",
    },
    "medical": {
        "hospital_name":       "What is the name of the hospital?",
        "patient_name":        "What is the patient name?",
        "dob":                 "What is the patient date of birth?",
        "patient_id":          "What is the patient ID?",
        "visit_date":          "What is the visit date?",
        "attending_physician": "Who is the attending physician?",
        "chief_complaint":     "What is the chief complaint?",
        "diagnosis":           "What is the diagnosis?",
        # list entities — one question covers all instances
        "medication_1":        "List all prescribed medications.",
        "medication_2":        "List all prescribed medications.",
        "medication_3":        "List all prescribed medications.",
        "lab_1_test":          "List all lab tests performed.",
        "lab_1_value":         "List all lab test results.",
        "lab_2_test":          "List all lab tests performed.",
        "lab_2_value":         "List all lab test results.",
        "lab_3_test":          "List all lab tests performed.",
        "lab_3_value":         "List all lab test results.",
    },
    "news": {
        "outlet_name":  "What is the name of the news outlet?",
        "headline":     "What is the article headline?",
        "byline":       "Who wrote the article?",
        "dateline":     "What is the dateline of the article?",
        "category_tag": "What is the article category or section?",
        "tag_1":        "List all topic tags for this article.",
        "tag_2":        "List all topic tags for this article.",
        "tag_3":        "List all topic tags for this article.",
        "tag_4":        "List all topic tags for this article.",
    },
    "copyright": {
        "title":          "What is the title of the work?",
        "author":         "Who is the author?",
        "publisher":      "Who is the publisher?",
        "copyright_line": "What is the copyright notice?",
        "page_number":    "What is the page number?",
        "chapter_scene":  "What is the chapter or scene title?",
    },
    "legal": {
        "title":           "What is the title of this legal document?",
        "jurisdiction":    "What is the jurisdiction?",
        "date":            "What is the date of this document?",
        "case_ref_number": "What is the case or reference number?",
        # party and signatory labels are dynamic — handled in extractor
    },
    "identity": {
        "document_type":     "What type of identity document is this?",
        "issuing_authority":  "What authority issued this document?",
        "surname":            "What is the surname?",
        "given_names":        "What are the given names?",
        "dob":                "What is the date of birth?",
        "document_number":    "What is the document number?",
        "nationality_state":  "What is the nationality or state?",
        "issue_date":         "What is the issue date?",
        "expiry_date":        "What is the expiry date?",
        "sex":                "What is the sex listed on the document?",
        "place_of_birth":     "What is the place of birth?",
        "mrz_line1":          "What does the first machine readable zone line say?",
        "address":            "What is the address on the document?",
        "licence_class":      "What is the licence class?",
        "employee_id":        "What is the employee ID?",
        "department":         "What is the department?",
        "job_title":          "What is the job title?",
        "organisation":       "What is the organisation?",
        "member_id":          "What is the member ID?",
        "group_number":       "What is the group number?",
        "plan_name":          "What is the plan name?",
        "primary_care_provider": "Who is the primary care provider?",
    },
    "communications": {
        "platform":      "What platform or app is this conversation on?",
        "subject":       "What is the subject line of this message?",
        # participant labels are dynamic — handled in extractor
    },
}


def _q(domain: str, label: str) -> str:
    """Look up the canonical question for a field label in a domain."""
    return FIELD_QUESTIONS.get(domain, {}).get(label, "")


def _entities_banking(rc: dict) -> tuple[list, list]:
    D = "banking"
    entities = [
        _e("bank_name",        "short_phrase", rc.get("bank_name"),        _q(D, "bank_name")),
        _e("account_holder",   "name",         rc.get("account_holder"),   _q(D, "account_holder")),
        _e("account_number",   "digit_seq",    rc.get("account_number"),   _q(D, "account_number")),
        _e("account_type",     "short_phrase", rc.get("account_type"),     _q(D, "account_type")),
        _e("statement_period", "date_range",   rc.get("statement_period"), _q(D, "statement_period")),
        _e("opening_balance",  "amount",       rc.get("opening_balance"),  _q(D, "opening_balance")),
        _e("closing_balance",  "amount",       rc.get("closing_balance"),  _q(D, "closing_balance")),
    ]
    txns = rc.get("transactions", [])
    if txns:
        lines = []
        for t in txns:
            lines.append(
                f"{t.get('date','')} {t.get('description','')} "
                f"{t.get('amount','')} {t.get('running_balance','')}"
            )
        content_blocks = [{"label": "transaction_history", "text": "\n".join(lines)}]
    else:
        content_blocks = []
    return [e for e in entities if e], content_blocks


def _entities_medical(rc: dict) -> tuple[list, list]:
    D = "medical"
    entities = [
        _e("hospital_name",       "short_phrase", rc.get("hospital_name"),       _q(D, "hospital_name")),
        _e("patient_name",        "name",         rc.get("patient_name"),        _q(D, "patient_name")),
        _e("dob",                 "date",         rc.get("dob"),                 _q(D, "dob")),
        _e("patient_id",          "digit_seq",    rc.get("patient_id"),          _q(D, "patient_id")),
        _e("visit_date",          "date",         rc.get("visit_date"),          _q(D, "visit_date")),
        _e("attending_physician", "name",         rc.get("attending_physician"), _q(D, "attending_physician")),
        _e("chief_complaint",     "short_phrase", rc.get("chief_complaint"),     _q(D, "chief_complaint")),
        _e("diagnosis",           "short_phrase", rc.get("diagnosis"),           _q(D, "diagnosis")),
    ]

    # Medications → one entity per medication
    for i, med in enumerate(rc.get("medications", [])):
        med_str = med if isinstance(med, str) else json.dumps(med)
        label = f"medication_{i+1}"
        entities.append(_e(label, "short_phrase", med_str, _q(D, label)))

    # Lab results → one entity per result
    for i, lab in enumerate(rc.get("lab_results", [])):
        if isinstance(lab, dict):
            test  = lab.get("test", f"lab_{i+1}")
            value = lab.get("value", "")
            entities.append(_e(f"lab_{i+1}_test",  "short_phrase", test,  _q(D, f"lab_{i+1}_test")))
            entities.append(_e(f"lab_{i+1}_value", "short_phrase", value, _q(D, f"lab_{i+1}_value")))

    content_blocks = []
    clinical = rc.get("clinical_notes", "")
    if clinical:
        content_blocks.append({"label": "clinical_notes", "text": clinical})
    follow_up = rc.get("follow_up", "")
    if follow_up:
        content_blocks.append({"label": "follow_up", "text": follow_up})

    return [e for e in entities if e], content_blocks


def _entities_news(rc: dict) -> tuple[list, list]:
    D = "news"
    entities = [
        _e("outlet_name",  "short_phrase", rc.get("outlet_name"),  _q(D, "outlet_name")),
        _e("headline",     "short_phrase", rc.get("headline"),     _q(D, "headline")),
        _e("byline",       "name",         rc.get("byline"),       _q(D, "byline")),
        _e("dateline",     "short_phrase", rc.get("dateline"),     _q(D, "dateline")),
        _e("category_tag", "short_phrase", rc.get("category_tag"), _q(D, "category_tag")),
    ]
    for i, tag in enumerate(rc.get("tags", [])):
        label = f"tag_{i+1}"
        entities.append(_e(label, "short_phrase", tag, _q(D, label)))

    content_blocks = []
    lead = rc.get("lead_paragraph", "")
    if lead:
        content_blocks.append({"label": "lead_paragraph", "text": lead})
    for i, para in enumerate(rc.get("body_paragraphs", [])):
        if para:
            content_blocks.append({"label": f"body_paragraph_{i+1}", "text": para})
    pull = rc.get("pull_quote", "")
    if pull:
        content_blocks.append({"label": "pull_quote", "text": pull})

    return [e for e in entities if e], content_blocks


def _entities_copyright(rc: dict) -> tuple[list, list]:
    D = "copyright"
    entities = [
        _e("title",          "short_phrase", rc.get("title"),             _q(D, "title")),
        _e("author",         "name",         rc.get("author"),            _q(D, "author")),
        _e("publisher",      "short_phrase", rc.get("publisher"),         _q(D, "publisher")),
        _e("copyright_line", "short_phrase", rc.get("copyright_line"),    _q(D, "copyright_line")),
        _e("page_number",    "short_phrase", rc.get("page_number"),       _q(D, "page_number")),
        _e("chapter_scene",  "short_phrase", rc.get("chapter_or_scene"),  _q(D, "chapter_scene")),
        # content_type is internal metadata, excluded from GT entities
    ]
    content_blocks = []
    body = rc.get("content", "")
    if body:
        content_blocks.append({"label": "body_text", "text": body})
    return [e for e in entities if e], content_blocks


def _entities_legal(rc: dict) -> tuple[list, list]:
    D = "legal"
    entities = [
        # document_type is internal metadata, excluded from GT entities
        _e("title",           "short_phrase", rc.get("title"),                _q(D, "title")),
        _e("jurisdiction",    "short_phrase", rc.get("jurisdiction"),         _q(D, "jurisdiction")),
        _e("date",            "date",         rc.get("date"),                 _q(D, "date")),
        _e("case_ref_number", "short_phrase", rc.get("case_or_ref_number"),   _q(D, "case_ref_number")),
    ]
    # Parties — dynamic labels, use a shared question
    for i, party in enumerate(rc.get("parties", [])):
        if isinstance(party, dict):
            name = party.get("name", "")
            role = party.get("role", f"party_{i+1}")
            label = f"party_{i+1}_{role.lower().replace(' ','_')}"
            entities.append(_e(label, "name", name,
                               "Who are the parties involved in this document?"))

    # Signature block — dynamic labels
    for i, sig in enumerate(rc.get("signature_block", [])):
        if isinstance(sig, dict) and sig.get("name"):
            entities.append(_e(f"signatory_{i+1}", "name", sig["name"],
                               "Who are the signatories of this document?"))

    content_blocks = []
    for clause in rc.get("clauses", []):
        if isinstance(clause, dict):
            num     = clause.get("number", "")
            heading = clause.get("heading", "")
            text    = clause.get("text", "")
            if text:
                label = f"clause_{num}_{heading}".replace(" ", "_")[:50]
                content_blocks.append({"label": label, "text": text})
    notary = rc.get("notary_note", "")
    if notary:
        content_blocks.append({"label": "notary_note", "text": notary})

    return [e for e in entities if e], content_blocks


def _entities_identity(rc: dict) -> tuple[list, list]:
    D = "identity"
    entities = [
        _e("document_type",     "short_phrase", rc.get("document_type"),         _q(D, "document_type")),
        _e("issuing_authority", "short_phrase", rc.get("issuing_authority"),     _q(D, "issuing_authority")),
        _e("surname",           "name",         rc.get("surname"),                _q(D, "surname")),
        _e("given_names",       "name",         rc.get("given_names"),            _q(D, "given_names")),
        _e("dob",               "date",         rc.get("dob"),                   _q(D, "dob")),
        _e("document_number",   "digit_seq",    rc.get("document_number"),        _q(D, "document_number")),
        _e("nationality_state", "short_phrase", rc.get("nationality_or_state"),   _q(D, "nationality_state")),
        _e("issue_date",        "date",         rc.get("issue_date"),             _q(D, "issue_date")),
        _e("expiry_date",       "date",         rc.get("expiry_date"),            _q(D, "expiry_date")),
    ]

    additional = rc.get("additional_fields", {})
    if isinstance(additional, dict):
        for field_key, label, type_ in [
            ("sex",                   "sex",                   "short_phrase"),
            ("place_of_birth",        "place_of_birth",        "short_phrase"),
            ("mrz_line1",             "mrz_line1",             "short_phrase"),
            # mrz_line2 excluded — captured as content_block below
            ("address",               "address",               "short_phrase"),
            ("licence_class",         "licence_class",         "short_phrase"),
            ("employee_id",           "employee_id",           "digit_seq"),
            ("department",            "department",            "short_phrase"),
            ("job_title",             "job_title",             "short_phrase"),
            ("organisation",          "organisation",          "short_phrase"),
            ("member_id",             "member_id",             "digit_seq"),
            ("group_number",          "group_number",          "digit_seq"),
            ("plan_name",             "plan_name",             "short_phrase"),
            ("primary_care_provider", "primary_care_provider", "name"),
        ]:
            if field_key in additional:
                entities.append(_e(label, type_, additional[field_key], _q(D, label)))

    content_blocks = []
    additional = rc.get("additional_fields", {})
    if isinstance(additional, dict) and additional.get("mrz_line2"):
        content_blocks.append({
            "label": "mrz_line2",
            "text": str(additional["mrz_line2"]),
        })

    return [e for e in entities if e], content_blocks


def _entities_communications(rc: dict) -> tuple[list, list]:
    D = "communications"
    entities = [
        # comm_type is internal metadata, excluded from GT entities
        _e("platform", "short_phrase", rc.get("platform"), _q(D, "platform")),
    ]

    # Participants / sender / recipient
    for field_key in ("sender", "recipient", "from_address", "to_address"):
        val = rc.get(field_key)
        if val:
            entities.append(_e(field_key, "name", val,
                               "Who are the participants in this conversation?"))

    for i, participant in enumerate(rc.get("participants", [])):
        if isinstance(participant, dict):
            p_name = participant.get("name", "")
        else:
            p_name = str(participant)
        if p_name:
            entities.append(_e(f"participant_{i+1}", "name", p_name,
                               "Who are the participants in this conversation?"))

    subject = rc.get("subject", "")
    if subject:
        entities.append(_e("subject", "short_phrase", subject, _q(D, "subject")))

    content_blocks = []
    messages = rc.get("messages", [])
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            body = msg.get("body", msg.get("text", msg.get("content", "")))
            raw_sender = msg.get("sender", msg.get("from", ""))
            sender = raw_sender.get("name", "") if isinstance(raw_sender, dict) else raw_sender
            label = f"message_{i+1}_{sender}".replace(" ", "_")[:50]
            if body:
                content_blocks.append({"label": label, "text": str(body)})
        elif isinstance(msg, str) and msg:
            content_blocks.append({"label": f"message_{i+1}", "text": msg})

    body = rc.get("body", rc.get("email_body", ""))
    if body and not messages:
        content_blocks.append({"label": "email_body", "text": body})

    context = rc.get("context_summary", "")
    if context:
        content_blocks.append({"label": "context_summary", "text": context})

    return [e for e in entities if e], content_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_EXTRACTORS = {
    "banking":        _entities_banking,
    "medical":        _entities_medical,
    "news":           _entities_news,
    "copyright":      _entities_copyright,
    "legal":          _entities_legal,
    "identity":       _entities_identity,
    "communications": _entities_communications,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core converter
# ─────────────────────────────────────────────────────────────────────────────

def label_to_gt(label: dict) -> dict:
    """
    Convert one labels_pil.jsonl entry to a ground-truth dict.

    Raises ValueError if the entry is missing required fields.
    """
    image_id  = label.get("image_id")
    image_path = label.get("image_path")
    category  = label.get("category")

    if not image_id or not image_path or not category:
        raise ValueError(
            f"Label entry missing image_id/image_path/category: "
            f"{list(label.keys())}"
        )

    extractor = DOMAIN_EXTRACTORS.get(category)
    if extractor is None:
        raise ValueError(
            f"No extractor for category={category!r}. "
            f"Add one to DOMAIN_EXTRACTORS in build_ground_truth.py."
        )

    raw_content = label.get("raw_content", {})
    entities, content_blocks = extractor(raw_content)

    return {
        "doc_type":      category,
        "image_path":    image_path,
        "entities":      entities,
        "content_blocks": content_blocks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    labels_path = Path(args.labels)
    out_dir     = Path(args.out_dir)

    if not labels_path.exists():
        print(f"ERROR: labels file not found: {labels_path}", file=sys.stderr)
        return 2

    # Load all labels (jsonl: one JSON object per line)
    labels = []
    with open(labels_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                labels.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: line {lineno} is not valid JSON — skipping: {e}",
                      file=sys.stderr)

    print(f"Loaded {len(labels)} label entries from {labels_path}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Determine manifest path (parent of out_dir, or --manifest-dir)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir \
        else labels_path.parent
    manifest_path = manifest_dir / "manifest.csv"

    counts    = {}
    errors    = []
    manifest_rows = []
    seen_domains  = set()

    for label in labels:
        image_id = label.get("image_id", "<unknown>")
        category = label.get("category", "<unknown>")

        try:
            gt = label_to_gt(label)
        except Exception as e:
            errors.append((image_id, str(e)))
            print(f"  ERROR [{image_id}]: {e}", file=sys.stderr)
            continue

        gt_path = out_dir / f"{image_id}.json"

        # Dry-run: print first of each domain
        if args.dry_run:
            if category not in seen_domains:
                seen_domains.add(category)
                print()
                print(f"=== DOMAIN: {category} ===")
                print(json.dumps(gt, indent=2, ensure_ascii=False))
        else:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt, f, indent=2, ensure_ascii=False)

        counts[category] = counts.get(category, 0) + 1
        manifest_rows.append({
            "image_id":   image_id,
            "image_path": label.get("image_path", ""),
            "gt_path":    str(gt_path),
            "doc_type":   category,
        })

    if args.dry_run:
        print()
        print("[dry-run] No files written.")
    else:
        # Write manifest
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_id", "image_path", "gt_path", "doc_type"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest written: {manifest_path} ({len(manifest_rows)} rows)")

    print()
    print("=" * 52)
    print(f"Ground Truth Summary")
    print("=" * 52)
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<20} {n:>5} images")
    print(f"  {'TOTAL':<20} {sum(counts.values()):>5}")
    if errors:
        print(f"\n  ERRORS: {len(errors)}")
        for img_id, msg in errors[:10]:
            print(f"    {img_id}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")
    print()
    return 0 if not errors else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--labels", required=True,
        help="Path to labels_pil.jsonl produced by render_images.py",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for GT JSON files (e.g. data/ui_dataset/ground_truth/pil)",
    )
    p.add_argument(
        "--manifest-dir", default=None,
        help="Where to write manifest.csv. Defaults to the parent of --labels.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print first GT of each domain without writing any files.",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())