#!/usr/bin/env python3
"""
generate_participant_maps.py
────────────────────────────
1. Adds any missing participants from the Excel questionnaire to
   affectai-data-processing-seed/metadata/participants.tsv.
2. Generates a  participant_map.tsv  in every real session folder under
       F:/processed_data/sub-01/{session}/
       F:/processed_audio/sub-01/{session}/
   with columns:
       seat, participant_id, name, age, sex, handedness,
       english_proficiency, education,
       bfi44_e, bfi44_a, bfi44_c, bfi44_n, bfi44_o

Run from the repo root:
    python tools/generate_participant_maps.py
"""

import csv
import glob
import re
import unicodedata
from pathlib import Path

import openpyxl

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
METADATA_DIR = Path("affectai-data-processing-seed/metadata")
PTABLE_PATH  = METADATA_DIR / "participants.tsv"
SESSION_META = METADATA_DIR / "session_metadata_report.tsv"
XL_PATH      = glob.glob(str(METADATA_DIR / "*.xlsx"))[0]

OUTPUT_DIRS = [
    Path("F:/processed_data/sub-01"),
    Path("F:/processed_audio/sub-01"),
]

MAP_COLS = [
    "seat", "participant_id", "name", "age", "sex", "handedness",
    "english_proficiency", "education",
    "bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o",
]

# ──────────────────────────────────────────────────────────────────────────────
# BFI scoring (Big Five Inventory, 44 items)
# ──────────────────────────────────────────────────────────────────────────────
_LIKERT = {
    "Strongly disagree": 1, "Disagree": 2,
    "Neither disagree nor agree": 3,
    "Agree": 4, "Strongly agree": 5,
}

def _lk(items, n):
    return _LIKERT.get(items[n - 1])

def _rev(items, n):
    v = _lk(items, n)
    return (6 - v) if v else None

def bfi_scores(row):
    """Return (E, A, C, N, O) averages from a questionnaire row (tuple)."""
    items = row[14:58]

    def i(n): return _lk(items, n)
    def r(n): return _rev(items, n)

    buckets = {
        "E": [i(1), r(6), i(11), i(16), r(21), i(26), r(31), i(36)],
        "A": [r(2), i(7), r(12), i(17), i(22), r(27), i(32), r(37), r(42)],
        "C": [i(3), r(8), i(13), r(18), r(23), i(28), i(33), i(38), r(43)],
        "N": [i(4), r(9), i(14), i(19), r(24), i(29), r(34), i(39)],
        "O": [i(5), i(10), i(15), i(20), i(25), i(30), r(35), r(40), r(41), i(44)],
    }

    def avg(lst):
        vals = [v for v in lst if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else "n/a"

    return tuple(avg(buckets[k]) for k in "EANOC")


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────
def norm_edu(s):
    if not s:
        return "n/a"
    sl = s.lower()
    if "phd" in sl or "ph.d" in sl:
        return "phd"
    if "master" in sl:
        return "master"
    if "bachelor" in sl:
        return "bachelor"
    if "graduate certificate" in sl:
        return "graduate_certificate"
    if "professional certificate" in sl or "higher professional" in sl:
        return "professional_certificate"
    if sl.strip() == "ap":
        return "ap"
    return sl.strip()


def norm_hand(s):
    if not s:
        return "n/a"
    sl = s.lower()
    if "ambidextrous" in sl:
        return "ambidextrous"
    if "left" in sl:
        return "left"
    if "right" in sl:
        return "right"
    return sl


def norm_sex(s):
    if not s:
        return "n/a"
    sl = s.lower()
    if "female" in sl:
        return "female"
    if "male" in sl:
        return "male"
    return sl


def norm_age(v):
    """Return integer string or 'n/a' for ranges / missing."""
    if v is None:
        return "n/a"
    try:
        return str(int(v))
    except (ValueError, TypeError):
        return "n/a"  # e.g. "45-54" ranges


def norm_name(s):
    """Lower-case, remove combining diacritics, strip."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


# ──────────────────────────────────────────────────────────────────────────────
# Load Excel questionnaire
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading Excel: {XL_PATH}")
wb = openpyxl.load_workbook(XL_PATH)
ws = wb.active
xl_rows = list(ws.iter_rows(values_only=True))

xl_by_name: dict[str, dict] = {}  # normalised_name → demographic dict
for xl_row in xl_rows[1:]:
    name = xl_row[4]
    if not name:
        continue
    e, a, c, n, o = bfi_scores(xl_row)
    xl_by_name[norm_name(name)] = dict(
        xl_name=name,
        age=norm_age(xl_row[8] if xl_row[8] is not None else xl_row[7]),
        sex=norm_sex(xl_row[9]),
        hand=norm_hand(xl_row[10]),
        english=xl_row[12] or "n/a",
        edu=norm_edu(xl_row[13]),
        e=e, a=a, c=c, n=n, o=o,
    )

print(f"  {len(xl_by_name)} questionnaire entries loaded.")

# ──────────────────────────────────────────────────────────────────────────────
# Load existing participants.tsv
# ──────────────────────────────────────────────────────────────────────────────
with open(PTABLE_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    ptable: list[dict] = list(reader)
    fieldnames = list(reader.fieldnames)

existing_ids   = {r["participant_id"] for r in ptable}
# normalised name → row
sub_by_norm = {norm_name(r["session_name"]): r for r in ptable}

# ──────────────────────────────────────────────────────────────────────────────
# New participants to add (those in Excel but not yet in participants.tsv)
# ──────────────────────────────────────────────────────────────────────────────
NEW_PARTICIPANTS = [
    ("sub-059", "Fabricio Batista Narcizo"),
    ("sub-061", "Ian Wermer Gibson"),
    ("sub-062", "Sahba Tahsini"),
    ("sub-063", "Arianna Fummi"),
    ("sub-064", "Lukas Kallestrup Brandt"),
]

added = []
for sub_id, name in NEW_PARTICIPANTS:
    if sub_id in existing_ids:
        print(f"  {sub_id} already exists — skipping.")
        continue
    xl = xl_by_name.get(norm_name(name), {})
    row = {
        "participant_id":     sub_id,
        "session_name":       name,
        "age":                xl.get("age",     "n/a"),
        "sex":                xl.get("sex",     "n/a"),
        "handedness":         xl.get("hand",    "n/a"),
        "english_proficiency": xl.get("english", "n/a"),
        "education":          xl.get("edu",     "n/a"),
        "bfi44_e":            xl.get("e",       "n/a"),
        "bfi44_a":            xl.get("a",       "n/a"),
        "bfi44_c":            xl.get("c",       "n/a"),
        "bfi44_n":            xl.get("n",       "n/a"),
        "bfi44_o":            xl.get("o",       "n/a"),
    }
    ptable.append(row)
    sub_by_norm[norm_name(name)] = row
    existing_ids.add(sub_id)
    added.append(sub_id)

# Write updated participants.tsv
with open(PTABLE_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(ptable)

print(f"Updated participants.tsv — {len(ptable)} rows total "
      f"(added: {added if added else 'none (already up to date)'})")

# ──────────────────────────────────────────────────────────────────────────────
# Load session metadata
# ──────────────────────────────────────────────────────────────────────────────
with open(SESSION_META, encoding="utf-8") as f:
    smeta = list(csv.DictReader(f, delimiter="\t"))

meta_by_session: dict[str, str] = {
    r["session"]: r.get("participants_names", "") for r in smeta
}

# ──────────────────────────────────────────────────────────────────────────────
# Manual seat-order overrides (P1..P4 names; None = absent seat)
# Both run01 and run02 listed so both get the same correction.
# ──────────────────────────────────────────────────────────────────────────────
SEAT_OVERRIDES: dict[str, list] = {
    "ses-20260311_grp-06_run01": [
        "Fabricio Batista Narcizo", "Oliver Walbom", None, "Katy Andersen-Young",
    ],
    "ses-20260311_grp-06_run02": [
        "Fabricio Batista Narcizo", "Oliver Walbom", None, "Katy Andersen-Young",
    ],
}


def get_seats(session_id: str, names_str: str) -> list:
    """Return list of exactly 4 names-or-None for P1..P4."""
    if session_id in SEAT_OVERRIDES:
        return SEAT_OVERRIDES[session_id]
    if not names_str:
        return [None, None, None, None]
    parts = [n.strip() or None for n in names_str.split(";")]
    while len(parts) < 4:
        parts.append(None)
    return parts[:4]


# ──────────────────────────────────────────────────────────────────────────────
# Participant lookup by name (handles encoding garbling and minor variations)
# ──────────────────────────────────────────────────────────────────────────────
# Extra aliases for known mismatches between session_metadata and participants.tsv
KNOWN_ALIASES: dict[str, str] = {
    # norm(session_metadata_name) → norm(participants.tsv name)
    norm_name("Kate Elizabeth Andersen"): norm_name("Kate Elizabeth Andersen"),
    norm_name("Jarle Schnoor Ostvedt"):   norm_name("Jarle Schnoor Ostvedt"),
    norm_name("harshavardhan reddy"):     norm_name("harshavardhan reddy"),
    norm_name("Grace Ceesay-Hanses"):     norm_name("Grace Ceesay-Hanses"),
    norm_name("Mathias Møller Bruhn"):    norm_name("Mathias Møller Bruhn"),
    norm_name("RJ Martz"):               norm_name("RJ Martz"),
}


def lookup_participant(name: str | None) -> dict | None:
    """Return participants.tsv row for a given (possibly garbled) name."""
    if not name:
        return None
    key = norm_name(name)
    # 1. Direct normalised match
    if key in sub_by_norm:
        return sub_by_norm[key]
    # 2. Alias map
    alias = KNOWN_ALIASES.get(key)
    if alias and alias in sub_by_norm:
        return sub_by_norm[alias]
    # 3. First-token + last-token match (handles middle-name differences)
    parts = key.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        for stored, row in sub_by_norm.items():
            sp = stored.split()
            if len(sp) >= 2 and sp[0] == first and sp[-1] == last:
                return row
    # 4. ASCII-token fallback for double-encoded UTF-8 corruption in metadata
    #    e.g. "Christian Buch RÃƒÂ¸nborg" → keep only pure-ASCII words ≥4 chars
    ascii_tokens = [t for t in name.split() if t.isascii() and len(t) >= 4]
    if ascii_tokens:
        candidates = []
        for stored, row in sub_by_norm.items():
            norm_stored = norm_name(stored)
            if all(t.lower() in norm_stored for t in ascii_tokens):
                candidates.append(row)
        if len(candidates) == 1:
            return candidates[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Generate participant_map.tsv for every session folder
# ──────────────────────────────────────────────────────────────────────────────
# Regex to skip internal alpha/beta/pilot sessions (grp-A, grp-B, grp-C …)
_SKIP_RE = re.compile(r"grp-[A-Za-z]_")

total_written = 0
unmatched_names: list[tuple] = []

for output_base in OUTPUT_DIRS:
    if not output_base.exists():
        print(f"  ⚠  Output dir not found: {output_base}")
        continue

    for ses_dir in sorted(output_base.iterdir()):
        if not ses_dir.is_dir():
            continue
        session_id = ses_dir.name
        if _SKIP_RE.search(session_id):
            continue  # skip grp-A / grp-B / grp-C sessions

        pnames_str = meta_by_session.get(session_id, "")
        seats = get_seats(session_id, pnames_str)

        rows = []
        for idx, pname in enumerate(seats):
            seat_label = f"P{idx + 1}"
            if not pname:
                row = {k: "n/a" for k in MAP_COLS}
                row["seat"] = seat_label
                rows.append(row)
                continue

            pr = lookup_participant(pname)
            if pr:
                rows.append({
                    "seat":                seat_label,
                    "participant_id":      pr["participant_id"],
                    "name":               pr["session_name"],
                    "age":                pr.get("age",                  "n/a"),
                    "sex":                pr.get("sex",                  "n/a"),
                    "handedness":         pr.get("handedness",           "n/a"),
                    "english_proficiency": pr.get("english_proficiency", "n/a"),
                    "education":          pr.get("education",            "n/a"),
                    "bfi44_e":            pr.get("bfi44_e",              "n/a"),
                    "bfi44_a":            pr.get("bfi44_a",              "n/a"),
                    "bfi44_c":            pr.get("bfi44_c",              "n/a"),
                    "bfi44_n":            pr.get("bfi44_n",              "n/a"),
                    "bfi44_o":            pr.get("bfi44_o",              "n/a"),
                })
            else:
                # Name present in metadata but not in participants.tsv
                row = {k: "n/a" for k in MAP_COLS}
                row["seat"] = seat_label
                row["name"] = pname   # preserve the raw name for reference
                rows.append(row)
                unmatched_names.append((session_id, seat_label, pname))

        out_file = ses_dir / "participant_map.tsv"
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MAP_COLS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        total_written += 1

print(f"\nWrote {total_written} participant_map.tsv files across {len(OUTPUT_DIRS)} output roots.")

if unmatched_names:
    print("\n⚠  Names in session metadata not matched to any participant row:")
    for ses, seat, nm in unmatched_names:
        print(f"   {ses}  {seat}  '{nm}'")
else:
    print("All participant names matched successfully.")
