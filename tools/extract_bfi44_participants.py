#!/usr/bin/env python3
"""Extract BFI-44 personality scores from the pre-session questionnaire and
produce BIDS-compatible participants.tsv + participants.json.

The script:
  1. Reads the Excel questionnaire → computes per-domain BFI-44 mean scores.
  2. Reads the session inventory CSV → identifies all unique participants that
     appeared in recorded sessions + assigns globally unique sub-IDs.
  3. Fuzzy-matches questionnaire respondents to session participants by name.
  4. Outputs three files:
       metadata/participants.tsv          — BIDS participants file
       metadata/participants.json         — BIDS column-description sidecar
       metadata/name_matching_review.tsv  — match audit for manual verification

Usage (from repo root):
    python tools/extract_bfi44_participants.py
    python tools/extract_bfi44_participants.py --match-threshold 0.55
    python tools/extract_bfi44_participants.py --review   # print match table only

BFI-44 reference:
    John, O. P., & Srivastava, S. (1999). The Big Five Trait taxonomy.
    https://doi.org/10.1006/jrpe.1999.2257
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from pathlib import Path

try:
    import openpyxl
except ImportError:
    sys.exit("openpyxl is required: pip install openpyxl")


# ──────────────────────────────────────────────────────────────────────────────
# BFI-44 item definitions
# (0-based index within BFI block, factor letter, is_reversed)
# Items are ordered as they appear in the questionnaire: Part1 (1-20),
# Part2 (21-40), Part3 (41-44).
# ──────────────────────────────────────────────────────────────────────────────
BFI_ITEMS: list[tuple[int, str, bool]] = [
    # idx  factor  reversed   # item text
    (0,  "E", False),  # 1.  Is talkative
    (1,  "A", True),   # 2.  Tends to find fault with others
    (2,  "C", False),  # 3.  Does a thorough job
    (3,  "N", False),  # 4.  Is depressed, blue
    (4,  "O", False),  # 5.  Is original, comes up with new ideas
    (5,  "E", True),   # 6.  Is reserved
    (6,  "A", False),  # 7.  Is helpful and unselfish with others
    (7,  "C", True),   # 8.  Can be somewhat careless
    (8,  "N", True),   # 9.  Is relaxed, handles stress well
    (9,  "O", False),  # 10. Is curious about many different things
    (10, "E", False),  # 11. Is full of energy
    (11, "A", True),   # 12. Starts quarrels with others
    (12, "C", False),  # 13. Is a reliable worker
    (13, "N", False),  # 14. Can be tense
    (14, "O", False),  # 15. Is ingenious, a deep thinker
    (15, "E", False),  # 16. Generates a lot of enthusiasm
    (16, "A", False),  # 17. Has a forgiving nature
    (17, "C", True),   # 18. Tends to be disorganized
    (18, "N", False),  # 19. Worries a lot
    (19, "O", False),  # 20. Has an active imagination
    (20, "E", True),   # 21. Tends to be quiet
    (21, "A", False),  # 22. Is generally trusting
    (22, "C", True),   # 23. Tends to be lazy
    (23, "N", True),   # 24. Is emotionally stable, not easily upset
    (24, "O", False),  # 25. Is inventive
    (25, "E", False),  # 26. Has an assertive personality
    (26, "A", True),   # 27. Can be cold and aloof
    (27, "C", False),  # 28. Perseveres until the task is finished
    (28, "N", False),  # 29. Can be moody
    (29, "O", False),  # 30. Values artistic, aesthetic experiences
    (30, "E", True),   # 31. Is sometimes shy, inhibited
    (31, "A", False),  # 32. Is considerate, kind to almost everyone
    (32, "C", False),  # 33. Does things efficiently
    (33, "N", True),   # 34. Remains calm in tense situations
    (34, "O", True),   # 35. Prefers work that is routine
    (35, "E", False),  # 36. Is outgoing, sociable
    (36, "A", True),   # 37. Is sometimes rude to others
    (37, "C", False),  # 38. Makes plans and follows through on them
    (38, "N", False),  # 39. Gets nervous easily
    (39, "O", False),  # 40. Likes to reflect, play with ideas
    (40, "O", True),   # 41. Has few artistic interests
    (41, "A", False),  # 42. Likes to cooperate with others
    (42, "C", True),   # 43. Is easily distracted
    (43, "O", False),  # 44. Is sophisticated in art, music, or literature
]

# 0-based column index of the first BFI item in the questionnaire spreadsheet
BFI_FIRST_COL = 14

RESPONSE_MAP: dict[str, int] = {
    "strongly disagree": 1,
    "disagree": 2,
    "neither disagree nor agree": 3,
    "agree": 4,
    "strongly agree": 5,
}

_DOMAIN_ITEMS: dict[str, list[int]] = {  # 1-based item numbers per factor
    "E": [1, 6, 11, 16, 21, 26, 31, 36],
    "A": [2, 7, 12, 17, 22, 27, 32, 37, 42],
    "C": [3, 8, 13, 18, 23, 28, 33, 38, 43],
    "N": [4, 9, 14, 19, 24, 29, 34, 39],
    "O": [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
}
_DOMAIN_REVERSED: dict[str, list[int]] = {
    "E": [6, 21, 31],
    "A": [2, 12, 27, 37],
    "C": [8, 18, 23, 43],
    "N": [9, 24, 34],
    "O": [35, 41],
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    """Lowercase + collapse whitespace for fuzzy comparison."""
    return " ".join(name.strip().lower().split())


def _score_item(text: object, is_reversed: bool) -> int | None:
    if text is None:
        return None
    val = RESPONSE_MAP.get(str(text).lower().strip())
    if val is None:
        return None
    return 6 - val if is_reversed else val


def _compute_bfi(row: list) -> dict[str, float | None]:
    """Return dict bfi44_{e,a,c,n,o} → mean score (or None if all missing)."""
    domain_scores: dict[str, list[int]] = {f: [] for f in "EACNO"}
    for idx, factor, rev in BFI_ITEMS:
        col = BFI_FIRST_COL + idx
        s = _score_item(row[col] if col < len(row) else None, rev)
        if s is not None:
            domain_scores[factor].append(s)
    result: dict[str, float | None] = {}
    for f in "EACNO":
        vals = domain_scores[f]
        result[f"bfi44_{f.lower()}"] = round(sum(vals) / len(vals), 3) if vals else None
    return result


def _normalize_sex(v: str | None) -> str:
    if not v:
        return "n/a"
    lv = v.lower()
    if "female" in lv:
        return "female"
    if "male" in lv:
        return "male"
    return lv


def _normalize_hand(v: str | None) -> str:
    if not v:
        return "n/a"
    lv = v.lower()
    if "right" in lv:
        return "right"
    if "left" in lv:
        return "left"
    if "ambid" in lv:
        return "ambidextrous"
    return lv


def _parse_age(col7: object, col8: object) -> str:
    """Return exact integer age as a string, or midpoint for band-only respondents.

    col7 ('What is you age?1'): numeric free-text entry (used by 51 respondents)
    col8 ('What is you age?'):  dropdown band string (used by 5 early respondents)

    For band-only respondents the midpoint of the band is stored and flagged
    in participants.json. For unknown/missing values returns 'n/a'.
    """
    # Prefer exact numeric age (col7)
    if col7 is not None:
        try:
            return str(int(float(str(col7).strip())))
        except ValueError:
            pass
    # Fall back to band midpoint (col8)
    if col8 is not None:
        band = str(col8).strip()
        # Patterns: '25-34', '45-54', '65+'
        import re
        m = re.match(r'^(\d+)-(\d+)$', band)
        if m:
            return str((int(m.group(1)) + int(m.group(2))) // 2)
        m = re.match(r'^(\d+)\+$', band)
        if m:
            return str(int(m.group(1)))
    return 'n/a'


def _normalize_education(v: str | None) -> str:
    if not v:
        return "n/a"
    lv = v.lower()
    if "phd" in lv or "doctorate" in lv:
        return "phd"
    if "master" in lv:
        return "master"
    if "bachelor" in lv:
        return "bachelor"
    if "professional" in lv or "diploma" in lv:
        return "professional_certificate"
    if "graduate certificate" in lv:
        return "graduate_certificate"
    if lv.strip() == "ap":
        return "ap"
    return v.strip()


def _fuzzy_match(name: str, candidates: list[str], threshold: float) -> str | None:
    norm = _norm_name(name)
    normed = [_norm_name(c) for c in candidates]
    matches = difflib.get_close_matches(norm, normed, n=1, cutoff=threshold)
    return candidates[normed.index(matches[0])] if matches else None


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_questionnaire(path: Path) -> list[dict]:
    """Return one record per questionnaire respondent."""
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    records = []
    for row in rows[1:]:  # skip header
        row = list(row)
        bfi = _compute_bfi(row)
        # Age: col 7 has exact numeric age (51 respondents);
        # col 8 has a dropdown band string (5 early respondents).
        records.append(
            {
                "name": (row[4] or "").strip(),
                "email": (row[3] or "").strip(),
                "age": _parse_age(row[7], row[8]),
                "sex": _normalize_sex(row[9]),
                "handedness": _normalize_hand(row[10]),
                "english_proficiency": str(row[12]).strip() if row[12] else "n/a",
                "education": _normalize_education(row[13]),
                **bfi,
            }
        )
    return records


def load_session_participants(path: Path) -> list[str]:
    """Return ordered list of unique participant names across all sessions."""
    seen: set[str] = set()
    ordered: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names_str = row.get("participants_names", "").strip()
            if not names_str:
                continue
            for name in names_str.split(";"):
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    ordered.append(name)
    return ordered


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--questionnaire",
        default="metadata/GN Hearing Research \u2013 Pre-Session Questionnaire(Sheet1) (1).xlsx",
        help="Path to the pre-session questionnaire Excel file.",
    )
    ap.add_argument(
        "--session-inventory",
        default="metadata/high_level_session_inventory.csv",
        help="Path to the high_level_session_inventory CSV.",
    )
    ap.add_argument(
        "--out-dir",
        default="metadata",
        help="Directory for output files (default: metadata/).",
    )
    ap.add_argument(
        "--match-threshold",
        type=float,
        default=0.6,
        help="Fuzzy name-match threshold 0–1 (default 0.6). Lower = more permissive.",
    )
    ap.add_argument(
        "--review",
        action="store_true",
        help="Print name-matching table to stdout and exit without writing TSV.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading questionnaire …")
    q_records = load_questionnaire(Path(args.questionnaire))
    q_by_name = {r["name"]: r for r in q_records}
    print(f"  {len(q_records)} questionnaire respondents")

    print("Loading session inventory …")
    session_names = load_session_participants(Path(args.session_inventory))
    print(f"  {len(session_names)} unique session participants")

    # ── Assign globally unique sub-IDs ───────────────────────────────────────
    sub_id_map: dict[str, str] = {
        name: f"sub-{i:03d}" for i, name in enumerate(session_names, 1)
    }

    # ── Fuzzy-match questionnaire → session names ─────────────────────────────
    match_rows: list[dict] = []
    q_to_session: dict[str, str | None] = {}

    for q_rec in q_records:
        q_name = q_rec["name"]
        best = _fuzzy_match(q_name, session_names, args.match_threshold)
        score = (
            difflib.SequenceMatcher(None, _norm_name(q_name), _norm_name(best)).ratio()
            if best
            else 0.0
        )
        q_to_session[q_name] = best
        match_rows.append(
            {
                "sub_id": sub_id_map.get(best, "UNMATCHED") if best else "UNMATCHED",
                "session_name": best or "",
                "questionnaire_name": q_name,
                "match_score": round(score, 3),
                "status": "matched" if best else "unmatched",
            }
        )

    # Participants in sessions with no questionnaire match
    matched_session = {m["session_name"] for m in match_rows if m["status"] == "matched"}
    for sn in session_names:
        if sn not in matched_session:
            match_rows.append(
                {
                    "sub_id": sub_id_map[sn],
                    "session_name": sn,
                    "questionnaire_name": "",
                    "match_score": 0.0,
                    "status": "no_questionnaire",
                }
            )

    match_rows.sort(key=lambda r: r["sub_id"])

    # ── Review mode ──────────────────────────────────────────────────────────
    if args.review:
        print(f"\n{'sub_id':<10} {'score':<6} {'status':<18} {'session_name':<40} {'questionnaire_name'}")
        print("-" * 120)
        for m in match_rows:
            print(
                f"{m['sub_id']:<10} {m['match_score']:<6} {m['status']:<18} "
                f"{m['session_name']:<40} {m['questionnaire_name']}"
            )
        unmatched_q = sum(1 for m in match_rows if m["status"] == "unmatched")
        no_q = sum(1 for m in match_rows if m["status"] == "no_questionnaire")
        print(f"\n  {len(session_names)} session participants")
        print(f"  {sum(1 for m in match_rows if m['status'] == 'matched')} matched")
        print(f"  {no_q} session participants without questionnaire response")
        print(f"  {unmatched_q} questionnaire responses without session match")
        return

    # ── Write match review file ───────────────────────────────────────────────
    review_path = out_dir / "name_matching_review.tsv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sub_id", "session_name", "questionnaire_name", "match_score", "status"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(match_rows)
    print(f"  Name-matching review → {review_path}")

    # Build a reverse map: session_name → questionnaire record
    session_to_q: dict[str, dict | None] = {sn: None for sn in session_names}
    for m in match_rows:
        if m["status"] == "matched":
            session_to_q[m["session_name"]] = q_by_name.get(m["questionnaire_name"])

    # ── Build participants.tsv ────────────────────────────────────────────────
    fieldnames = [
        "participant_id",
        "session_name",   # NOTE: remove before public data release
        "age",
        "sex",
        "handedness",
        "english_proficiency",
        "education",
        "bfi44_e",
        "bfi44_a",
        "bfi44_c",
        "bfi44_n",
        "bfi44_o",
    ]

    tsv_rows: list[dict] = []
    for sn in session_names:
        sub = sub_id_map[sn]
        q = session_to_q.get(sn)
        tsv_rows.append(
            {
                "participant_id": sub,
                "session_name": sn,
                "age": q["age"] if (q and q["age"] != "n/a") else "n/a",
                "sex": q["sex"] if q else "n/a",
                "handedness": q["handedness"] if q else "n/a",
                "english_proficiency": q["english_proficiency"] if q else "n/a",
                "education": q["education"] if q else "n/a",
                "bfi44_e": q["bfi44_e"] if (q and q["bfi44_e"] is not None) else "n/a",
                "bfi44_a": q["bfi44_a"] if (q and q["bfi44_a"] is not None) else "n/a",
                "bfi44_c": q["bfi44_c"] if (q and q["bfi44_c"] is not None) else "n/a",
                "bfi44_n": q["bfi44_n"] if (q and q["bfi44_n"] is not None) else "n/a",
                "bfi44_o": q["bfi44_o"] if (q and q["bfi44_o"] is not None) else "n/a",
            }
        )

    tsv_path = out_dir / "participants.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(tsv_rows)
    print(f"  participants.tsv → {tsv_path} ({len(tsv_rows)} rows)")

    # ── Write participants.json sidecar ───────────────────────────────────────
    sidecar = {
        "participant_id": {
            "Description": "Unique participant identifier (anonymised, globally unique across all sessions)"
        },
        "session_name": {
            "Description": (
                "Full name as recorded in session metadata. "
                "INTERNAL USE ONLY — remove this column before any public data release."
            )
        },
        "age": {
            "Description": (
                "Self-reported age in years (integer). "
                "51 of 56 respondents entered an exact numeric age (col 7 of the questionnaire). "
                "5 early respondents used a dropdown band (col 8); for these, "
                "the band midpoint is stored (e.g. '25-34' → 29). "
                "Participants not in the questionnaire are set to n/a."
            ),
            "Units": "years",
        },
        "sex": {
            "Description": "Self-reported biological sex (from pre-session questionnaire)",
            "Levels": {"male": "Male", "female": "Female", "n/a": "Not available"},
        },
        "handedness": {
            "Description": "Self-reported dominant hand (from pre-session questionnaire)",
            "Levels": {
                "right": "Right-handed",
                "left": "Left-handed",
                "ambidextrous": "Ambidextrous",
                "n/a": "Not available",
            },
        },
        "english_proficiency": {
            "Description": "Self-reported English language proficiency (from pre-session questionnaire)",
            "Levels": {
                "Native speaker": "Native speaker",
                "Fluent": "Fluent non-native speaker",
                "Intermediate": "Intermediate proficiency",
                "n/a": "Not available",
            },
        },
        "education": {
            "Description": "Self-reported highest education level (from pre-session questionnaire)",
            "Levels": {
                "phd": "PhD / Doctorate",
                "master": "Master's degree",
                "bachelor": "Bachelor's degree",
                "professional_certificate": "Professional Certificate / Higher Professional Diploma",
                "graduate_certificate": "Graduate Certificate",
                "ap": "AP (Erhvervsakademiuddannelse)",
                "n/a": "Not available",
            },
        },
        "bfi44_e": {
            "Description": (
                "BFI-44 Extraversion domain: mean Likert score across 8 items "
                "(1=Strongly disagree … 5=Strongly agree, after reverse scoring). "
                "Higher = more extraverted."
            ),
            "TermURL": "https://doi.org/10.1006/jrpe.1999.2257",
            "Units": "mean Likert score",
            "Range": [1.0, 5.0],
            "Items": _DOMAIN_ITEMS["E"],
            "ReversedItems": _DOMAIN_REVERSED["E"],
        },
        "bfi44_a": {
            "Description": (
                "BFI-44 Agreeableness domain: mean Likert score across 9 items. "
                "Higher = more agreeable."
            ),
            "TermURL": "https://doi.org/10.1006/jrpe.1999.2257",
            "Units": "mean Likert score",
            "Range": [1.0, 5.0],
            "Items": _DOMAIN_ITEMS["A"],
            "ReversedItems": _DOMAIN_REVERSED["A"],
        },
        "bfi44_c": {
            "Description": (
                "BFI-44 Conscientiousness domain: mean Likert score across 9 items. "
                "Higher = more conscientious."
            ),
            "TermURL": "https://doi.org/10.1006/jrpe.1999.2257",
            "Units": "mean Likert score",
            "Range": [1.0, 5.0],
            "Items": _DOMAIN_ITEMS["C"],
            "ReversedItems": _DOMAIN_REVERSED["C"],
        },
        "bfi44_n": {
            "Description": (
                "BFI-44 Neuroticism domain: mean Likert score across 8 items. "
                "Higher = higher neurotic tendency."
            ),
            "TermURL": "https://doi.org/10.1006/jrpe.1999.2257",
            "Units": "mean Likert score",
            "Range": [1.0, 5.0],
            "Items": _DOMAIN_ITEMS["N"],
            "ReversedItems": _DOMAIN_REVERSED["N"],
        },
        "bfi44_o": {
            "Description": (
                "BFI-44 Openness domain: mean Likert score across 10 items. "
                "Higher = more open to experience."
            ),
            "TermURL": "https://doi.org/10.1006/jrpe.1999.2257",
            "Units": "mean Likert score",
            "Range": [1.0, 5.0],
            "Items": _DOMAIN_ITEMS["O"],
            "ReversedItems": _DOMAIN_REVERSED["O"],
        },
    }

    json_path = out_dir / "participants.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    print(f"  participants.json  → {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_matched = sum(1 for r in tsv_rows if r["bfi44_e"] != "n/a")
    n_no_q = sum(1 for r in tsv_rows if r["bfi44_e"] == "n/a")
    n_unmatched_q = sum(1 for m in match_rows if m["status"] == "unmatched")

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  {len(tsv_rows):3d}  unique session participants")
    print(f"  {n_matched:3d}  with BFI-44 scores (questionnaire matched)")
    print(f"  {n_no_q:3d}  without BFI-44 scores (no matching questionnaire)")
    print(f"  {n_unmatched_q:3d}  questionnaire responses not matched to any session")
    print(f"\n  Review {review_path.name} to verify fuzzy name matches.")
    print(f"  ⚠  Remove 'session_name' column before any public data release.")


if __name__ == "__main__":
    main()
