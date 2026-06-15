# BFI-44 Personality Scores

This document describes how Big Five personality scores are extracted from the
pre-session questionnaire and stored in the BIDS `participants.tsv` file.

---

## Instrument

**Big Five Inventory – 44 items (BFI-44)**

> John, O. P., & Srivastava, S. (1999). The Big Five trait taxonomy: History,
> measurement, and theoretical perspectives. In L. A. Pervin & O. P. John (Eds.),
> *Handbook of personality: Theory and research* (2nd ed., pp. 102–138). Guilford Press.

The BFI-44 measures five broad personality traits using 44 short descriptive
phrases rated on a 5-point Likert scale:

| Rating | Label |
|--------|-------|
| 1 | Strongly disagree |
| 2 | Disagree |
| 3 | Neither disagree nor agree |
| 4 | Agree |
| 5 | Strongly agree |

---

## Five Domains

| Column in `participants.tsv` | Domain | Items (1-based) | Reversed items |
|---|---|---|---|
| `bfi44_e` | **Extraversion** | 1, 6, 11, 16, 21, 26, 31, 36 | 6, 21, 31 |
| `bfi44_a` | **Agreeableness** | 2, 7, 12, 17, 22, 27, 32, 37, 42 | 2, 12, 27, 37 |
| `bfi44_c` | **Conscientiousness** | 3, 8, 13, 18, 23, 28, 33, 38, 43 | 8, 18, 23, 43 |
| `bfi44_n` | **Neuroticism** | 4, 9, 14, 19, 24, 29, 34, 39 | 9, 24, 34 |
| `bfi44_o` | **Openness to Experience** | 5, 10, 15, 20, 25, 30, 35, 40, 41, 44 | 35, 41 |

Each domain score is the **mean** of its items after reverse scoring, rounded to
3 decimal places. Range: **1.0 – 5.0**. Higher values = stronger trait expression.

Reverse scoring: $\text{score}_{\text{rev}} = 6 - \text{raw score}$

---

## All 44 Items

Items are numbered in questionnaire order (Parts 1–3 as collected).

| # | Item text | Domain | Reversed |
|---|-----------|--------|----------|
| 1 | Is talkative | E | No |
| 2 | Tends to find fault with others | A | **Yes** |
| 3 | Does a thorough job | C | No |
| 4 | Is depressed, blue | N | No |
| 5 | Is original, comes up with new ideas | O | No |
| 6 | Is reserved | E | **Yes** |
| 7 | Is helpful and unselfish with others | A | No |
| 8 | Can be somewhat careless | C | **Yes** |
| 9 | Is relaxed, handles stress well | N | **Yes** |
| 10 | Is curious about many different things | O | No |
| 11 | Is full of energy | E | No |
| 12 | Starts quarrels with others | A | **Yes** |
| 13 | Is a reliable worker | C | No |
| 14 | Can be tense | N | No |
| 15 | Is ingenious, a deep thinker | O | No |
| 16 | Generates a lot of enthusiasm | E | No |
| 17 | Has a forgiving nature | A | No |
| 18 | Tends to be disorganized | C | **Yes** |
| 19 | Worries a lot | N | No |
| 20 | Has an active imagination | O | No |
| 21 | Tends to be quiet | E | **Yes** |
| 22 | Is generally trusting | A | No |
| 23 | Tends to be lazy | C | **Yes** |
| 24 | Is emotionally stable, not easily upset | N | **Yes** |
| 25 | Is inventive | O | No |
| 26 | Has an assertive personality | E | No |
| 27 | Can be cold and aloof | A | **Yes** |
| 28 | Perseveres until the task is finished | C | No |
| 29 | Can be moody | N | No |
| 30 | Values artistic, aesthetic experiences | O | No |
| 31 | Is sometimes shy, inhibited | E | **Yes** |
| 32 | Is considerate, kind to almost everyone | A | No |
| 33 | Does things efficiently | C | No |
| 34 | Remains calm in tense situations | N | **Yes** |
| 35 | Prefers work that is routine | O | **Yes** |
| 36 | Is outgoing, sociable | E | No |
| 37 | Is sometimes rude to others | A | **Yes** |
| 38 | Makes plans and follows through on them | C | No |
| 39 | Gets nervous easily | N | No |
| 40 | Likes to reflect, play with ideas | O | No |
| 41 | Has few artistic interests | O | **Yes** |
| 42 | Likes to cooperate with others | A | No |
| 43 | Is easily distracted | C | **Yes** |
| 44 | Is sophisticated in art, music, or literature | O | No |

---

## Questionnaire Administration

- **Format:** Microsoft Forms (online); split into three parts within one form.
- **Timing:** Completed by participants before their session as part of the pre-session questionnaire.
- **Source file:** `metadata/GN Hearing Research – Pre-Session Questionnaire(Sheet1) (1).xlsx`
- **Respondents:** 57 completed questionnaires across 58 unique session participants.

Combined columns in the Excel file (BFI columns 15–58, 0-based indices 14–57):
- Part 1: items 1–20 (columns 15–34)
- Part 2: items 21–40 (columns 35–54)
- Part 3: items 41–44 (columns 55–58)

---

## Extraction Pipeline

**Script:** `tools/extract_bfi44_participants.py`

```bash
# Preview name matching (no output written)
python tools/extract_bfi44_participants.py --review

# Generate all output files
python tools/extract_bfi44_participants.py

# Adjust fuzzy-match threshold (default 0.6, lower = more permissive)
python tools/extract_bfi44_participants.py --match-threshold 0.55
```

**Steps:**
1. Load questionnaire Excel → compute per-domain BFI-44 means.
2. Load `metadata/high_level_session_inventory.csv` → build ordered list of all
   unique session participants and assign globally unique `sub-NNN` IDs.
3. Fuzzy name-match questionnaire respondents → session participant names
   (`difflib.SequenceMatcher`, threshold 0.6).
4. Write output files.

**Match statistics (2026-04-08, after Diana Taune form added):**

| Status | Count |
|--------|-------|
| Matched (BFI-44 available) | 50 |
| Session participants without questionnaire | 8 |
| Questionnaire responses not matched to a session | 7 |

Review `metadata/name_matching_review.tsv` to audit all matches, especially
fuzzy matches with score < 1.0.

---

## Output Files

### `metadata/participants.tsv`

BIDS-compliant tab-separated participant roster. One row per unique session
participant across all sessions/groups.

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | string | `sub-NNN` (globally unique, anonymised) |
| `session_name` | string | Full name as in session metadata — **remove before public release** |
| `age` | integer | Exact self-reported age in years. For the 3 matched respondents who used a dropdown band (e.g. `25-34`), the band midpoint is stored. `n/a` for unmatched participants. |
| `sex` | string | `male` / `female` / `n/a` |
| `handedness` | string | `right` / `left` / `ambidextrous` / `n/a` |
| `english_proficiency` | string | `Native speaker` / `Fluent` / `Intermediate` / `n/a` |
| `education` | string | Normalised education level (see below) |
| `bfi44_e` | float | Extraversion mean score (1.0–5.0) or `n/a` |
| `bfi44_a` | float | Agreeableness mean score (1.0–5.0) or `n/a` |
| `bfi44_c` | float | Conscientiousness mean score (1.0–5.0) or `n/a` |
| `bfi44_n` | float | Neuroticism mean score (1.0–5.0) or `n/a` |
| `bfi44_o` | float | Openness mean score (1.0–5.0) or `n/a` |

**Education level values:**

| Value | Meaning |
|-------|---------|
| `phd` | PhD / Doctorate |
| `master` | Master's degree |
| `bachelor` | Bachelor's degree |
| `professional_certificate` | Professional Certificate / Higher Professional Diploma |
| `graduate_certificate` | Graduate Certificate |
| `ap` | AP (Erhvervsakademiuddannelse) |
| `n/a` | Not available |

### `metadata/participants.json`

BIDS column-description sidecar. Contains `Description`, `Levels`, `Units`,
`Range`, `Items`, and `ReversedItems` for every column, plus the BFI-44
reference DOI.

### `metadata/name_matching_review.tsv`

Audit file for manual verification of fuzzy name matches.

| Column | Description |
|--------|-------------|
| `sub_id` | Assigned `sub-NNN` (or `UNMATCHED`) |
| `session_name` | Name from session inventory |
| `questionnaire_name` | Matched name from questionnaire |
| `match_score` | `SequenceMatcher` ratio (0–1; 1.0 = exact) |
| `status` | `matched` / `no_questionnaire` / `unmatched` |

---

## Privacy & Data Release

- The `session_name` column in `participants.tsv` contains real names and must be
  **removed** before any public or shared data release.
- BFI-44 scores are stored as aggregate means — individual item responses are not
  retained in `participants.tsv`.
- Raw questionnaire responses remain in the source Excel file and must be treated
  as personally identifiable information.

---

## Interpretation Notes

- **Extraversion (E):** sociable, assertive, energetic vs. reserved, quiet.
- **Agreeableness (A):** cooperative, trusting, helpful vs. competitive, cold.
- **Conscientiousness (C):** organised, reliable, thorough vs. impulsive, careless.
- **Neuroticism (N):** anxious, moody, tense vs. calm, stable (high N = more neurotic).
- **Openness (O):** curious, imaginative, aesthetic vs. conventional, routine-preferring.

Population norms (US adults, John & Srivastava 1999): E ≈ 3.2, A ≈ 3.8,
C ≈ 3.5, N ≈ 3.1, O ≈ 3.9 (approximate means).

For group-level analysis (e.g. group composition effects on collaboration), see
the `annot/` and `beh/` BIDS modalities for task-level outcomes to correlate
with these trait scores.
