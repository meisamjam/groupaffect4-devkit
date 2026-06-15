"""Quick cross-check: session_schedule.tsv vs session_metadata_report.tsv"""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
schedule = {}
with open(root / "configs" / "session_schedule.tsv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        schedule[r["group_id"].strip()] = r

with open(root / "metadata" / "session_metadata_report.tsv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

hdr = f"{'Session':<38s} {'Sched':>10s} {'Actual':>10s} {'Date?':>5s} {'Time':>11s}  Names?"
print(hdr)
print("-" * len(hdr) + "-" * 40)

for r in rows:
    gid = r["group_id"]
    ses = r["session"]
    d = ses.replace("ses-", "").split("_")[0]
    actual = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    if gid not in schedule:
        print(f"{ses:<38s} {'--':>10s} {actual:>10s} {'--':>5s} {'--':>11s}  (no schedule)")
        continue

    s = schedule[gid]
    sd = s["date"].strip()
    date_ok = "OK" if sd == actual else "DIFF"
    time_str = f"{s['start_time'].strip()}-{s['end_time'].strip()}"

    sn = {n.strip().lower() for n in [s.get("name_1",""),s.get("name_2",""),s.get("name_3",""),s.get("name_4","")] if n.strip()}
    rn = {n.strip().lower() for n in r.get("participants_names","").split(";") if n.strip()}

    if sn == rn:
        nm = "OK"
    elif sn & rn:
        diff_s = sn - rn
        diff_r = rn - sn
        parts = []
        if diff_s:
            parts.append(f"sched_only: {', '.join(sorted(diff_s))}")
        if diff_r:
            parts.append(f"report_only: {', '.join(sorted(diff_r))}")
        nm = "PARTIAL  " + " | ".join(parts)
    elif not rn:
        nm = "NO_NAMES"
    else:
        nm = "MISMATCH"

    print(f"{ses:<38s} {sd:>10s} {actual:>10s} {date_ok:>5s} {time_str:>11s}  {nm}")
