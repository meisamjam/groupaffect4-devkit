"""Annotation schema + JSON I/O.

One `annotations.json` file lives next to a task's media in `annot/`.
Tiers are named tracks (e.g. "P1.backchannel"); entries are spans or points
inside a tier. Load/save is deliberately tolerant of missing or unknown
fields so we can evolve the schema without breaking older files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Tier:
    id: str
    participant: str = ""  # "" = session-level tier
    kind: str = "span"  # "span" | "point"
    source: str = "manual"  # "manual" | "whisper" | "mocap/..." | ...
    color: str = ""  # optional hex override; empty = auto-assigned
    readonly: bool = False


@dataclass
class Entry:
    tier: str
    start: float
    end: float  # == start for "point" tiers
    label: str = ""
    note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AnnotationDoc:
    sub: str = ""
    ses: str = ""
    task: str = ""
    run: str = ""
    participants: dict[str, str] = field(default_factory=dict)
    tiers: list[Tier] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ tier ops
    def tier(self, tier_id: str) -> Tier | None:
        return next((t for t in self.tiers if t.id == tier_id), None)

    def add_tier(self, tier: Tier) -> Tier:
        if self.tier(tier.id):
            raise ValueError(f"tier already exists: {tier.id}")
        self.tiers.append(tier)
        return tier

    def remove_tier(self, tier_id: str) -> None:
        self.tiers = [t for t in self.tiers if t.id != tier_id]
        self.entries = [e for e in self.entries if e.tier != tier_id]

    # ---------------------------------------------------------------- entry ops
    def add_entry(self, entry: Entry) -> Entry:
        if self.tier(entry.tier) is None:
            raise ValueError(f"unknown tier: {entry.tier}")
        self.entries.append(entry)
        self.entries.sort(key=lambda e: (e.tier, e.start))
        return entry

    def remove_entry(self, entry: Entry) -> None:
        self.entries = [e for e in self.entries if e is not entry]

    def entries_in_tier(self, tier_id: str) -> list[Entry]:
        return [e for e in self.entries if e.tier == tier_id]


def load(path: Path) -> AnnotationDoc:
    """Load an annotations.json; return an empty doc if the file is missing."""
    if not path.is_file():
        return AnnotationDoc()
    raw = json.loads(path.read_text(encoding="utf-8"))
    tiers = [
        Tier(
            id=t["id"],
            participant=t.get("participant", ""),
            kind=t.get("kind", "span"),
            source=t.get("source", "manual"),
            color=t.get("color", ""),
            readonly=bool(t.get("readonly", False)),
        )
        for t in raw.get("tiers", [])
    ]
    entries = [
        Entry(
            tier=e["tier"],
            start=float(e["start"]),
            end=float(e.get("end", e["start"])),
            label=e.get("label", ""),
            note=e.get("note", ""),
        )
        for e in raw.get("entries", [])
    ]
    return AnnotationDoc(
        sub=raw.get("sub", ""),
        ses=raw.get("ses", ""),
        task=raw.get("task", ""),
        run=raw.get("run", ""),
        participants=dict(raw.get("participants", {})),
        tiers=tiers,
        entries=entries,
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
    )


def save(path: Path, doc: AnnotationDoc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": doc.schema_version,
        "sub": doc.sub,
        "ses": doc.ses,
        "task": doc.task,
        "run": doc.run,
        "participants": doc.participants,
        "tiers": [asdict(t) for t in doc.tiers],
        "entries": [asdict(e) for e in doc.entries],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
