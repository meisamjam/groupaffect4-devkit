---
applyTo: "**/*.py"
---

# Python Coding Standards — AffectAI Data Processing

These rules apply to all Python files in this repository (tools, src, tests).

## Type hints

- All **public** function and method signatures must have type-annotated parameters and return types
- Use `from __future__ import annotations` for forward references if needed
- Prefer specific types: `list[str]` over `List[str]`, `dict[str, int]` over `Dict[str, int]` (Python 3.10+)
- Include `-> None` explicitly when a function returns nothing

```python
# Good
def load_events(events_path: Path, task: str) -> pd.DataFrame:
    ...

# Bad
def load_events(events_path, task):
    ...
```

## Imports

Order: **stdlib → third-party → local**, each group separated by a blank line. No star imports.

```python
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from affectai_capture.utils import resolve_session_dir
```

## Logging

- Use `logging` module everywhere — never `print()` for diagnostic output
- Configure level at entry point with `--verbose` / `--quiet` flags
- Log at appropriate levels: `DEBUG` for trace detail, `INFO` for progress, `WARNING` for recoverable issues, `ERROR` for failures

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing session %s", session_id)
logger.warning("Frame log missing for cam3 — skipping")
```

## CLI conventions

- Use `argparse` for all `tools/` scripts
- Every argument must have a meaningful `help=` string
- Scripts must be runnable as `python tools/<script>.py --help`
- Use `--verbose` / `--quiet` flags to control log level
- Include `--dry-run` on write-heavy pipeline scripts

## File I/O

- Use `pathlib.Path` for all file and directory operations — never `os.path`
- Validate paths early; raise `FileNotFoundError` with a descriptive message if required inputs are missing

```python
# Good
calibration = Path(args.calibration)
if not calibration.exists():
    raise FileNotFoundError(f"Calibration TOML not found: {calibration}")

# Bad
import os
if not os.path.exists(args.calibration):
    ...
```

## Error handling

- Never use bare `except:` — always catch specific exceptions
- Log the error with context before re-raising or returning

```python
# Good
try:
    data = json.loads(text)
except json.JSONDecodeError as e:
    logger.error("Failed to parse events JSONL at line %d: %s", lineno, e)
    raise

# Bad
try:
    ...
except:
    pass
```

## Docstrings

- `tools/` scripts: module-level docstring describing purpose and key CLI arguments
- Non-trivial functions: single-line or Google-style docstring
- Keep docstrings honest — update them when behaviour changes

## Configuration

- Read session paths, group IDs, and device parameters from CLI args or config files
- Never hardcode paths, group identifiers, or device parameters in source code
