---
applyTo: "**"
---

# Participant Privacy & Data Security — AffectAI Data Processing

These rules apply to every file in this repository. Participant privacy is a hard requirement.

## Identifier rules (CRITICAL)

| Context | Allowed | Never allowed |
|---------|---------|--------------|
| Code (variable names, comments) | `P1`–`P4`, `participant_id` | Real names, initials, birth dates |
| Log files | `P1`–`P4` | Real names |
| events.tsv, LSL markers | `P1`–`P4` | Real names |
| BIDS output filenames & content | `sub-P1`–`sub-P4` | Real names |
| Display strings (tablet / bigscreen) | First name only (in-memory, not persisted) | Full names written to disk or LSL |

Real participant names live **exclusively** in `.private/registration_ledger.jsonl` (gitignored).

## Docstring privacy annotation

Any function or method that accepts or processes real participant names must include this annotation:

```python
def _normalize_display_name(real_name: str) -> str:
    """Return first-name-only string for display.

    # Privacy: real names are never written to disk or LSL.
    """
```

## What to check before committing

1. Search for any string that looks like a real name in generated output or new code
2. Confirm no name reaches `logging.*` calls at any level
3. Confirm no name is written into TSV, JSON, or NDJSON files
4. Confirm `.private/` is listed in `.gitignore`

## gitignore — sensitive paths (must remain ignored)

```
.private/
configs/azure_blob_credentials*.json
*.env
```

Do not remove these from `.gitignore`. Do not add credential files to version control under any path.

## Credential handling

- Azure blob credentials: JSON or `.env` format, stored outside the repo or in `.private/`
- Never pass credentials as CLI positional arguments (they appear in shell history)
- Use `--credentials-file` patterns or environment variables for secrets
- Log "credentials loaded" — never log the credential values themselves

## OWASP top concerns for this codebase

| Risk | Mitigation in place |
|------|---------------------|
| Injection (command) | Use `subprocess` list-form only; no `shell=True` with user input |
| Sensitive data exposure | P1–P4 IDs only in all outputs; `.private/` gitignored |
| Insecure design | Fail-loud on missing config; no silent fallbacks that produce wrong data |
| Security misconfiguration | Credentials via files/env vars, never hardcoded |

## Subprocess security rule

Always use list form for subprocess calls; never construct shell strings from user input:

```python
# Good — safe from injection
subprocess.run(["python", str(script_path), "--session", str(session_dir)], check=True)

# Bad — injection risk if session_dir contains shell metacharacters
subprocess.run(f"python {script_path} --session {session_dir}", shell=True)
```
