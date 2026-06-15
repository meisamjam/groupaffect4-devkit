.PHONY: check test format sync-sources progress

check:
	python -m ruff check .
	python -m pytest -q

test:
	python -m pytest -q

format:
	python -m ruff format .

sync-sources:
	python tools/sync_sources.py

progress:
	python tools/progress_report.py
