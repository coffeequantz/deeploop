# Brief: fix the demo project

The small Python project in this folder (`mathlib.py`, `test_mathlib.py`) has a
failing test suite. `add()` returns the wrong value, and the tests catch it.

## What done looks like

- `pytest -q` passes in this folder.
- The existing tests are unchanged: do not delete, skip, or loosen them.
- No new dependencies are added; this project must stay stdlib-only.

## Context

- `context/notes.md` describes how the bug was introduced.
- `assets/status.png` is a screenshot of the failing run for reference.

## Non-goals

- Do not refactor the module into a package.
- Do not add type annotations or docstrings.
