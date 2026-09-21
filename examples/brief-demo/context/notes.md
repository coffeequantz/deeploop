# Notes on the regression

`add()` was changed from `a + b` to `a - b` during an experiment with a "difference"
helper. The experiment was abandoned but the change was never reverted.

`multiply()` is unrelated and correct.

The suite is run with `pytest -q` from this folder.
