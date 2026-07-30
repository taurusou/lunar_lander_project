# Tests

Future tests should check small, important behaviors such as action selection,
replay-buffer sampling, neural-network output shapes, saved-model loading, and
deterministic evaluation with fixed seeds.

Run the current test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

The suite covers component behavior and small training-loop helpers. It does
not run the full 25-episode development training.
