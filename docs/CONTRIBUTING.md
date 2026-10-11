# Contributing

1. Fork, then `pip install -e .`
2. Run `python -m strataforge doctor` and `python -m unittest discover -s tests -t .`
3. Keep the core zero-dependency; add optional integrations as extras.
4. Every new benchmark must be deterministic (seeded).
