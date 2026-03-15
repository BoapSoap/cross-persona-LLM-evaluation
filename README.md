# Cross-Persona Relationship Verifier

This repository now uses an LLM-first architecture for cross-persona verification. Python handles loading, schema normalization, selective pair extraction, request packaging, structured-response validation, and report writing. The OpenAI model is the source of truth for `Pass`, `Warning`, and `Fail`.

Default run:

```bash
python3 main.py --output output/social_world_verification.json
```

Dry-run mode:

```bash
python3 main.py --dry-run --output output/social_world_verification.json
```

Environment:

```bash
export OPENAI_API_KEY=...
```

Notes:

- The scope is cross-persona verification only.
- Put your input file at `data/input/social_world.json` for the default path.
- You can also pass `--input some_file.json`; if that file is not found, the loader will also check `data/input/some_file.json`.
- The real dataset schema is inferred from `social_world.json`, not from `persona.json` in `Downloads` because that file is a commented reference spec rather than valid JSON.
- The prompt in `prompts/pair_eval_prompt.txt` is the evaluation source of truth.
- `rule_checks.py` is no longer part of the main execution flow.
