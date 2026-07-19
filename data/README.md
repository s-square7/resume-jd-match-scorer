# Data

## Files

| File | Tracked | Description |
|:-----|:-------:|:------------|
| `train_sample.jsonl` | yes | 800-record sample, enough for a smoke run of the full pipeline |
| `train.jsonl` | no (gitignored) | The full training corpus — place it here yourself |

## Record schema

Each line is one JSON object:

```json
{
  "Job-Description": "## Job Title ... ## Job Summary ... ## Required Skill ...",
  "Resume": "...",
  "Skills": ["Attention to Detail", "Time Management", "..."]
}
```

- `Job-Description` — markdown-structured posting (title, summary, required skills).
- `Resume` — the ground-truth matching résumé for that posting.
- `Skills` — the annotated skill list, used both for label construction and for the
  skill-coverage explainer.

The corpus spans **10,000+ distinct skills across many domains** — healthcare,
skilled trades, administration, logistics — not only software roles. Any
skill-matching logic added to this project must therefore be data-driven rather
than a hardcoded technology list.

## Getting the full dataset

`train.jsonl` is not committed because of its size. If you have the original
`train.txt` from the programme materials, rename it:

```bash
mv train.txt data/train.jsonl
```

`src/train.py` prefers `data/train.jsonl` and silently falls back to
`data/train_sample.jsonl` when it is absent, so nothing breaks either way.
