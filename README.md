# Codex ML Skill

This repo packages a Codex skill for building classical ML models from
local files or HTTP(S) datasets.

## Contents
- `skills/ml-model-builder/` - skill source and references
- `dist/ml-model-builder.skill` - packaged skill archive
- `SPEC_full.md` - full project specification

## Rebuild the package
From the repo root:

```sh
cd skills
python3 -m zipfile -c ../dist/ml-model-builder.skill ml-model-builder
```

## Notes
- Local artifacts and virtualenvs should stay untracked (see `.gitignore`).
