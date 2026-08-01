# Contributing

## Notebook outputs

Jupyter notebooks must be committed without cell outputs or execution counts.
This keeps reviews readable and prevents generated data from creating large Git
diffs. The final report notebook may retain output only after its exact path
has been added to `.notebook-output-allowlist`.

Before staging a notebook, clear it with either Python or `uv`:

```bash
python scripts/clean_notebooks.py --fix
# or
uv run python scripts/clean_notebooks.py --fix
```

Enable the repository's optional pre-commit hook once after cloning:

```bash
git config core.hooksPath .githooks
```

The hook checks the notebook content staged for commit and rejects outputs
without modifying or staging files. GitHub Actions performs the same check on
every push and pull request, so the policy does not depend on installing
`pre-commit`, `nbstripout`, Jupyter, or any other Python package.
