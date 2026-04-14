# PRs to open on GitHub

Go to **<https://github.com/edadaltocg/edadaltocg.github.io/pulls>** and create one PR per branch.

---

## PR 1 — `feat/i18n` → `main`

**Title:** `feat(i18n): implement UI internationalization with trans() calls`

**Body:**

```markdown
## Summary

- Replace hardcoded English strings in all templates with Zola `trans()` calls so the UI language follows the page language automatically
- Add `[translations]` section to `config.toml` and `default_language = "en"` to activate the i18n system
- Add Italian locale (`i18n/it.toml`); remove Dutch locale (`i18n/nl.toml`)
- Extend all existing locales (en/es/fr/pt/zh) with new UI keys: `home`, `menu`, `search_placeholder`, `back_home`, `back_blog`, `toc`, `references`, `comments`, `reading_time_prefix`, `reading_time_unit`

## Test plan

- [ ] Run `zola build` — must complete without errors
- [ ] Verify `zola serve` renders pages with translated UI strings
- [ ] Check that `<html lang="...">` reflects the page language
```

---

## PR 2 — `feat/deploy` → `main`

**Title:** `feat(ci): add deploy pipeline with automated minification`

**Body:**

```markdown
## Summary

- Update `deploy.yml` to trigger only on push to `main` (+ manual dispatch via `workflow_dispatch`)
- Upgrade Zola from 0.17.2 to 0.21.0 in CI
- Add HTML/CSS/JS minification step using `tdewolff/minify` in CI
- Switch deployment to `peaceiris/actions-gh-pages@v4`
- Add `build.yml` PR check: verifies `zola build` passes on every PR to `main`
- Add `just minify` recipe (local `tdewolff/minify` CLI)
- Add `just deploy` recipe: build → minify → push `public/` to `gh-pages` via git worktree

## Test plan

- [ ] Merge to `main` and verify GitHub Actions deploys to `gh-pages`
- [ ] Run `just deploy` locally and confirm `gh-pages` is updated
- [ ] Open a test PR and confirm the build check runs and passes
```

---

## PR 3 — `feat/renovate` → `main`

**Title:** `feat(deps): add Renovate for automated dependency updates`

**Body:**

```markdown
## Summary

- Add `renovate.json` configured to open dependency update PRs every weekend
- Groups all GitHub Actions updates into a single PR
- `automerge` disabled — all PRs require manual approval
- Requires the Renovate GitHub App (https://github.com/apps/renovate) to be installed on the repo

## Test plan

- [ ] Install the Renovate GitHub App on this repository
- [ ] Confirm Renovate opens its onboarding PR
- [ ] Verify no PRs are created on weekdays
```

---

## PR 4 — `feat/tooling` → `main`

**Title:** `feat(tooling): add code formatters, linter, and pre-commit hooks`

**Body:**

```markdown
## Summary

Formatters added:

- **prettier** — CSS, JS, JSON, Markdown (config: `.prettierrc.json`; HTML templates excluded via `.prettierignore`)
- **taplo** — TOML files (config: `taplo.toml`)
- **markdownlint-cli2** — Markdown style rules (config: `.markdownlint.json`; MD013/MD033/MD041 disabled for prose and front matter)
- **bibtex-tidy** — BibTeX bibliography files

Pre-commit hook:

- `.pre-commit-config.yaml` runs all formatters on staged files before every commit
- Run `just install` to register the hook locally via `pre-commit install`

CI:

- `build.yml` gains a `format` job that runs all format checks on every PR to `main`

Also applies all formatters to existing files (CSS, JS, JSON, BibTeX).

## Test plan

- [ ] Run `just install` and confirm pre-commit hook is registered
- [ ] Make a formatting change and confirm the hook blocks the commit
- [ ] Run `just format-check` — must pass with no errors
- [ ] Open a test PR and confirm the `format` CI job passes
```

---

## Notes

- **Merge order:** `feat/tooling` and `feat/deploy` both add `build.yml` — merge one first, then resolve the trivial conflict on the other. `feat/i18n` and `feat/renovate` are independent.
