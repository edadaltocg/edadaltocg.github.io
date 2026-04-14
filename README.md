# My personal website

## Tech

- [Zola](https://www.getzola.org/)

## Tooling

- **Formatter:** `just format` — runs prettier (CSS/JS/JSON/MD), taplo (TOML), markdownlint-cli2 (Markdown), bibtex-tidy (BibTeX)
- **Spell checker:** typos — auto-fixes typos in Markdown files via pre-commit
- **Pre-commit:** `just install` registers all hooks; runs on every commit and on PRs to `main`
- **Deploy:** `just deploy` — builds, minifies HTML/CSS/JS, pushes to `gh-pages`

## TODO

- [x] Publications
- [x] Social links
- [x] Internationalization
- [x] Spell checker
- [ ] Dark mode/light mode button
- [ ] Blog: From Zero
