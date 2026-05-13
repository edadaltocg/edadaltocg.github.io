# My personal website

## Tech

- [Zola](https://www.getzola.org/) — static site generator; configured in `config.toml` with multilingual builds (en/fr/es/it/pt/zh), Atom feeds, sitemap, robots.txt, taxonomies, and an elasticlunr search index
- [KaTeX](https://katex.org/) — math rendering, enabled via `extra.enable_katex`
- [Sass](https://sass-lang.com/) — stylesheets compiled by Zola (`compile_sass = true`); syntax highlighting uses `ayu-dark`/`ayu-light` themes
- [Matplotlib](https://matplotlib.org/) — blog plots rendered as SVG by `generate_plots.py`, run with [uv](https://docs.astral.sh/uv/)
- [Pandoc](https://pandoc.org/) + `citeproc` — renders `publications/publications.bib` to HTML for the publications page
- [Font Awesome](https://fontawesome.com/) — social and UI icons via CDN
- PWA assets — `manifest.min.json` and favicons served from `static/`
- GitHub Pages — `main` is built by CI and published to `gh-pages`

## Tooling

All commands are wrapped in the `justfile`:

- **`just build`** (`b`) — runs `just plots` then `zola build`
- **`just preview`** (`p`) — `zola serve --open` for live reload
- **`just plots`** — regenerates blog SVGs via `uv run generate_plots.py`
- **`just lint`** (`l`) — runs every pre-commit hook against all files
- **`just install`** (`i`) — `brew install zola pre-commit` and registers the hooks
- **`just check-refs`** — `uv run check_refs.py` to validate citations against the `.bib` files
- **`just bib2html`** — Pandoc + citeproc render of `publications/publications.bib`
- **`just notebook2markdown <path>`** — `jupyter-nbconvert` to convert notebooks into blog markdown
- **`just optimize_media`** — `pngquant`, `oxipng`, `leanify`, `ffmpeg`, and Ghostscript pipeline for shipping smaller images, audio, and PDFs

Pre-commit hooks (see `.pre-commit-config.yaml`):

- [prettier](https://prettier.io/) — formats markdown, HTML, JS, CSS, YAML
- [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2) — markdown linting with `--fix`
- [ruff](https://docs.astral.sh/ruff/) — Python lint and format for `generate_plots.py` / `check_refs.py`
- [typos](https://github.com/crate-ci/typos) — spell-checks markdown and `i18n/*.toml`
- [taplo](https://taplo.tamasfe.dev/) — TOML formatter (`config.toml`, `taplo.toml`)
- [bibtex-tidy](https://github.com/FlamingTempura/bibtex-tidy) — normalizes `.bib` entries

## Deploy

Push to `main`. CI builds with Zola, minifies HTML/CSS/JS, and publishes to the `gh-pages` branch.

## TODO

- [ ] Dark mode/light mode button
