alias i := install
alias b := build
alias p := preview
alias f := format
alias fmt := format-check
alias l := lint

default:
  just --list

install:
  brew install zola pre-commit
  pre-commit install

build:
  zola build

preview:
  zola serve --open

optimize_media:
  pngquant --skip-if-larger --strip --quality=93-93 --speed 1 *.png
  oxipng -o max --strip all -a -Z *.png
  leanify -i 7777 *.png
  leanify -i 7777 *.ico
  ffmpeg -i thesis_notebooklm.wav -ar 16000 -b:a 32000 -ac 1 thesis_notebooklm.opus
  gs -sDEVICE=pdfwrite -dNOPAUSE -dQUIET -dBATCH -dPDFSETTINGS=/screen -dCompatibilityLevel=1.4 -sOutputFile=output.pdf input.pdf

minify:
  #!/usr/bin/env bash
  set -euo pipefail
  command -v minify >/dev/null 2>&1 || { echo "minify not found. Install with: brew install minify"; exit 1; }
  find public -name "*.html" -exec minify --type=html -o {} {} \;
  find public -name "*.css" -not -name "*.min.css" -exec minify --type=css -o {} {} \;
  find public -name "*.js" -not -name "*.min.js" -exec minify --type=js -o {} {} \;

PANDOC_INPUT := "publications/pandoc-input.md"
OUTPUT_HTML := "out/list.html"
BIBLIOGRAPHY := "publications/publications.bib"

bib2html:
  mkdir -p out
  pandoc --citeproc {{PANDOC_INPUT}} -o {{OUTPUT_HTML}} --bibliography={{BIBLIOGRAPHY}}

notebook2markdown path:
  jupyter-nbconvert --to markdown {{path}}

lint:
  pre-commit run --all-files --verbose

format:
  prettier --write .
  taplo fmt
  markdownlint-cli2 --fix "**/*.md"
  find publications -name "*.bib" -exec bibtex-tidy --modify {} \;

format-check:
  prettier --check .
  taplo fmt --check
  markdownlint-cli2 "**/*.md"
  find publications -name "*.bib" -exec bibtex-tidy --no-modify {} \;

