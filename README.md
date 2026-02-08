# mdq.py

A command-line tool for semantic search on plaintext files using local LLM
langauge embeddings. Uses [fastembed](https://qdrant.github.io/fastembed/) with
a SQLite cache.

## Motivation

Functionally, this is simonw's [llm](https://github.com/simonw/llm) tool, minus
all the features I don't care about:

- `mdq` can only use local LLMs, not web services
- `mdq` only computes embeddings for the prompt and input documents; there's no
  chat
- `mdq` uses [sqlite_vec](https://github.com/asg017/sqlite-vec) for vector
  search; there is no reranker

## Usage

```shell
# Supply query as either flag or stdin
mdq -q "sasquatch"
echo "sasquatch" | mdq

# Specify paths to search
mdq -q "sasquatch" -p ./notes/*.md ./docs/*.md

# Match only specific file extensions
mdq -q "sasquatch" -e "txt" "tex" -p ./notes

# Adjust the number of matches
mdq -q "sasquatch" -k 20
```

## Installation

Clone the repo, then run:

```shell
pipx install .
```
