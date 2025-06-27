# Agent Search

## Setup Environment

To run this project, you'll need:

1. **Python:** Environment with the required dependencies
2. **Docker:** Deamon must be running
3. **Ollama:** Running with required models

```
# Create conda environment (optional)
conda create -n agentsearch python=3.13
conda activate agentsearch

# Install dependencies, build docker image, and pull models
make install
```

## Dataset Creation

Creating the dataset is a three-step process. The steps are documented below.

To execute the entire pipeline with a single command, use:

```
make dataset
```

### 1. Download LaTeX Projects

This will download the LaTeX source code (as `.tar.gz`) from arXiv.org for all available papers for a specified set of authors. Files are exported to `papers/src/{authorId}/{arxivPaperId}.tar.gz`.

```
python -m agentsearch.dataset.download
```

### 2. Convert LaTeX to HTML

This will convert all papers in `papers/src` to single HTML files, exported to `papers/html/{authorId}/{arxivPaperId}.html`. The conversion is run in a Docker container and parallelized over all available CPU cores.

```
python -m agentsearch.dataset.convert
```

### 3. Chunking and Embedding

For all available papers in `papers/html`, the papers are chunked into content segments identified by paragraphs. Every paragraph is embedded and stored in `chroma_db`.

```
python -m agentsearch.dataset.embed
```

## Chatbot

A chabot can be started via

```
python -m agentsearch.agent.qa
```
