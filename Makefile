.PHONY: install dataset

install:
	pip install -r requirements.txt
	docker build -t engrafo engrafo/.
	ollama pull mxbai-embed-large
	ollama pull llama3.2:3b

dataset:
	python -m agentsearch.dataset.download
	python -m agentsearch.dataset.convert
	python -m agentsearch.dataset.embed

