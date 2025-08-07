.PHONY: install dataset clean

clean:
	rm papers/*.log

install:
	pip install -r requirements.txt
	docker build -t engrafo engrafo/.
	ollama pull nomic-embed-text

