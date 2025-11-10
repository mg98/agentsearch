.PHONY: install dataset clean

clean:
	rm papers/*.log

install:
	pip install -r requirements.txt
	ollama pull nomic-embed-text

