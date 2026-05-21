install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

setup:
	.venv/bin/python -m src.data.courtlistener
	.venv/bin/python -m src.data.uscode
	.venv/bin/python -m src.data.ecfr
	.venv/bin/python -m src.data.preprocessor
	.venv/bin/python -m src.rag.indexer

setup-full:
	# Full ingestion: fetch all available raw data, preprocess, and index
	.venv/bin/python scripts/setup_full.py

app:
	.venv/bin/streamlit run streamlit_app.py

benchmark:
	.venv/bin/python -m src.evaluation.benchmarks

test:
	.venv/bin/pytest tests/ -v --tb=short

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format --check src/ tests/

clean:
	rm -rf chroma_db/ data/raw/ data/processed/ logs/ .venv/

# Custom setup target: allow overriding counts via make variables
# Usage: `make setup-custom CASES=1000 GRANULES=500 ECFR_TITLES=5`
CASES ?= 1000
GRANULES ?= 500
ECFR_TITLES ?= 5

setup-custom:
	.venv/bin/python scripts/setup_full.py --cases $(CASES) --granules $(GRANULES) --ecfr-titles $(ECFR_TITLES)
