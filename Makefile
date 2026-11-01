# One command per step. Run `make all` once on a laptop, then `make app`.
#   make setup     create the virtual environment and install everything
#   make all       download + clean data, categories, segments, model   (about 4 minutes)
#   make eval      measure the AI analyst (needs GROQ_API_KEY in .env)
#   make test      run the unit tests
#   make app       start the dashboard at http://localhost:8501

PY ?= .venv/bin/python
export PYTHONPATH := src:.

.PHONY: setup data products segments model all eval results test app

setup:
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements-train.txt
	$(PY) -m pip install -e .

data:
	$(PY) -m rci.pipeline
	$(PY) -m rci.quality

products:
	$(PY) -m rci.products

segments:
	$(PY) -m rci.segments

model:
	$(PY) -m rci.model

all: data products segments model

eval:
	$(PY) -m evals.run_analyst_eval

results:
	$(PY) scripts/render_results.py

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app/Overview.py
