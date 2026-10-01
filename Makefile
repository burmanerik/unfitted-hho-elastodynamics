# Convenience targets; see README.md for the full description.
PY ?= python3

.PHONY: help test figures all clean

help:
	@echo "make test     run the verification suite (about one minute)"
	@echo "make figures  rebuild every table and figure from results/"
	@echo "make all      recompute everything from scratch (several hours)"
	@echo "make clean    remove the generated tables and figures"

test:
	$(PY) -m pytest tests -q

figures:
	$(PY) make_figures.py

all:
	$(PY) produce.py --all
	$(PY) make_figures.py

clean:
	rm -f paper/tab/*.tex paper/fig/*.pdf
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
