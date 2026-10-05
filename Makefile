# Shortcuts. Every target is just a `python -m hybridrec ...` command.
DATASET ?= ml-1m
PY ?= python

.PHONY: install data tune evaluate all serve test

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m hybridrec download --dataset $(DATASET)

tune: data
	$(PY) -m hybridrec tune --dataset $(DATASET) --als

evaluate: data
	$(PY) -m hybridrec evaluate --dataset $(DATASET)

all: data tune evaluate

serve:
	$(PY) -m hybridrec serve --dataset $(DATASET)

test:
	$(PY) -m pytest -q
