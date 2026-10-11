.PHONY: doctor train bench serve test install

install:
	pip install -e .

doctor:
	python -m strataforge doctor

train:
	python -m strataforge train --steps 6

bench:
	python -m strataforge bench

serve:
	python -m strataforge serve --host 0.0.0.0 --port 8080

test:
	python -m unittest discover -s tests -t .
