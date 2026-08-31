.PHONY: demo full visuals test

demo:
	PYTHONPATH=src python3 -m sos_rca.cli --rows 100000 --output artifacts-demo

full:
	PYTHONPATH=src python3 -m sos_rca.cli --rows 847000 --output artifacts

visuals: full
	python3 scripts/generate_readme_visuals.py

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
