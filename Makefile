.PHONY: run test lint clean setup data train

setup:
	pip install -r requirements.txt
	pre-commit install

data:
	python data/generate_data.py
	python data/validate_data.py

train:
	python features/engineering.py
	python models/churn_model.py
	python models/uplift_model.py

run:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v --tb=short

lint:
	black .
	flake8 . --max-line-length=100 --exclude=__pycache__,.env
	isort .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

all: setup data train run
