.PHONY: install test lint format clean airflow mlflow api streamlit docker-build

install:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf .coverage htmlcov/

airflow:
	export AIRFLOW_HOME=$(PWD)/airflow && airflow standalone

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

api:
	uvicorn src.api.main:app --reload --port 8000

streamlit:
	streamlit run streamlit_app/app.py

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up