.PHONY: install-api test lint api frontend compose

install-api:
	python -m pip install -r requirements-api.txt pytest ruff

test:
	pytest

lint:
	ruff check src/api tests

api:
	uvicorn src.api.main:app --reload --port 8000

frontend:
	streamlit run frontend/streamlit_app.py

compose:
	docker compose up --build
