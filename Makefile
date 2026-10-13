.PHONY: install-api test lint api frontend compose

install-api:
	python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -r requirements-ci.txt ruff

test:
	pytest

lint:
	ruff check .

api:
	uvicorn src.api.main:app --reload --port 8000

frontend:
	streamlit run frontend/streamlit_app.py

compose:
	docker compose up --build
