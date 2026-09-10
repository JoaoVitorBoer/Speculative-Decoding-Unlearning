.PHONY: quality style test summaries dashboard figures

check_dirs := scripts src #setup.py

quality:
	ruff check $(check_dirs) setup.py setup_data.py
	ruff format --check $(check_dirs) setup.py setup_data.py

style:
	ruff check $(check_dirs) setup.py setup_data.py --fix
	ruff format $(check_dirs) setup.py setup_data.py

test:
	CUDA_VISIBLE_DEVICES= pytest tests/

# ── results dashboard ────────────────────────────────────────────────────────
summaries:
	python scripts/summarize_sud_results.py
	python scripts/summarize_sud_baselines.py

dashboard: summaries
	conda run -n unlearning streamlit run dashboard/app.py

figures: summaries
	conda run -n unlearning python scripts/plot_results.py
