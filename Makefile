.PHONY: setup test ingest ask eval eval-accept eval-report ablations scorecard-png serve clean

setup:
	pip install -r requirements-dev.txt
	cp -n project.config.example.json project.config.json || true
	cp -n .env.example .env || true

test:
	python -m pytest tests -q
	python -m mypy core/eval --strict --follow-imports=silent
	python -m ruff check core/eval

ingest:
	python -m app.cli ingest $(or $(PATH_),data/sample)

ask:
	python -m app.cli ask "$(Q)"

RAG_ADAPTER := app.evaluation.rag_adapter:build_adapter

# The command to run before and after every change you make for a client.
eval:
	python -m core.eval run --suite eval/data/golden.jsonl --adapter $(RAG_ADAPTER) \
		--out eval/report --label $(or $(LABEL),baseline) --baseline eval/baseline.json

eval-accept:
	python -m core.eval accept --suite eval/data/golden.jsonl --adapter $(RAG_ADAPTER) \
		--baseline eval/baseline.json

# Self-contained HTML report: scorecard, ablation table, per-slice breakdown, worst cases.
eval-report:
	python -m core.eval.html_report --scorecard eval/report/latest.json \
		--ablations eval/report/ablations.json --out eval/report/report.html

# Chunking x retrieval x reranker matrix. Retrieval-only, no LLM calls -- see
# eval/ablations.py for why that's the right scope.
ablations:
	python -m eval.ablations --out eval/report

# Upwork portfolio thumbnail.
scorecard-png:
	python -m eval.scorecard_png --scorecard eval/report/latest.json --out eval/report/scorecard.png

serve:
	uvicorn app.api:app --reload --port 8000

clean:
	rm -rf data/index.db data/ingest_state.json traces eval/report eval/.judge_cache.sqlite \
		results .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
