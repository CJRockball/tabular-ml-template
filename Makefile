# Makefile - project pipeline, one command end to end.
PKG    ?= ml_template
DS     ?= ds_v1
CONFIG ?= configs/ai4i_binary.yaml
RUN    := uv run python

.PHONY: ingest eda silver pipeline clean rebuild check

ingest:
	$(RUN) -m $(PKG).data.ingest --data_file $(CONFIG) --dataset_version_id $(DS)
eda:
	$(RUN) -m $(PKG).scripts.eda_script --dataset-version-id $(DS)
silver:
	$(RUN) -m $(PKG).data.silver --dataset-version-id $(DS)
train:
	$(RUN) -m $(PKG).training.train_model --config configs/experiment/xgb_baseline.yaml
sweep:
	$(RUN) -m $(PKG).training.train_model --sweep configs/sweep/xgb_grid.yaml
evaluate:
	$(RUN) -m $(PKG).evaluation.evaluate_run --run-id $(RUN_ID)

pipeline: ingest eda silver split train
rebuild: clean pipeline

clean:
	find data -mindepth 1 -maxdepth 1 ! -name .gitignore ! -name raw -exec rm -rf {} +
	find artifacts logs assets -mindepth 1 ! -name .gitkeep ! -name .gitignore -delete 2>/dev/null || true

check:
	uv run ruff check && uv run pytest -q

