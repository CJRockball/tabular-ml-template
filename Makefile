# Makefile - project pipeline, one command end to end.
PKG    ?= ml_template
DS     ?= ds_v1
CONFIG ?= configs/ai4i_binary.yaml
EXP    ?= configs/experiment/xgb_baseline.yaml
SWEEP  ?= configs/sweep/xgb_grid.yaml
RUN    := uv run python

.PHONY: ingest eda silver split features train sweep evaluate pipeline clean rebuild check

ingest:
	$(RUN) -m $(PKG).data.ingest --data_file $(CONFIG) --dataset_version_id $(DS)
eda:
	$(RUN) -m $(PKG).scripts.eda_script --dataset-version-id $(DS)
silver:
	$(RUN) -m $(PKG).data.silver --dataset-version-id $(DS)
split:
	$(RUN) -m $(PKG).training.split --config $(EXP)
features:
	$(RUN) -m $(PKG).features.build_features --config $(EXP) \
	--blocks $(BLOCKS) $(if $(PARENT),--parent $(PARENT),) $(if $(COLUMNS),--columns $(COLUMNS),)  \
	$(if $(BLOCK_PARAMS),--block-params $(BLOCK_PARAMS),)
train:
	$(RUN) -m $(PKG).training.train_model --config $(EXP)
sweep:
	$(RUN) -m $(PKG).training.train_model --sweep $(SWEEP)
evaluate:
	$(RUN) -m $(PKG).evaluation.evaluate_run --run-id $(RUN_ID)

pipeline: ingest eda silver split train
rebuild: clean pipeline

clean:
	find data -mindepth 1 -maxdepth 1 ! -name .gitignore ! -name raw -exec rm -rf {} +
	find artifacts logs assets -mindepth 1 ! -name .gitkeep ! -name .gitignore -delete 2>/dev/null || true

check:
	uv run ruff check && uv run pytest -q