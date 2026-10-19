# Makefile - project pipeline, one command end to end.
PKG    ?= ml_template
DS     ?= ds_v1
CONFIG ?= configs/ai4i_binary.yaml
EXP    ?= configs/experiment/xgb_baseline.yaml
SWEEP  ?= configs/sweep/xgb_grid.yaml
RUN    := uv run python
CALIBRATION_CONFIG ?= configs/calibration/test_958bec.yaml
EVALUATION_CONFIG ?= configs/evaluation/test_958bec.yaml
RELEASE_CONFIG ?= configs/release/example_958bec.yaml
RELEASE_DIR ?=
SCORE_INPUT ?=
SCORE_ARGS ?=

.PHONY: ingest eda silver split features train sweep evaluate pipeline clean rebuild check dashboard calibrate calibrate-check calibration-tests evaluate-check evaluation-tests release-check release score-check score scoring-tests

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
dashboard:
	$(RUN) -m $(PKG).dashboard.app
calibrate:
	$(RUN) -m $(PKG).training.calibrate --config $(CALIBRATION_CONFIG)
calibrate-check:
	$(RUN) -m $(PKG).training.calibrate --config $(CALIBRATION_CONFIG) --check
calibration-tests:
	$(RUN) -m pytest -q tests/test_probability_calibrator.py tests/test_calibration_*.py
evaluate-check:
	$(RUN) -m $(PKG).evaluation.evaluate_run --config $(EVALUATION_CONFIG) --check
evaluate:
	$(RUN) -m $(PKG).evaluation.evaluate_run --config $(EVALUATION_CONFIG) --execute
evaluation-tests:
	$(RUN) -m pytest -q tests/evaluation/test_final_evaluation_*.py tests/dashboard/test_final_evaluation_*.py

pipeline: ingest eda silver split train
rebuild: clean pipeline

clean:
	# Preserve datasets, project.db, model provenance, releases and scored batches.
	rm -rf .pytest_cache .ruff_cache .mypy_cache

check:
	uv run ruff check && uv run pytest -q

release-check:
	$(RUN) -m $(PKG).scoring.release_builder --config "$(RELEASE_CONFIG)" --check
release:
	$(RUN) -m $(PKG).scoring.release_builder --config "$(RELEASE_CONFIG)" --execute
score-check:
	@test -n "$(RELEASE_DIR)" -a -n "$(SCORE_INPUT)" || (echo 'Set RELEASE_DIR and SCORE_INPUT'; exit 2)
	$(RUN) -m $(PKG).scoring.batch --release "$(RELEASE_DIR)" --input "$(SCORE_INPUT)" --check $(SCORE_ARGS)
score:
	@test -n "$(RELEASE_DIR)" -a -n "$(SCORE_INPUT)" || (echo 'Set RELEASE_DIR and SCORE_INPUT'; exit 2)
	$(RUN) -m $(PKG).scoring.batch --release "$(RELEASE_DIR)" --input "$(SCORE_INPUT)" --execute $(SCORE_ARGS)
scoring-tests:
	$(RUN) -m pytest -q tests/scoring
