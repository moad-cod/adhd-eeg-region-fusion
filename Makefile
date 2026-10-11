DATA := EEG Dataset for ADHD/adhdata.csv
MODEL ?= logistic
RUN_NAME ?= $(subst $(space),_,$(strip $(MODEL)))
FOLDS ?= 5
SEED ?= 42
PYTHON_ENV ?= .venv
MPLCONFIGDIR ?= /tmp/adhd-eeg-matplotlib
EXTRA_ARGS ?=
COMBINATIONS ?=
REGION_RUN_NAME ?= three_region_benchmark

empty :=
space := $(empty) $(empty)
RUN_NAME := $(if $(strip $(RUN_NAME)),$(RUN_NAME),$(subst $(space),_,$(strip $(MODEL))))
REGION_COMBINATION_ARGS := $(if $(strip $(COMBINATIONS)),--combinations $(COMBINATIONS),)

.PHONY: help run smoke regions regions-smoke

help:
	@echo "ADHD EEG experiment commands"
	@echo ""
	@echo "  make run MODEL=random_forest"
	@echo "  make run MODEL=random_forest RUN_NAME=rf_final"
	@echo "  make run MODEL='logistic random_forest' RUN_NAME=comparison"
	@echo "  make smoke MODEL=rbf_svm"
	@echo "  make regions MODEL='random_forest rbf_svm'"
	@echo "  make regions-smoke MODEL=logistic COMBINATIONS='F-C-P F-T-O'"
	@echo ""
	@echo "Optional variables: RUN_NAME, FOLDS, SEED, DATA, PYTHON_ENV, EXTRA_ARGS"

run:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON_ENV)/bin/adhd-eeg-global \
		--data "$(DATA)" \
		--models $(MODEL) \
		--folds $(FOLDS) \
		--random-state $(SEED) \
		--output "outputs/$(RUN_NAME)" $(EXTRA_ARGS)

smoke:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON_ENV)/bin/adhd-eeg-global \
		--data "$(DATA)" \
		--models $(MODEL) \
		--folds 2 \
		--random-state $(SEED) \
		--max-subjects-per-class 5 \
		--output "outputs/smoke_$(RUN_NAME)" $(EXTRA_ARGS)

regions:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON_ENV)/bin/adhd-eeg-three-regions \
		--data "$(DATA)" \
		--models $(MODEL) $(REGION_COMBINATION_ARGS) \
		--folds $(FOLDS) \
		--random-state $(SEED) \
		--output "outputs/$(REGION_RUN_NAME)" $(EXTRA_ARGS)

regions-smoke:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON_ENV)/bin/adhd-eeg-three-regions \
		--data "$(DATA)" \
		--models $(MODEL) \
		--combinations $(if $(strip $(COMBINATIONS)),$(COMBINATIONS),F-C-P F-T-O) \
		--folds 2 \
		--random-state $(SEED) \
		--max-subjects-per-class 5 \
		--output "outputs/smoke_$(REGION_RUN_NAME)" $(EXTRA_ARGS)
