DATA := EEG Dataset for ADHD/adhdata.csv
MODEL ?= logistic
RUN_NAME ?= $(subst $(space),_,$(strip $(MODEL)))
FOLDS ?= 5
SEED ?= 42
PYTHON_ENV ?= .venv
EXTRA_ARGS ?=

empty :=
space := $(empty) $(empty)
RUN_NAME := $(if $(strip $(RUN_NAME)),$(RUN_NAME),$(subst $(space),_,$(strip $(MODEL))))

.PHONY: help run smoke

help:
	@echo "ADHD EEG experiment commands"
	@echo ""
	@echo "  make run MODEL=random_forest"
	@echo "  make run MODEL=random_forest RUN_NAME=rf_final"
	@echo "  make run MODEL='logistic random_forest' RUN_NAME=comparison"
	@echo "  make smoke MODEL=rbf_svm"
	@echo ""
	@echo "Optional variables: RUN_NAME, FOLDS, SEED, DATA, PYTHON_ENV, EXTRA_ARGS"

run:
	$(PYTHON_ENV)/bin/adhd-eeg-global \
		--data "$(DATA)" \
		--models $(MODEL) \
		--folds $(FOLDS) \
		--random-state $(SEED) \
		--output "outputs/$(RUN_NAME)" $(EXTRA_ARGS)

smoke:
	$(PYTHON_ENV)/bin/adhd-eeg-global \
		--data "$(DATA)" \
		--models $(MODEL) \
		--folds 2 \
		--random-state $(SEED) \
		--max-subjects-per-class 5 \
		--output "outputs/smoke_$(RUN_NAME)" $(EXTRA_ARGS)
