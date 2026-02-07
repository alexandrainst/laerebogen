# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

dataset: ## Generate the dataset
	@echo "Generating the dataset..."
	@uv run src/scripts/generate_base_dataset.py && \
		uv run src/scripts/correct_grammar_in_dataset.py && \
		uv run src/scripts/correct_quality_in_dataset.py && \
		uv run src/scripts/evolve_dataset.py && \
		uv run src/scripts/add_follow_ups.py && \
		uv run src/scripts/push_to_hub.py
	@echo "Dataset generated successfully!"

install: ## Install dependencies
	@echo "Installing the 'lærebogen' project..."
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet install-pre-commit
	@echo "Installed the 'lærebogen' project! You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a 'uv' project. Use 'uv add <package>' to install new dependencies and 'uv remove <package>' to remove them."

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		if [ "$(shell which rustup)" = "" ]; then \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			echo "Installed Rust."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update || true; \
	fi

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

install-dependencies:
	@uv python install 3.12
	@uv sync --all-extras --python 3.12

lint:  ## Lint the project
	uv run ruff check . --fix --unsafe-fixes

format:  ## Format the project
	uv run ruff format .

type-check:  ## Type-check the project
	@uv run mypy . \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--show-error-codes \
		--check-untyped-defs

check: lint format type-check  ## Lint, format, and type-check the code
