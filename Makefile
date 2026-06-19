##@ Configuration

VERSION ?= $(shell uv version --short 2>/dev/null)
GIT_SHA := $(shell git rev-parse --short HEAD)
RELEASE := v$(VERSION)-g$(GIT_SHA)

##@ General

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: version
version: ## Display the current version
	@echo "$(VERSION)"

.PHONY: release
release: ## Display the release (version + git SHA)
	@echo "$(RELEASE)"

##@ Setup

.PHONY: install
install: ## Install dependencies and git hooks
	@echo "--- 📦 Installing dependencies and git hooks"
	uv sync
	uv run lefthook install

.PHONY: install-locked
install-locked: ## Install dependencies exactly as locked (used by CI)
	@echo "--- 📦 Installing locked dependencies"
	uv sync --locked

##@ Build

.PHONY: build
build: ## Build sdist and wheel into dist/
	@echo "--- 🛠️ Building reactor-runtime $(RELEASE)"
	uv build

.PHONY: publish
publish: ## Publish dist/ to PyPI (trusted publishing in CI)
	@echo "--- 📦 Publishing reactor-runtime $(VERSION) to PyPI"
	uv publish

##@ Quality

.PHONY: lint
lint: ## Run ruff lint and format checks
	@echo "--- 🔎 Linting"
	uv run ruff check
	uv run ruff format --check

.PHONY: format
format: ## Format the codebase and fix auto-fixable lints
	uv run ruff format
	uv run ruff check --fix

.PHONY: typecheck
typecheck: ## Run mypy in strict mode
	@echo "--- 🔎 Type checking"
	uv run mypy

##@ Testing

.PHONY: test
test: ## Run unit tests
	@echo "--- 🧪 Running unit tests"
	uv run pytest -q

.PHONY: check
check: lint typecheck test ## Run all checks (lint, typecheck, test)
	@echo "--- ✅ All checks passed"

##@ Cleanup

.PHONY: clean
clean: ## Remove build artifacts and caches
	@echo "--- 🧹 Cleaning build artifacts"
	@rm -rf dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/
