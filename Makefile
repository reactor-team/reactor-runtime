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
install: ## Install dependencies, vendor the pinned wire bindings, and install git hooks
	@echo "--- 📦 Installing dependencies and git hooks"
	uv sync
	$(MAKE) wire-vendor
	uv run lefthook install

.PHONY: install-locked
install-locked: ## Install dependencies exactly as locked (used by CI)
	@echo "--- 📦 Installing locked dependencies"
	uv sync --locked

##@ Protobuf

.PHONY: proto-gen
proto-gen: ## Generate the wire-protocol bindings from proto/ into src/ (dev only)
	@echo "--- 🛠️ Generating wire-protocol bindings"
	buf generate

.PHONY: proto-lint
proto-lint: ## Lint the proto sources
	@echo "--- 🔎 Linting protos"
	buf lint

.PHONY: proto-breaking
proto-breaking: ## Detect breaking proto changes against main (skips if main has no protos)
	@base=main; git rev-parse --verify --quiet "$$base" >/dev/null 2>&1 || base=origin/main; \
	if git cat-file -e "$$base:buf.yaml" 2>/dev/null; then \
		echo "--- 🔎 Checking protos for breaking changes against $$base"; \
		buf breaking --against ".git#ref=$$base"; \
	else \
		echo "--- ⏭️  No proto module on $$base yet; skipping breaking check"; \
	fi

.PHONY: proto-breaking-release
proto-breaking-release: ## Guard the published contract: breaking-check against the latest wire/v* tag
	@tag=$$(git tag --list 'wire/v*' --sort=-v:refname | head -n1); \
	if [ -z "$$tag" ]; then \
		echo "--- ⏭️  No prior wire/v* release tag; skipping release breaking check"; \
	else \
		echo "--- 🔎 Checking protos for breaking changes against $$tag"; \
		buf breaking --against ".git#tag=$$tag"; \
	fi

.PHONY: wire-vendor
wire-vendor: ## Download the pinned wire/v* release and vendor its bindings into src/
	@echo "--- 📥 Vendoring pinned wire-protocol bindings"
	uv run python scripts/fetch-wire.py

##@ Build

.PHONY: build
build: wire-vendor ## Build sdist and wheel into dist/ (bundles the vendored wire bindings)
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
check: proto-lint proto-breaking lint typecheck test ## Run all checks
	@echo "--- ✅ All checks passed"

##@ Cleanup

.PHONY: clean
clean: ## Remove build artifacts and caches
	@echo "--- 🧹 Cleaning build artifacts"
	@rm -rf dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/
