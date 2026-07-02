# =============================================================================
# CANONICAL mise-shim Makefile - DO NOT EDIT, DO NOT ADD MODULE TARGETS.
#
# Every ./Makefile in this repo is a symlink to Makefile.mise. Module targets
# are DISCOVERED from mise at parse time (one phony rule per task), so adding or
# removing a task in mise.toml or mise-tasks/ changes `make` with no edit here.
# To change a target, edit the module's mise tasks - never hand-edit this file.
# The only built-in targets are the bridge affordances `help` and `tool-install`
# (thin wrappers over mise).
#
# Module-scoped: run from a directory it exposes THAT config root's tasks, so
# `make` at the repo root drives the root tasks and `make help` in proto/ lists
# the proto module's tasks.
#
# Shim-free by design: every recipe invokes `mise` explicitly, so it works in
# shells with no mise activation (e.g. Warp). The only hard dependency is the
# `mise` binary on PATH.
# =============================================================================

# --- Preflight: fail fast at parse time with actionable fixes. Kept inline (not
# deferred to `make doctor`) so a broken environment errors before any target
# runs. Generic across modules: no module-specific tool names.

# 1. mise installed (located without relying on shims/activation).
MISE := $(shell command -v mise 2>/dev/null)
ifeq ($(MISE),)
$(info )
$(info mise is not installed.)
$(info )
$(info    # install mise (see https://mise.jdx.dev), then:)
$(info    mise trust    # trust this project's config)
$(info    mise install  # install the pinned tools)
$(info )
$(error Aborting: install mise first)
endif

# 2. mise trusts this project (mise.toml is not blocked).
_MISE_TRUSTED := $(shell $(MISE) trust --show 2>/dev/null | grep -qw trusted && echo yes)
ifneq ($(_MISE_TRUSTED),yes)
$(info )
$(info mise has not trusted this project yet.)
$(info )
$(info    mise trust && mise install)
$(info )
$(error Aborting: run 'mise trust' first)
endif

# 3. Pinned tools installed - generic: ANY specified-but-uninstalled tool, not a
#    hardcoded binary. Empty `mise ls --missing` means everything is installed.
#    Advisory, not fatal: mise auto-installs missing tools on demand when a task
#    runs, so a hard stop here would be wrong - and in CI the toolchain is baked
#    into the image, where only a subset is exercised per job. Warn and point at
#    the fix (`make tool-install`); let the target proceed.
_MISE_MISSING := $(shell $(MISE) ls --missing 2>/dev/null)
ifneq ($(_MISE_MISSING),)
$(warning Some mise-pinned tools are not installed; run 'make tool-install' (or 'mise install') if a target fails to find one.)
endif

# FORCE=1 -> bypass mise sources/outputs caching for this invocation.
_FORCE := $(if $(filter 1,$(FORCE)),--force,)

# Discover this module's tasks: list as JSON here (cwd = this module dir), then
# pipe into the hidden, always-rooted filter task - a pure stdin filter, so no
# nested subprocess. It prints "<target>|<run name>" pairs; keep only words
# containing '|' so any mise runner chatter drops.
_RAW := $(shell $(MISE) tasks ls -l -J 2>/dev/null | $(MISE) run //:_extract_local_tasks -- $(CURDIR) 2>/dev/null)
_PAIRS := $(foreach w,$(_RAW),$(if $(findstring |,$(w)),$(w)))
_TARGETS := $(foreach pr,$(_PAIRS),$(word 1,$(subst |, ,$(pr))))

.PHONY: $(_TARGETS) help tool-install

# One phony rule per discovered task. The make target is word 1; the mise run
# name (full //module:verb address) is word 2, so it resolves from any cwd.
define _mise_rule
$(word 1,$(subst |, ,$(1))):
	@$(MISE) run $(_FORCE) $(word 2,$(subst |, ,$(1)))
endef
$(foreach pr,$(filter-out help|% tool-install|%,$(_PAIRS)),$(eval $(call _mise_rule,$(pr))))

# Rich, grouped help: pipe the JSON listing (cwd = this module dir) into the
# hidden renderer, which prints a `Usage:` header, cyan padded names,
# descriptions, and ':'-namespace sections.
help:
	@$(MISE) tasks ls -l -J 2>/dev/null | $(MISE) run //:_render_help -- $(CURDIR)

# Bridge affordance: install this module's pinned mise tools so `make` users get
# a one-liner without needing to know mise. cwd is the module dir, so this
# installs the tools resolved for this config_root (root [tools] cascade plus any
# module-local pins).
tool-install:
	@$(MISE) install

.DEFAULT_GOAL := help
