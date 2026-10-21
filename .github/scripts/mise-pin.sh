#!/usr/bin/env bash
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
set -euo pipefail

case "${RUNNER_OS:-}/${RUNNER_ARCH:-}" in
  Linux/X64) platform=linux_x64 ;;
  Linux/ARM64) platform=linux_arm64 ;;
  macOS/X64) platform=macos_x64 ;;
  macOS/ARM64) platform=macos_arm64 ;;
  Windows/X64) platform=windows_x64 ;;
  *) echo "mise pin: unsupported runner OS or architecture" >&2; exit 1 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
awk -v checksum_key="mise_sha256_$platform" '
  function fail(message) {
    print "mise pin: " message > "/dev/stderr"
    failed = 1
    exit 1
  }
  function read_value(line, key) {
    if (++counts[key] != 1) fail("duplicate " key)
    sub(/^[^=]*=[ \t]*/, "", line)
    if (line !~ /^"[^"]*"[ \t]*(#.*)?$/) fail("invalid " key)
    sub(/^"/, "", line)
    sub(/"[ \t]*(#.*)?$/, "", line)
    return line
  }
  { sub(/\r$/, "") }
  /^[ \t]*\[/ {
    section = ($0 ~ /^[ \t]*\[_\][ \t]*(#.*)?$/) ? "metadata" : "other"
    next
  }
  section == "" && /^[ \t]*min_version[ \t]*=/ {
    version = read_value($0, "min_version")
    if (version !~ /^[0-9]+\.[0-9]+\.[0-9]+$/) fail("invalid min_version")
  }
  section == "metadata" && /^[ \t]*mise_sha256_(linux_x64|linux_arm64|macos_x64|macos_arm64|windows_x64)[ \t]*=/ {
    key = $0
    sub(/^[ \t]*/, "", key)
    sub(/[ \t]*=.*/, "", key)
    value = read_value($0, key)
    if (length(value) != 64 || value ~ /[^0-9a-f]/) fail("invalid " key)
    if (key == checksum_key) checksum = value
  }
  END {
    if (failed) exit 1
    if (counts["min_version"] != 1) fail("missing min_version")
    if (counts[checksum_key] != 1) fail("missing " checksum_key)
    printf "MISE_VERSION=%s\nMISE_SHA256=%s\n", version, checksum
  }
' "$repo_root/mise.toml"
