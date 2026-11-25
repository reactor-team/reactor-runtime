#!/usr/bin/env bash
# Mint a short-lived session token (JWT) that can start sessions on ONE model.
#
#   export REACTOR_JWT="$(scripts/mint_token.sh <org>/<name>[:<environment>] [options])"
#
# <org>/<name> is the production environment; <org>/<name>:staging names another environment.
#
# Options:
#   --expires SECONDS        token lifetime (default 1800; the API caps it at 21600)
#   --max-sessions N         sessions the token may create over its whole life (default 2)
#   --max-duration SECONDS   longest a session it creates may run (default 1800)
#
# Reads the API key from $REACTOR_API_KEY and the endpoint from $REACTOR_SERVER
# (default https://api.reactor.inc). The key is sent as a header read from a process
# substitution, so it never appears in argv or in a file. stdout carries the JWT only;
# the grant (models, limits, expiry) goes to stderr.
# Needs curl and jq.
set -euo pipefail

usage() { sed -n '2,17p' "$0" >&2; exit 2; }
[[ $# -ge 1 ]] || usage
MODEL="$1"; shift
EXPIRES=1800; MAX_SESSIONS=2; MAX_DURATION=1800
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expires) EXPIRES="$2"; shift 2 ;;
    --max-sessions) MAX_SESSIONS="$2"; shift 2 ;;
    --max-duration) MAX_DURATION="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$MODEL" == */* ]] || { echo "model must be <org>/<name>[:<environment>], got '$MODEL'" >&2; exit 2; }
[[ -n "${REACTOR_API_KEY:-}" ]] || { echo "REACTOR_API_KEY is not set" >&2; exit 2; }
SERVER="${REACTOR_SERVER:-https://api.reactor.inc}"

BODY="$(jq -n --arg m "$MODEL" --argjson e "$EXPIRES" --argjson s "$MAX_SESSIONS" --argjson d "$MAX_DURATION" '
  {expires_after: $e,
   authorization_details: [{type: "session",
     resources: {models: {match: [$m]}},
     constraints: {max_sessions: $s, max_session_duration_seconds: $d}}]}')"

RESP="$(curl -sS -X POST "$SERVER/tokens" \
  -H @<(printf 'Authorization: Bearer %s\n' "$REACTOR_API_KEY") \
  -H 'Content-Type: application/json' -d "$BODY")"

JWT="$(jq -r '.jwt // empty' <<<"$RESP" 2>/dev/null || true)"
if [[ -z "$JWT" ]]; then
  echo "token mint failed: $RESP" >&2
  exit 1
fi
jq 'del(.jwt)' <<<"$RESP" >&2
printf '%s\n' "$JWT"
