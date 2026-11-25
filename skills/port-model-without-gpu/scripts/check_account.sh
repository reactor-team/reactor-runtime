#!/usr/bin/env bash
# Phase 0: is this account ready to deploy a GPU model with the project + environment CLI?
#
#   scripts/check_account.sh [GPU_TYPE] [GPU_COUNT]
#   e.g. scripts/check_account.sh NVIDIA_B200 1
#
# Prints one PASS/FAIL line per item and exits non-zero if any item fails:
#   key         the API key is valid on this endpoint (GET /me): user, account id, roles
#   cli         `reactor whoami` answers on the same endpoint (skipped if the CLI is not installed)
#   slug        the account has an org slug, so projects can be named <org>/<name>
#   allocation  the account holds GPUs (GET /accounts/{id}/accelerators); with GPU_TYPE, a region
#               has at least GPU_COUNT of that type available. "held" counts GPUs your running
#               environments hold, the one you are about to replace included; the models holding
#               them are listed under each row.
#
# Reads $REACTOR_API_KEY and $REACTOR_SERVER (default https://api.reactor.inc). A `reactor auth login`
# keeps the key in the CLI's store, not in the environment: export it for this script. The key is
# sent as a header from a process substitution, never in argv. Needs curl and jq.
set -uo pipefail

GPU_TYPE="${1:-}"; GPU_COUNT="${2:-1}"
SERVER="${REACTOR_SERVER:-https://api.reactor.inc}"
[[ -n "${REACTOR_API_KEY:-}" ]] || { echo "FAIL key         REACTOR_API_KEY is not set"; exit 1; }
FAILS=0
pass() { printf 'PASS %-11s %s\n' "$1" "$2"; }
fail() { printf 'FAIL %-11s %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

api() {  # api <path> -> sets BODY and CODE (called directly, not in $(...), so both survive)
  local out
  if ! out="$(curl -sS -w '\n%{http_code}' "$SERVER$1" \
      -H @<(printf 'Authorization: Bearer %s\n' "$REACTOR_API_KEY") 2>&1)"; then
    CODE=000; BODY="$out"; return
  fi
  CODE="${out##*$'\n'}"
  BODY="${out%$'\n'*}"
}

echo "endpoint: $SERVER"

# key
api /me; ME="$BODY"
if [[ "$CODE" == 200 ]]; then
  ACCOUNT_ID="$(jq -r '.account_id' <<<"$ME")"
  pass key "user $(jq -r '.email' <<<"$ME"), account $ACCOUNT_ID, roles $(jq -r '.roles | join(",")' <<<"$ME")"
else
  fail key "GET /me returned HTTP $CODE: $ME"
  echo "cannot continue without a valid key"; exit 1
fi

# cli, and the slug as the CLI reports it ("Account Slug")
SLUG=""
if command -v reactor >/dev/null 2>&1; then
  if W="$(REACTOR_SERVER="$SERVER" reactor whoami --json 2>&1)"; then
    pass cli "account $(jq -r '.account_id // "?"' <<<"$W" 2>/dev/null), server $(jq -r '.server // "?"' <<<"$W" 2>/dev/null)"
    SLUG="$(jq -r '.account_slug // empty' <<<"$W" 2>/dev/null)"
    [[ "$SLUG" == "not set" || "$SLUG" == "unknown" ]] && SLUG=""
  else
    fail cli "reactor whoami failed: $(tr '\n' ' ' <<<"$W")"
  fi
else
  echo "SKIP cli         the reactor CLI is not installed"
fi

# slug: from the CLI, else from GET /accounts (the key's account, as a bare list or under "items")
if [[ -z "$SLUG" ]]; then
  api /accounts; ACCTS="$BODY"
  if [[ "$CODE" == 200 ]]; then
    SLUG="$(jq -r --arg id "$ACCOUNT_ID" '
      (if type == "array" then . else (.items // []) | if type == "array" then . else [.] end end)
      | map(select(.id == $id)) | .[0].slug // empty' <<<"$ACCTS")"
  fi
fi
if [[ -n "$SLUG" ]]; then
  pass slug "'$SLUG': name the project $SLUG/<name> (model.name in reactor.yaml)"
else
  fail slug "the account has no slug; a project create needs one. Finish the account's Serve setup, or ask Reactor support"
fi

# allocation
api "/accounts/$ACCOUNT_ID/accelerators"; ACC="$BODY"
if [[ "$CODE" != 200 ]]; then
  fail allocation "GET /accounts/$ACCOUNT_ID/accelerators returned HTTP $CODE: $ACC"
else
  N="$(jq '.accelerators | length' <<<"$ACC")"
  if [[ "$N" == 0 ]]; then
    fail allocation "the account holds no GPUs; ask for an allocation before deploying"
  else
    echo "     region        gpu_type              granted held available"
    jq -r '.accelerators[]
      | "     \(.region|.+" "*(14-length))\(.gpu_type|.+" "*(22-length))\(.gpu_count)       \(.held)    \(.available)"
        + (if .inventory_status == "probed" and .hardware_present == false then "   (no such hardware in this region)" else "" end)
        + ([.models[]? | "\n         held by \(.model): \(.gpu_count) GPU(s) x \(.instances) instance(s)"] | join(""))' <<<"$ACC"
    if [[ -n "$GPU_TYPE" ]]; then
      REGIONS="$(jq -r --arg t "$GPU_TYPE" --argjson c "$GPU_COUNT" \
        '[.accelerators[] | select(.gpu_type == $t and .available >= $c) | .region] | join(",")' <<<"$ACC")"
      if [[ -n "$REGIONS" ]]; then
        pass allocation "$GPU_COUNT x $GPU_TYPE available in: $REGIONS (deploy with --gpu ${GPU_TYPE}x$GPU_COUNT --require-regions <one of them>)"
      else
        fail allocation "no region has $GPU_COUNT x $GPU_TYPE available; pick a type and region from the table above (GPUs held by an environment you will replace count as held)"
      fi
    else
      pass allocation "$N allocation row(s); pass GPU_TYPE [GPU_COUNT] to check one"
    fi
  fi
fi

exit $(( FAILS > 0 ))
