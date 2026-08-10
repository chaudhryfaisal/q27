#!/usr/bin/env bash
# Watch the Hugging Face Hub for a checkpoint that has not shipped yet, and the
# moment it appears, run the porting checklist against it.
#
# Two probes, because a release can arrive under a name we did not predict:
#   1. the exact repo (config.json reachable = published)
#   2. every repo in the org whose id matches a pattern, newest first
#
# The Hub answers 401 (not 404) for a repo that does not exist, so an
# unreleased model is indistinguishable from an auth failure by status code
# alone. Treat 200 as the only positive.
#
# Fires once. A state file stops it re-reporting every poll; delete the state
# file to re-arm.
#
# Usage: tools/watch_checkpoint.sh [EXACT_REPO] [ORG] [PATTERN]
set -u
REPO="${1:-Qwen/Qwen3.8-27B}"
ORG="${2:-Qwen}"
PATTERN="${3:-Qwen3\\.8}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/q27-watch}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/$(echo "$REPO" | tr '/' '_').fired"
LOG="$STATE_DIR/watch.log"

stamp() { date -Is; }
log()   { printf '%s %s\n' "$(stamp)" "$*" >> "$LOG"; }

[ -f "$STATE" ] && { log "already fired for $REPO, nothing to do"; exit 0; }

code="$(curl -s -o /dev/null -w '%{http_code}' -m 30 -L \
        "https://huggingface.co/$REPO/resolve/main/config.json" 2>/dev/null)"

# Secondary probe: the org listing, which surfaces a release under any name.
matches="$(curl -s -m 30 "https://huggingface.co/api/models?author=$ORG&sort=createdAt&direction=-1&limit=50" 2>/dev/null \
           | python3 -c "
import sys, json, re
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
pat = re.compile(r'''$PATTERN''')
for m in data:
    if pat.search(m.get('id', '')):
        print(m['id'])
" 2>/dev/null)"

if [ "$code" != "200" ] && [ -z "$matches" ]; then
    log "not yet: $REPO http=$code, no org matches"
    exit 0
fi

# Something landed. Record everything before doing anything that can fail.
REPORT="$STATE_DIR/RELEASED-$(date +%Y%m%d-%H%M%S).md"
{
    echo "# $REPO detected $(stamp)"
    echo
    echo "config.json HTTP status: $code"
    echo "org repos matching /$PATTERN/:"
    if [ -n "$matches" ]; then printf '  %s\n' $matches; else echo "  (none)"; fi
    echo
    echo '## Porting checklist'
    echo '```'
    if [ "$code" = "200" ]; then
        python3 "$REPO_DIR/tools/check_checkpoint.py" "$REPO" 2>&1
        echo "check_checkpoint exit: $?"
    else
        echo "config.json not reachable at $REPO; a matching repo appeared under"
        echo "another name. Run tools/check_checkpoint.py against it by hand:"
        printf '  tools/check_checkpoint.py %s\n' $matches
    fi
    echo '```'
} > "$REPORT" 2>&1

touch "$STATE"
log "FIRED: $REPO http=$code matches='$matches' report=$REPORT"

verdict="$(grep -m1 '^VERDICT' "$REPORT" 2>/dev/null || echo 'see report')"
command -v notify-send >/dev/null 2>&1 && \
    notify-send -u critical "$REPO is out" "$verdict"$'\n'"$REPORT" 2>/dev/null

# Optional email. Credentials are READ from an existing 0600 env file rather
# than copied into a second one: duplicating a secret doubles the number of
# places it can leak from and the number that must be rotated.
#   MAIL_ENV        override the file (default ~/.env)
#   SMTP_USER/PASS  taken from FASTMAIL_USER/SMTP_PASSWORD when not set directly
# Nothing here is ever logged or echoed. Absent or unreadable = skip quietly;
# the report and desktop notification still happen.
MAIL_ENV="${MAIL_ENV:-$HOME/.env}"
if [ -r "$MAIL_ENV" ]; then
    set -a; . "$MAIL_ENV"; set +a
    SMTP_USER="${SMTP_USER:-${FASTMAIL_USER:-}}"
    SMTP_PASS="${SMTP_PASS:-${SMTP_PASSWORD:-}}"
    SMTP_URL="${SMTP_URL:-smtps://smtp.fastmail.com:465}"
    MAIL_TO="${MAIL_TO:-$SMTP_USER}"
    MAIL_FROM="${MAIL_FROM:-$SMTP_USER}"
    if [ -n "${MAIL_TO:-}" ] && [ -n "${SMTP_USER:-}" ] && [ -n "${SMTP_PASS:-}" ]; then
        msg="$STATE_DIR/mail.txt"
        {
            echo "From: ${MAIL_FROM:-$SMTP_USER}"
            echo "To: $MAIL_TO"
            echo "Subject: $REPO is out -- $verdict"
            echo "Date: $(date -R)"
            echo
            echo "$REPO appeared on the Hub at $(stamp)."
            echo
            echo "$verdict"
            echo
            echo "Porting checklist ran automatically. Full report on haight:"
            echo "  $REPORT"
            echo
            sed -n '/## Porting checklist/,$p' "$REPORT" | head -60
        } > "$msg"
        # Config on stdin, not argv: --user on the command line would expose the
        # password in `ps` to anything else running as this user. --silent keeps
        # a failure from spilling anything into the log; only the code is kept.
        if curl --silent --show-error --ssl-reqd -K - \
             >/dev/null 2>>"$STATE_DIR/mail.err" <<CURLCFG
url = "$SMTP_URL"
user = "$SMTP_USER:$SMTP_PASS"
mail-from = "$MAIL_FROM"
mail-rcpt = "$MAIL_TO"
upload-file = "$msg"
CURLCFG
        then
            log "email sent to $MAIL_TO"
        else
            log "email FAILED (see $STATE_DIR/mail.err); report still at $REPORT"
        fi
        rm -f "$msg"
    else
        log "mail.env present but incomplete; skipping email"
    fi
else
    log "no mail.env; skipping email"
fi

echo "$REPO detected. $verdict. Report: $REPORT"
