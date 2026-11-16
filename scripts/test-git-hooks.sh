#!/bin/sh
# Prove the attribution hook without touching the real repository.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

identity="Agent <1+agent@users.noreply.github.com>"

git -C "$tmp" init -q
git -C "$tmp" config user.name "Hook Test"
git -C "$tmp" config user.email "hook-test@example.com"
git -C "$tmp" config core.hooksPath "$root/.githooks"
git -C "$tmp" config cuepilot.coauthor "$identity"

commit() {
    echo "$1" >>"$tmp/file"
    git -C "$tmp" add file
    shift
    git -C "$tmp" commit -q "$@"
}

message() {
    git -C "$tmp" log -1 --format=%B
}

trailers() {
    message | git -C "$tmp" interpret-trailers --parse
}

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

# -F keeps the trailer literal: BSD grep reads "\+" as "one or more", which
# would turn the noreply "+" tag into a false negative.
count_coauthor() {
    message | grep -cF 'Co-authored-by:' || true
}

# The marker turns into exactly one trailer, in the marker's own block.
commit one -m "feat: assisted change" -m "Assisted-by: OpenCode"
message | grep -qxF "Co-authored-by: $identity" || fail "trailer was not appended"
[ "$(count_coauthor)" -eq 1 ] || fail "trailer was duplicated"
trailers | grep -qxF 'Assisted-by: OpenCode' || fail "marker left the trailer block"
trailers | grep -qxF "Co-authored-by: $identity" ||
    fail "co-author is not in the final trailer block"

# Reusing the message (amend, squash, cherry-pick -x) does not duplicate it.
git -C "$tmp" commit -q --amend --no-edit
[ "$(count_coauthor)" -eq 1 ] || fail "amend duplicated the trailer"

# No marker, no trailer.
commit two -m "chore: unassisted change"
message | grep -qF 'Co-authored-by:' && fail "trailer added without the marker"

# A human co-author already on the commit is left alone.
commit three -m "feat: pair change" -m "Assisted-by: OpenCode" \
    -m "Co-authored-by: Human <human@example.com>"
[ "$(count_coauthor)" -eq 1 ] || fail "human co-author was not preserved"
message | grep -qxF 'Co-authored-by: Human <human@example.com>' ||
    fail "human co-author was replaced"

# The marker without a configured identity is a no-op, not a failure.
git -C "$tmp" config --unset cuepilot.coauthor
commit four -m "feat: marker but no identity" -m "Assisted-by: OpenCode"
message | grep -qF 'Co-authored-by:' && fail "trailer added without an identity"

echo "git hook checks passed"
