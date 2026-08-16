#!/usr/bin/env bash
# Put /home/user/BioMedical_QA on an exact commit with a clean tree, so the n=100 manifest records
# a reproducible `git_sha` instead of `<sha>-dirty`. `harness.git_sha()` shells out to
# `git status --porcelain`, which lists untracked files too, so stray artifacts count as dirty.
#
# Nothing is deleted. Tracked modifications go to a stash, untracked files go to an attic beside
# the repo — the earlier baseline run's outputs are in there and are not reproducible.
set -euo pipefail
COMMIT="${1:?usage: reset_to_commit.sh <sha>}"
REPO=/home/user/BioMedical_QA
ATTIC="/home/user/attic/$(date -u +%Y%m%dT%H%M%SZ)"

cd "$REPO"
mkdir -p "$ATTIC"

# Untracked, non-ignored files -> attic, keeping their paths.
mapfile -t untracked < <(git ls-files --others --exclude-standard)
for f in "${untracked[@]}"; do
  mkdir -p "$ATTIC/$(dirname "$f")"
  mv -f "$f" "$ATTIC/$f"
done
echo "attic: ${#untracked[@]} untracked files -> $ATTIC"

# Tracked modifications -> stash (recoverable with `git stash list` / `git stash show -p`).
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git stash push -m "pre-$COMMIT remote tree" >/dev/null
  echo "stashed tracked modifications: $(git stash list | head -1)"
else
  echo "no tracked modifications to stash"
fi

git fetch origin --quiet
git checkout --quiet "$COMMIT"
git reset --hard --quiet "$COMMIT"

echo "HEAD: $(git rev-parse HEAD)"
status="$(git status --porcelain)"
if [ -n "$status" ]; then
  echo "TREE NOT CLEAN — the manifest would be marked -dirty:"
  printf '%s\n' "$status"
  exit 1
fi
echo "tree is clean — git_sha will be recorded without the -dirty suffix"
