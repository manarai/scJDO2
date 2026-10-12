#!/usr/bin/env bash
# Install v1 scJDO at the preprint tag into the current (active) venv.
#
# We go via a dedicated git worktree so that the pip install does not force a
# checkout of the main scJDO repo. This keeps the v1 repo frozen at HEAD for
# day-to-day work while v1 is pinned to its submission tag here.
#
# Usage:
#     source .venv/bin/activate
#     bash scripts/install_v1_compare.sh
#
# Override the tag or source by setting SCJDO_V1_TAG / SCJDO_V1_REPO.

set -euo pipefail

SCJDO_V1_REPO="${SCJDO_V1_REPO:-/Users/terooatt/Documents/scJDO}"
SCJDO_V1_TAG="${SCJDO_V1_TAG:-scjdo-v56-submission-20260922-r25}"
WORKTREE_DIR="${SCJDO_V1_WORKTREE:-${TMPDIR:-/tmp}/scJDO-v1-${SCJDO_V1_TAG}}"

if [[ ! -d "$SCJDO_V1_REPO/.git" ]]; then
    echo "scJDO v1 repo not found at $SCJDO_V1_REPO" >&2
    exit 1
fi

if [[ ! -d "$WORKTREE_DIR" ]]; then
    echo "Creating v1 worktree at $WORKTREE_DIR (tag $SCJDO_V1_TAG)"
    git -C "$SCJDO_V1_REPO" worktree add --detach "$WORKTREE_DIR" "$SCJDO_V1_TAG"
else
    echo "Reusing existing v1 worktree at $WORKTREE_DIR"
fi

echo "Installing v1 scJDO from $WORKTREE_DIR into current venv..."
pip install -e "$WORKTREE_DIR"
python -c "import scjdo; print('v1 scjdo', scjdo.__version__, 'OK')"
