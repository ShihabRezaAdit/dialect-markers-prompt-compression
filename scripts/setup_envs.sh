#!/usr/bin/env bash
# Create the five environments (internal plan Section 27), install the pilot package into each,
# install the spaCy models, and lock the exact installed versions into envs/lock/
# (the lock files are part of the P0 pre-commitment: commit them).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source cluster.env
mkdir -p envs/lock

make_env() {  # make_env <short name: render|compress|sc|gen|stats>
  local short="$1" name="env-$1"
  echo "=== $name ==="
  local mv="MODULES_${short}"; local mods="${!mv:-${MODULES:-}}"
  ( [ -z "$mods" ] || module load $mods
    "$PYTHON_BIN" -m venv "$ENV_ROOT/$name"
    source "$ENV_ROOT/$name/bin/activate"
    pip install --upgrade pip wheel setuptools
    pip install -r "envs/$name.txt"
    pip install -e "$HERE"
    if [ "$short" = render ] || [ "$short" = sc ]; then python -m spacy download en_core_web_sm; fi
    pip freeze --exclude-editable > "envs/lock/$name.lock.txt"
    deactivate )
}

for e in ${1:-stats gen compress sc render}; do make_env "$e"; done
echo "Environments ready under $ENV_ROOT. Locks written to envs/lock/ (commit them)."
