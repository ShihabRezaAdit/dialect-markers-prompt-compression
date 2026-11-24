#!/usr/bin/env bash
# Create the five environments (internal plan Section 27), install the pilot package into each,
# install the spaCy models, and lock the exact installed versions into envs/lock/
# (the lock files are part of the P0 pre-commitment: commit them).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source cluster.env
mkdir -p envs/lock

# Install en_core_web_sm in the version spaCy itself picks as compatible (as `spacy download`
# does), from the plain wheel URL: current pip rejects the "#egg=name==version" URL that
# `spacy download` passes for older spaCy versions.
spacy_model() {
  local v
  v=$(python -c "from spacy.cli.download import get_compatibility, get_version; print(get_version('en_core_web_sm', get_compatibility()))")
  echo "spaCy model en_core_web_sm $v"
  pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-$v/en_core_web_sm-$v-py3-none-any.whl"
}

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
    if [ "$short" = render ] || [ "$short" = sc ]; then spacy_model; fi
    pip freeze --exclude-editable > "envs/lock/$name.lock.txt"
    deactivate )
}

envs=("$@"); [ ${#envs[@]} -gt 0 ] || envs=(stats gen compress sc render)
for e in "${envs[@]}"; do make_env "$e"; done
echo "Environments ready under $ENV_ROOT. Locks written to envs/lock/ (commit them)."
