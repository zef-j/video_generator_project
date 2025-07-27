#!/usr/bin/env bash
set -euo pipefail

export PYTHONIOENCODING=UTF-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# If we were given exactly 4 args and they look like typical execute_task usage,
# call the Python module directly.
if [ "$#" -ge 4 ] && [ -f "$1" ] && [ -f "$2" ]; then
  audio="$1"; text="$2"; cfg="$3"; out="$4"
  shift 4
  python3 -m aeneas.tools.execute_task "$audio" "$text" "$cfg" "$out" "$@"
else
  # Otherwise, run whatever the user asked (e.g., python3 -m …, bash, etc.)
  exec "$@"
fi
