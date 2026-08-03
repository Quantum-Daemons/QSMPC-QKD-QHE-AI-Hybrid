#!/usr/bin/env bash
# Add/initialize the upstream OSS git submodules under submodules/.
#
# All repos below are commercially-usable (MIT / Apache-2.0 / BSD) and were verified
# actively maintained (<= ~6 months) as of June 2026.
#
# IMPORTANT (originality/attribution): we do NOT clone or vendor the user's
# `RhombusEnd2End_HEonGPU` repo (mixed authorship; upstream may be RAPID.git).
# The GPU MPC-FHE benchmark is our OWN integration over the upstream OSS pulled here.
#
# Usage:
#   bash scripts/bootstrap_submodules.sh            # MIN set (light, needed early)
#   QSH_SET=all  bash scripts/bootstrap_submodules.sh   # full set (incl. heavy C++/CUDA)
#   QSH_DEPTH=0  bash scripts/bootstrap_submodules.sh   # full-history clones (default: shallow)
set -uo pipefail
cd "$(dirname "$0")/.."

QSH_SET="${QSH_SET:-min}"
QSH_DEPTH="${QSH_DEPTH:-1}"     # 1 = shallow add; 0 = full

add() {  # add <url> <path> <tier:min|all>
  local url="$1" path="$2" tier="$3"
  if [[ "$tier" == "all" && "$QSH_SET" != "all" ]]; then
    echo "  skip (tier=all): $path"; return 0
  fi
  if [[ -e "$path/.git" || -f "$path/.git" ]]; then
    echo "  exists: $path"; return 0
  fi
  echo "==> add $path  <-  $url"
  if [[ "$QSH_DEPTH" == "0" ]]; then
    git submodule add --force "$url" "$path" || echo "  !! failed: $path (continuing)"
  else
    git submodule add --force --depth 1 "$url" "$path" || echo "  !! failed: $path (continuing)"
  fi
}

echo "Submodule set: $QSH_SET (depth=$QSH_DEPTH)"

# Only the submodules that code actually builds against are vendored.
#
# Round 17 removed ten others (CQC-QHE, AMLSim, shark, mpyc, oqs-provider, openfhe, SEAL, liboqs,
# MP-SPDZ, synthea). None was referenced by any import, script or Dockerfile; five were superseded
# by the pip/npm package of the same project (mpyc, openfhe, liboqs via liboqs-python, SEAL via the
# node-seal npm build, MP-SPDZ was only ever a registry string); and .dockerignore excludes
# submodules/ wholesale, so none of them ever reached an image anyway. Cloning ~10 GB of C++ that
# nothing links against is a cost with no corresponding capability. They remain cited as upstream
# work in references/REFERENCES.md, which is the accurate relationship.
add https://github.com/Alisah-Ozcan/HEonGPU.git        submodules/HEonGPU       all  # GPU BFV/CKKS (linear layers), Apache-2.0
add https://github.com/chart21/hpmpc.git               submodules/hpmpc         all  # GPU secret-sharing MPC (PIGEON), MIT
# CROSS (HE on AI-ASIC) — confirm public URL + license before enabling:
# add https://github.com/EfficientHE/CROSS.git         submodules/CROSS         all

echo
echo "Done. Initialized submodules:"
git submodule status 2>/dev/null || true
echo
echo "Tip: build C++/CUDA submodules with their own CMake (see README)."
