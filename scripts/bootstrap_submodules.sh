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

# --- MIN: light, pure-Python/Java, needed in early phases ---------------------
add https://github.com/OrtegaSA/CQC-QHE-repo.git       submodules/CQC-QHE       min  # QHE + Clifford+T QCNN (WS4), Apache-2.0
add https://github.com/IBM/AMLSim.git                  submodules/AMLSim        min  # synthetic AML graphs (finance), Apache-2.0
add https://github.com/kanav99/shark.git               submodules/shark         min  # FSS actively-secure 2PC (reference), MIT
add https://github.com/lschoe/mpyc.git                 submodules/mpyc          min  # non-linear MPC: Shamir ReLU/argmax (the hybrid's non-linear half), MIT

# --- ALL: heavy C++/CUDA builds + large data generators ----------------------
add https://github.com/openfheorg/openfhe-development.git submodules/openfhe    all  # classical FHE (CKKS/BGV/BFV), BSD-2
add https://github.com/microsoft/SEAL.git              submodules/SEAL          all  # classical FHE; node-seal WASM, MIT
add https://github.com/open-quantum-safe/liboqs.git    submodules/liboqs        all  # PQC C lib (Python via pip), MIT
add https://github.com/data61/MP-SPDZ.git              submodules/MP-SPDZ       all  # classical CPU MPC baseline
add https://github.com/Alisah-Ozcan/HEonGPU.git        submodules/HEonGPU       all  # GPU BFV/CKKS (linear layers), Apache-2.0
add https://github.com/chart21/hpmpc.git               submodules/hpmpc         all  # GPU secret-sharing MPC (PIGEON), MIT
add https://github.com/synthetichealth/synthea.git     submodules/synthea       all  # synthetic EHR (medical), Apache-2.0
add https://github.com/open-quantum-safe/oqs-provider.git submodules/oqs-provider min  # OpenSSL 3 PQC provider (crypto-agility / hybrid TLS), MIT
# CROSS (HE on AI-ASIC) — confirm public URL + license before enabling:
# add https://github.com/EfficientHE/CROSS.git         submodules/CROSS         all

echo
echo "Done. Initialized submodules:"
git submodule status 2>/dev/null || true
echo
echo "Tip: build C++/CUDA submodules with their own CMake (see README)."
