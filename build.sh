#!/usr/bin/env bash
# Bygger Obsidianalys 2: React-frontend (Vite) → inbäddning i C++ → binär (CMake).
set -euo pipefail
cd "$(dirname "$0")"

(cd web && npm ci --no-fund --no-audit && npm run build)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

echo
echo "Klart: $(pwd)/build/obsidianalys"
