#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
puffer="$here/pufferlib-5"
raylib="$puffer/raylib-5.5_linux_amd64"
commit=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2

if [[ ! -d "$puffer/.git" ]]; then
  git init "$puffer"
  git -C "$puffer" remote add origin https://github.com/PufferAI/PufferLib.git
  git -C "$puffer" fetch --depth 1 origin "$commit"
  git -C "$puffer" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$puffer" rev-parse HEAD) != "$commit" ]]; then
  echo "PufferLib checkout has a different commit; expected $commit" >&2
  exit 1
fi
if [[ ! -f "$raylib/lib/libraylib.a" ]]; then
  curl -fsSL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz \
    -o "$here/raylib-5.5_linux_amd64.tar.gz"
  tar -xf "$here/raylib-5.5_linux_amd64.tar.gz" -C "$puffer"
  rm "$here/raylib-5.5_linux_amd64.tar.gz"
fi

gcc -w -O2 -fPIC -shared -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$puffer" -I"$puffer/src" -I"$puffer/vendor" -I"$raylib/include" \
  "$here/overcooked_bridge.c" -o "$here/overcooked_bridge.so" \
  "$raylib/lib/libraylib.a" \
  /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libX11.so.6 \
  -lm -lpthread -ldl
echo "Built $here/overcooked_bridge.so"
