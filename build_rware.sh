#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
puffer="$here/pufferlib-5"
raylib="$puffer/raylib-5.5_linux_amd64"
if [[ ! -f "$raylib/lib/libraylib.a" ]]; then bash "$here/build_env.sh"; fi
[[ $(git -C "$puffer" rev-parse HEAD) == 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2 ]]
gcc -w -O2 -fPIC -shared -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$puffer" -I"$puffer/src" -I"$puffer/vendor" -I"$raylib/include" \
  "$here/rware_bridge.c" -o "$here/rware_bridge.so" \
  "$raylib/lib/libraylib.a" \
  /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libX11.so.6 \
  -lm -lpthread -ldl
echo "Built $here/rware_bridge.so"
