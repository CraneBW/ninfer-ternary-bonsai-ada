#!/usr/bin/env bash
# 旋转内核 oracle：编译 `harness/rot_test.cu`（它 include 引擎真正在用的
# `ops/linear/ternary/ternary_rotation_kernels.cuh`），跑六个用例把输入/符号/输出
# dump 成裸 f32，再让 `oracle_rot.py` 逐元素比对文档化变换 y = H*(s*(P*x)) 与 h = s*(H*z)，
# 同时要求**每个"看似合理但错"的变体都必须失败**（双侧判据）。
#
# 为什么值得跑：六个用例里 `plain_t3` / `perm_t2` / `inv_t2` 是 T>1 的。
# T=1 时 token-major 与 row-major 布局重合 ⇒ 只测 T=1 的话，真实 prefill 全错也能全绿。
#
# 用法：bash tools/verify/run-rot-oracle.sh [dump-dir]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DUMP="${1:-/tmp/rot-oracle-dump}"
ARCH="${ARCH:-sm_89}"

mkdir -p "$DUMP"

echo "=== 编译 harness（arch=$ARCH）==="
nvcc -O2 -std=c++20 -arch="$ARCH" -I "$REPO/src" \
     -o "$DUMP/rot_test" "$HERE/harness/rot_test.cu" \
     -Xcompiler -Wno-unused-result 2>&1 | tail -20
echo "  产物 $DUMP/rot_test"

echo
echo "=== 跑六个用例 ==="
"$DUMP/rot_test" "$DUMP/"

echo
echo "=== numpy oracle ==="
python3 "$HERE/oracle_rot.py" "$DUMP/"
rc=$?

echo
echo "=== 结论 ==="
if [ "$rc" -eq 0 ]; then
    echo "  ✅ 旋转内核与文档化变换一致，且所有错变体都被分离（rel_l2 双侧判据通过）"
else
    echo "  ❌ oracle 返回 $rc —— 看上面的逐用例输出，别跳过"
fi
echo "  dump 与二进制留在 $DUMP/"
exit "$rc"
