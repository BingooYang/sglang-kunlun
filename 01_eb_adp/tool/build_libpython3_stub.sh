#!/usr/bin/env bash
# 生成 libpython3 占位库到 $BINGOO/env/klx_stub/。
#
# 为什么需要：本机基础 CPython 3.10 是静态编译的（sysconfig 里 Py_ENABLE_SHARED=0，
# /usr/local/lib 只有 libpython3.10.a），没有共享库。而
#   - torch_plugin.initialize_runtime() 会 ctypes.CDLL("libpython3.so") 探活
#   - torch_xmlir 的 7 个扩展 .so 直接 DT_NEEDED libpython3.10.so.1.0
# 两者都失败时，xpytorch 导入钩子只打一条 WARNING 就跳过，XPU 运行时静默不初始化，
# torch.cuda.device_count() 恒为 0。
#
# 解释器是 -rdynamic 链接的、已把 ~900 个 Py* 符号导出到全局作用域，且
# torch_plugin/initialize_runtime.so 的未定义 Py* 符号数为 0 —— 所以空实现占位库足够，
# 真正的符号由主程序解析。详见 libpython3.c。
#
# 不要试图链出真的 libpython3.10.so：libpython3.10.a 不是 PIC 编译的，
# gcc -shared 会报 "relocation R_X86_64_PC32 ... recompile with -fPIC"。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 工作区根：从脚本位置向上找含 env/bin/python 的目录，不依赖当前目录。
BINGOO=${BINGOO:-$HERE}
while [ "$BINGOO" != / ] && [ ! -x "$BINGOO/env/bin/python" ]; do
  BINGOO=$(dirname "$BINGOO")
done
OUT="$BINGOO/env/klx_stub"

mkdir -p "$OUT"
gcc -shared -fPIC -o "$OUT/libpython3.so" "$HERE/libpython3.c"
gcc -shared -fPIC -o "$OUT/libpython3.10.so.1.0" "$HERE/libpython3.c"
ln -sf libpython3.10.so.1.0 "$OUT/libpython3.10.so"

echo "生成完毕："
ls -la "$OUT"
