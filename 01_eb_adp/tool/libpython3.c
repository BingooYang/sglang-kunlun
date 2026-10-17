/*
 * libpython3.so stub.
 *
 * torch_plugin.initialize_runtime() calls ctypes.CDLL("libpython3.so") purely
 * as a preload/probe (its return value is discarded).  On this machine the
 * base CPython 3.10 is a STATIC build (Py_ENABLE_SHARED=0 in sysconfig, and
 * /usr/local/lib only ships libpython3.10.a), so no libpython3.so exists and
 * the probe fails with "cannot open shared object file".
 *
 * That failure aborts the whole xpytorch import hook (torch_plugin never runs,
 * the XPU runtime is never initialised and no XPU device is visible).
 *
 * Nothing actually needs symbols from this library: the interpreter is linked
 * with -rdynamic and already exports ~900 Py* symbols from the main executable,
 * and torch_plugin/initialize_runtime.so has ZERO undefined Py* symbols.
 * So an empty shared object is a faithful stand-in for "there is no shared
 * libpython to preload".
 */
void _xpytorch_libpython3_stub(void) {}
