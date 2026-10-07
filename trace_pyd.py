"""Trace DLLs in mediapipe.tasks.python.core."""
import os, ctypes, glob as g

import mediapipe.tasks.python.core as core_mod
d = os.path.dirname(core_mod.__file__)
native_dir = os.path.join(d, "mediapipe_c_bindings")
print("Bindings dir:", native_dir)
print("Contents:", os.listdir(native_dir) if os.path.isdir(native_dir) else "NOT DIR")

dlls = g.glob(os.path.join(native_dir, "**/*.pyd"), recursive=True) + g.glob(os.path.join(d, "**/*.pyd"), recursive=True)
print("DLLs:", dlls[:5])
for dll in dlls[:3]:
    try:
        lib = ctypes.CDLL(dll)
        print("  OK:", os.path.basename(dll), "free:", hasattr(lib, "free"))
    except OSError as e:
        print("  FAIL:", os.path.basename(dll), str(e)[:80])
