"""Trace mediapipe C library loading."""
import ctypes
import os

# Step 1: list files in mediapipe tasks python core
import mediapipe.tasks.python.core as core_mod
core_dir = os.path.dirname(core_mod.__file__)
print("mediapipe.tasks.python.core dir:", core_dir)
for f in os.listdir(core_dir):
    if f.endswith(('.pyd', '.so', '.dll'):
        print("  native lib:", f)

# Step 2: try loading the DLLs directly
dll_dir = os.path.join(core_dir, 'mediapipe_c_bindings')
if os.path.exists(dll_dir):
    for f in os.listdir(dll_dir):
        print("  bindings DLLs:", f)

# Step 3: try to load each DLL and check free()
import glob as glob_mod
all_dlls = glob_mod.glob(os.path.join(core_dir, '**', '*.dll'), recursive=True)
print("All DLLs in mediapipe.tasks.python.core:", all_dlls)
for dll_path in all_dlls:
    print("  Trying:", dll_path)
    try:
        lib = ctypes.CDLL(dll_path)
        print("    Loaded OK - free:", hasattr(lib, 'free'))
    except OSError as e:
        print("    Load failed:", e)
