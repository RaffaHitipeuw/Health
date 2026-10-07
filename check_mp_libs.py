"""Check mediapipe C libraries."""
import os, glob
mp_dir = "C:/minconda3/Lib/site-packages/mediapipe"
# Find DLLs
dlls = glob.glob(f"{mp_dir}/**/*.dll", recursive=True)
print("DLLs found:", dlls[:10] if dlls else "none")
# Check what mediapipe_tasks_c_bindings needs
try:
    import mediapipe.tasks.python.core.mediapipe_c_bindings as bindings
    print("Has mediapipe_c_bindings:", True)
    print("Looking for libmediapipe_c:")
    import ctypes
    # Try common names
    for libname in ["mediapipe_c", "libmediapipe_c", "mediapipe_c.dll"]:
        try:
            lib = ctypes.CDLL(libname)
            print(f"  Loaded {libname}: OK")
        except Exception as e:
            print(f"  {libname}: {e}")
except ImportError as e:
    print("Cannot import mediapipe_c_bindings:", e)
