"""Trace the exact mediapipe C library loading failure."""
import sys
import ctypes

# Step 1: Find the library file mediapipe is trying to load
import mediapipe.tasks.python.core.mediapipe_c_bindings as bindings
import inspect
src = inspect.getsource(bindings.load_raw_library)
print("=== load_raw_library source ===")
print(src)
print("===")

# Step 2: Try manual DLL loading to see which one fails
lib_names = [
    "mediapipe_tasks_python_core",
    "libmediapipe_tasks_python_core",
    "mediapipe",
    "libmediapipe",
    "mediapipe_c",
    "libmediapipe_c",
]
for name in lib_names:
    try:
        lib = ctypes.CDLL(name)
        print(f"LOADED {name}: OK (has 'free': {'free' in dir(lib))")
    except FileNotFoundError:
        print(f"FILE NOT FOUND: {name}")
    except Exception as e:
        print(f"ERROR {name}: {type(e).__name__}: {e}")
