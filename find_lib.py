"""Find what mediapipe actually loads as its native library."""
import ctypes, os

# Simulate load_raw_library logic
# mediapipe looks for these names in order:
lib_names = [
    "_mediapipe_tasks_python_core",
    "mediapipe_tasks_python_core",
    "mediapipe",
    "libmediapipe",
]
exe_dir = os.path.dirname(__import__("sys").executable.replace("\\", "/")
print("Python DLL:", os.path.join(exe_dir, "python313.dll"))

for name in lib_names:
    try:
        ctypes.CDLL(name)
        print("Loaded", name)
    except FileNotFoundError:
        print("Not found:", name)
    except OSError as e:
        print("Load error", name + ":", str(e)[:100])
