"""Find free() in Python DLLs."""
import ctypes, os, sysconfig, sys
exe_dir = os.path.dirname(sys.executable.replace('\\', '/'))
python_dll = os.path.join(exe_dir, 'python313.dll')
print("Python DLL:", python_dll)
print("Exists:", os.path.exists(python_dll))
if os.path.exists(python_dll):
    try:
        dll = ctypes.CDLL(python_dll)
        print("Loaded. Has free:", hasattr(dll, 'free'))
    except Exception as e:
        print("Load failed:", type(e).__name__, str(e)[:100])
