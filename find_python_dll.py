"""Find free() in Python DLL."""
import ctypes, os, sysconfig

# Find python DLL
config = sysconfig.get_config()
platlib = sysconfig.get_path('platlib')
# Try python313.dll
python_dll = os.path.join(os.path.dirname(sys.executable.replace('/', os.sep), 'python313.dll')
print("Python DLL:", python_dll)
if os.path.exists(python_dll):
    try:
        dll = ctypes.CDLL(python_dll)
        print("Loaded. Has free:", hasattr(dll, 'free'))
    except Exception as e:
        print("Load failed:", e)

# Try in Python DLLs dir
pydll_dir = os.path.dirname(sys.executable.replace('/', os.sep))
print("DLL dir:", pydll_dir)
for f in os.listdir(pydll_dir):
    if f.startswith('python') and f.endswith('.dll'):
        print("  DLL:", f)
        try:
            dll = ctypes.CDLL(os.path.join(pydll_dir, f))
            print("    Loaded. free:", hasattr(dll, 'free'))
        except Exception as e:
            print("    Failed:", e)
