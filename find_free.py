"""Find which DLL/PYD has free()."""
import os, ctypes

base = r"C:\miniconda3\Lib\site-packages\mediapipe"
found = []
fail = []
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.pyd'):
            path = os.path.join(root, f)
            try:
                lib = ctypes.CDLL(path)
                has_free = hasattr(lib, 'free')
                if has_free:
                    found.append(path)
                else:
                    fail.append(path)
            except OSError as e:
                fail.append(f"LOAD FAILED: {f}: {e}")
                pass

print("PYDs WITH free():", [os.path.basename(x) for x in found])
print("PYDs WITHOUT free():", [os.path.basename(x) for x in fail[:5]])
