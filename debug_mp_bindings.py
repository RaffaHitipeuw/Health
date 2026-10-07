"""Debug mediapipe C binding load failure."""
import sys
import os
import ctypes

# Try to find what library mediapipe is trying to load
import mediapipe.tasks.python.core.mediapipe_c_bindings as bindings
print("Module location:", bindings.__file__)
import inspect
src = inspect.getsource(bindings.load_raw_library)
print("load_raw_library source:")
print(src[:2000])
