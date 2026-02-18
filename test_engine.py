import sys
import os

build_path = os.path.join(os.getcwd(), "out", "build", "x64-Release", "cpp_core")

if os.path.exists(build_path):
    sys.path.append(build_path)
    print(f"Added build path: {build_path}")
else:
    print(f"Error: Could not find build folder at {build_path}")
    print("Make sure you have clicked 'Build All' in Visual Studio.")

try:
    import radoptima_core
    print("Successfully imported RadOptima Core!")
    
    radoptima_core.check_cuda()
    
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")