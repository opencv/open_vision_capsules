import importlib
import sys
import os
import subprocess
from pkg_resources import get_distribution, DistributionNotFound

def get_openvino_version(openvino_path):
    # Method 1: Try to use openvino.runtime
    try:
        from openvino.runtime import get_version
        return get_version()
    except ImportError:
        pass

    # Method 2: Try to use openvino.inference_engine
    try:
        from openvino.inference_engine import get_version
        return get_version()
    except ImportError:
        pass

    # Method 3: Check for version.txt in various locations
    version_files = [
        os.path.join(openvino_path, 'deployment_tools', 'model_optimizer', 'version.txt'),
        os.path.join(openvino_path, 'version.txt'),
        os.path.join(openvino_path, '..', 'version.txt'),
    ]
    for version_file in version_files:
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                return f.read().strip()

    # Method 4: Try to run the command-line version check
    try:
        result = subprocess.run(['mo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    return "Version not found"

def print_module_info(module_name):
    try:
        # Try to get distribution info
        distribution = get_distribution(module_name)
        ver = distribution.version
        location = distribution.location
    except DistributionNotFound:
        # If distribution is not found, try to import the module directly
        try:
            module = importlib.import_module(module_name)
            if module_name == 'openvino':
                # Special handling for OpenVINO
                openvino_path = os.path.dirname(os.path.dirname(os.path.dirname(module.__file__)))
                ver = get_openvino_version(openvino_path)
            else:
                ver = getattr(module, '__version__', 'Unknown')
            location = getattr(module, '__file__', 'Unknown location')
        except ImportError:
            ver = 'Not found'
            location = 'ImportError'
        except AttributeError:
            ver = 'Version not found'
            location = getattr(module, '__file__', 'Unknown location')
    except Exception as e:
        ver = 'Error'
        location = str(e)
    
    print(f'{module_name}:')
    print(f'  Version: {ver}')
    print(f'  Location: {location}')
    
    # If the module is already imported, print its actual location in sys.modules
    if module_name in sys.modules:
        print(f'  Imported from: {sys.modules[module_name].__file__}')
    print()
