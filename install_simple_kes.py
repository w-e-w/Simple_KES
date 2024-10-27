import os
import subprocess
import sys
import venv

def install_requirements(venv_path):
    """Install requirements into the specified virtual environment."""
    # Determine the path to pip in the virtual environment
    pip_executable = os.path.join(venv_path, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_path, 'Scripts', 'pip')

    # Check if the requirements file exists
    requirements_file = 'simple_kes_requirements.txt'
    if not os.path.isfile(requirements_file):
        print(f"Requirements file '{requirements_file}' not found in root.")
        sys.exit(1)

    # Run pip install
    try:
        print(f"Installing requirements from {requirements_file} into virtual environment at {venv_path}...")
        subprocess.check_call([pip_executable, 'install', '-r', requirements_file])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def create_virtualenv(venv_path):
    """Create a new virtual environment at the specified path."""
    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    print(f"Virtual environment created at {venv_path}")

def main():
    # Define paths
    root_path = os.path.abspath(os.getcwd())
    venv_path = os.path.join(root_path, 'venv')
    fallback_venv_path = os.path.join(root_path, 'simple_kes_requirements')

    # Check if the 'venv' folder exists
    if os.path.isdir(venv_path):
        print(f"'venv' folder found at {venv_path}")
        install_requirements(venv_path)
    else:
        print(f"'venv' folder not found. Using fallback path '{fallback_venv_path}'.")
        if not os.path.isdir(fallback_venv_path):
            create_virtualenv(fallback_venv_path)
        install_requirements(fallback_venv_path)

if __name__ == "__main__":
    main()
