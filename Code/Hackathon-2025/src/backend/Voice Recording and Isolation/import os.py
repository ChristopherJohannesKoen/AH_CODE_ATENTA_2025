import os
import sys
import DeepFilterNet
from importlib.resources import files

# --- User Instructions ---
# 1. Save this code as a Python file (e.g., check_files.py).
# 2. Run it from your terminal using: python check_files.py
# 3. The output will help us diagnose the problem.
# -------------------------

def check_deepfilternet_files():
    """
    Checks for the presence and location of DeepFilterNet model files
    using a more robust method that handles zipped installations.
    """
    print("--- Checking DeepFilterNet Installation ---")
    try:
        # Get the installation path of the deepfilternet package using a robust method
        # This works even if the package is installed in a .zip file
        pkg_path = files(DeepFilterNet).joinpath('')
        print(f"DeepFilterNet package found at: {pkg_path}")

        # Construct the path to the 'models' directory where the zip files should be
        models_path = pkg_path.joinpath("models")
        print(f"Looking for models in: {models_path}")

        # Check if the models directory exists
        if not models_path.exists():
            print("ERROR: The 'models' directory does not exist.")
            print("This indicates a corrupted or incomplete installation.")
            return

        # List all files in the models directory to find the zip files
        found_zips = False
        for file in models_path.iterdir():
            if file.suffix in ['.zip', '.gz']:
                print(f"Found model file: {file}")
                found_zips = True

        if not found_zips:
            print("WARNING: No compressed model files (.zip, .tar.gz) found.")
            print("This could mean the library has already extracted them,")
            print("or they were never downloaded.")

    except ImportError:
        print("ERROR: 'deepfilternet' module not found.")
        print("Please ensure it is installed correctly in this Python environment.")
        print("Run: pip install deepfilternet")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_deepfilternet_files()
