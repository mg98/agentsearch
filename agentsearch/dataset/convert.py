import os
import glob
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import shutil

def convert_file(file_path: str) -> None:
    """
    Convert a LaTeX source file to HTML using Engrafo.
    
    Args:
        file_path (str): Path to the source tar.gz file
    """
    author_id = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    # remove ".tar.gz" extension from filename
    output_filename = os.path.splitext(os.path.splitext(filename)[0])[0]
    output_dir = f"papers/html/{author_id}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run engrafo docker command
    subprocess.run([
        "docker", "run",
        "-v", f"{os.getcwd()}:/papers",
        "-w", "/papers",
        "engrafo", "engrafo",
        file_path,
        output_dir,
        f"{output_filename}.html"
    ], check=True)

if __name__ == "__main__":
    # Clear existing HTML files
    shutil.rmtree("papers/html", ignore_errors=True)
    
    # Find all source files
    source_files = glob.glob("papers/src/*/*.tar.gz")
    
    # Convert files in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(convert_file, source_files)
