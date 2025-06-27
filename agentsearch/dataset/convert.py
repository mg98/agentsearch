import os
import subprocess
import shutil

if __name__ == "__main__":
    # Clear existing HTML files
    shutil.rmtree("papers/html", ignore_errors=True)
    
    try:
        # Run engrafo docker command
        subprocess.run([
            "docker", "run",
            "--rm",
            "-v", f"{os.getcwd()}/papers:/papers",
            "-w", "/papers",
            "engrafo",
            "engrafo",
            "src",
            "html",
            ], check=True)
    finally:
        # Clean up LaTeXML log files
        for f in os.listdir():
            if f.endswith('.latexml.log'):
                os.remove(f)
        print("Conversion completed")
