import os
import sys
import subprocess

def check_and_install_dependencies():
    print("Checking dependencies...", flush=True)
    required_packages = ["fastapi", "uvicorn", "torch", "transformers", "websockets", "google-generativeai"]
    
    missing_packages = []
    for pkg in required_packages:
        try:
            if pkg == "websockets":
                import websockets
            elif pkg == "transformers":
                import transformers
            elif pkg == "torch":
                import torch
            elif pkg == "fastapi":
                import fastapi
            elif pkg == "uvicorn":
                import uvicorn
            elif pkg == "google-generativeai":
                import google.generativeai
        except ImportError:
            missing_packages.append(pkg)
            
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}...", flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("Dependencies installed successfully!", flush=True)
        except Exception as e:
            print(f"Error installing dependencies: {e}", flush=True)
            print("Please run manually: pip install " + " ".join(missing_packages), flush=True)
    else:
        print("All dependencies are satisfied.", flush=True)

if __name__ == "__main__":
    # Perform dependency checks
    check_and_install_dependencies()
    
    # Run the Uvicorn server
    import uvicorn
    print("\n" + "="*50)
    print("   Starting AI Code Architect Server on http://127.0.0.1:5000")
    print("="*50 + "\n", flush=True)
    
    # We run uvicorn server in app/server.py
    uvicorn.run("app.server:app", host="127.0.0.1", port=5000, log_level="info")
