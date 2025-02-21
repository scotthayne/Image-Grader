# How to deploy Streamlit App to Modal

## Setup Environment
```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install modal streamlit
# Add any other dependencies your app needs
```

## Setup Modal (One-time)
```bash
source .venv/bin/activate && modal setup
```

## Deployment Files
1. Your Streamlit app: `app.py`
2. Modal deployment script: `serve_streamlit.py`

## Modal Deployment Script
Create `serve_streamlit.py`:
```python
import shlex
import subprocess
from pathlib import Path
import modal

# Container setup with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "streamlit~=1.35.0",
    # Add your app's dependencies here
)

app = modal.App(name="streamlit-dashboard", image=image)

# Mount app.py
streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
)
@modal.web_server(8000)
def run():
    cmd = f"streamlit run {streamlit_script_remote_path} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(shlex.split(cmd))

if __name__ == "__main__":
    app.serve()
```

## Deploy
```bash
modal deploy serve_streamlit.py
```

Your app will be available at: `https://[username]--[app-name]-run.modal.run`
