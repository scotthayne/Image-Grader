import shlex
import subprocess
from pathlib import Path
import modal

# Define container dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "streamlit~=1.35.0",
    "numpy~=1.26.4",
    "pandas~=2.2.2"
)

app = modal.App(name="streamlit-dashboard", image=image)

# Mount the app.py script
streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

# Define the web server function
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