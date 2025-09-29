import subprocess
import webbrowser
import requests
import time
import os

server_process = subprocess.Popen([
    'python', '-m', 'uvicorn', 'backend.server:app', '--reload'
])

status_code = 0
while status_code != 200:
    try:
        r = requests.get("http://localhost:8000/api/health")
        status_code = r.status_code
    except Exception:
        pass
    time.sleep(1)

frontend_path = os.path.abspath(os.path.join('frontend', 'homepage.html'))
webbrowser.open(f'file://{frontend_path}')

try:
    print("Press Ctrl+C to stop the server and exit...")
    while server_process.poll() is None:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting...")
finally:
    server_process.terminate()
    server_process.wait()