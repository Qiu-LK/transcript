from pathlib import Path
import os
import socket
import subprocess
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, port)) == 0


def main():
    port = next((candidate for candidate in range(8765, 8785) if not port_open(candidate)), None)
    if port is None:
        print("8765 至 8784 端口均被占用，请关闭之前的项目启动窗口后重试。")
        return 2
    env = os.environ.copy()
    env["WX_PORT"] = str(port)
    url = f"http://{HOST}:{port}"
    print(f"正在启动当前版本：{url}")
    opener = threading.Timer(2, lambda: webbrowser.open(url))
    opener.daemon = True
    opener.start()
    return subprocess.call([sys.executable, str(ROOT / "server.py")], cwd=ROOT, env=env)

if __name__ == "__main__":
    raise SystemExit(main())
