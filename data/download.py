#!/usr/bin/env python3
"""
Data download script
Data source: https://drive.google.com/file/d/1rH4oZODuC77ABOZbueVxbwv8rH9h71cr/view?usp=drive_link
"""

import os
import subprocess
import sys

# 配置
FILE_ID = "1rH4oZODuC77ABOZbueVxbwv8rH9h71cr"
OUTPUT = "dataset.zip"
URL = f"https://drive.google.com/uc?id={FILE_ID}"


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download():
    import gdown

    if os.path.exists(OUTPUT):
        print(f"{OUTPUT} already exists, skipping.")
        return

    print(f"Downloading {URL}")
    gdown.download(URL, OUTPUT, quiet=False)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    ensure_gdown()
    download()