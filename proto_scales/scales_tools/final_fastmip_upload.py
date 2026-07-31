import os
import argparse
from ftplib import FTP

def ensure_remote_dir(ftp, remote_dir):
    """Create remote_dir if it does not already exist."""
    existing = ftp.nlst()
    print(existing)
    if remote_dir not in existing:
        ftp.mkd(remote_dir)
        print(f"Created remote directory: {remote_dir}")
    else:
        print(f"Remote directory already exists: {remote_dir}")


def upload_directory(ftp, local_dir, remote_dir):
    """Upload all files in local_dir into remote_dir on the FTP server."""
    files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    if not files:
        print(f"No files found in {local_dir}")
        return

    ftp.cwd(remote_dir)
    print(f"Current remote directory {ftp.pwd()}")
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        with open(local_path, "rb") as f:
            try:
              ftp.storbinary(f"STOR {filename}", f)
              print(f"Uploaded: {filename}")
            except error_perm as e:
                print(f"Skipped {filename}: {e}")
    print(f"\nUploaded {len(files)} file(s) to {remote_dir}/")


def download_file(ftp, remote_dir, filename, local_dir):
    """Download a single file from remote_dir into local_dir."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    ftp.cwd(remote_dir)
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
    print(f"Downloaded {filename} to {os.path.abspath(local_path)}")


HOST = "data.iac.ethz.ch"
USER = "fastmip"#"user4fastmip"


ftp = FTP(HOST)
ftp.login(USER, PASSWORD)
print(f"Connected to {HOST}, current dir: {ftp.pwd()}")
ftp.cwd("/FASTMIP_phase2")
print(f"Connected to {HOST}, current dir: {ftp.pwd()}")
ensure_remote_dir(ftp,"SCALES")
print(ftp.nlst())
local_path = "/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/fastMIP/results/"

upload_directory(ftp,local_path,"SCALES")
