import os
from ftplib import FTP, error_perm


def ensure_remote_dir(ftp, dirname):
    """Create dirname in the current remote directory if it does not already exist."""
    existing = ftp.nlst()
    if dirname not in existing:
        ftp.mkd(dirname)
        print(f"Created remote directory: {dirname}")
    else:
        print(f"Remote directory already exists: {dirname}")


def ensure_remote_path(ftp, remote_path):
    """Create each component of remote_path (relative to ftp's current dir) as needed, cd'ing into it."""
    for part in remote_path.strip("/").split("/"):
        ensure_remote_dir(ftp, part)
        ftp.cwd(part)


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


# indicators live in separate local/remote subdirectories, both laid out as
# {indicator}/ann/{scenario}
INDICATORS = ["tas", "pr"]
LOCAL_BASE = "/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/fastMIP/results"


ftp = FTP(HOST)
ftp.login(USER, PASSWORD)
print(f"Connected to {HOST}, current dir: {ftp.pwd()}")
ftp.cwd("/FASTMIP_phase2")
ensure_remote_dir(ftp, "SCALES")
ftp.cwd("SCALES")
scales_root = ftp.pwd()

for indicator in INDICATORS:
    ann_dir = os.path.join(LOCAL_BASE, indicator, "ann")
    scenarios = sorted(
        s for s in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, s))
    )

    for scenario in scenarios:
        ftp.cwd(scales_root)
        ensure_remote_path(ftp, f"{indicator}/ann/{scenario}")
        remote_dir = ftp.pwd()

        local_dir = os.path.join(ann_dir, scenario)
        upload_directory(ftp, local_dir, remote_dir)
