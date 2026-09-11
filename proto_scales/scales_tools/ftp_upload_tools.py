import os
import argparse
from ftplib import FTP


LOCAL_DIR   = os.path.join(os.path.dirname(__file__), "downloads")
REMOTE_DIR  = "SCALES"


def ensure_remote_dir(ftp, remote_dir):
    """Create remote_dir if it does not already exist."""
    existing = ftp.nlst()
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
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {filename}", f)
        print(f"Uploaded: {filename}")
    print(f"\nUploaded {len(files)} file(s) to {remote_dir}/")


def download_file(ftp, remote_dir, filename, local_dir):
    """Download a single file from remote_dir into local_dir."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    ftp.cwd(remote_dir)
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
    print(f"Downloaded {filename} to {os.path.abspath(local_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FTP upload/download tool")
    parser.add_argument("action", choices=["upload", "download", "mkdir", "ls"],
                        help="Action to perform")
    parser.add_argument("--local_dir",  default=LOCAL_DIR,  help="Local directory for upload/download")
    parser.add_argument("--remote_dir", default=REMOTE_DIR, help="Remote directory on FTP server")
    parser.add_argument("--filename",   default=None,        help="Filename for download action")
    args = parser.parse_args()

    ftp = FTP(HOST)
    ftp.login(USER, PASSWORD)
    print(f"Connected to {HOST}, current dir: {ftp.pwd()}")

    if args.action == "ls":
        ftp.dir()

    elif args.action == "mkdir":
        ensure_remote_dir(ftp, args.remote_dir)

    elif args.action == "upload":
        ensure_remote_dir(ftp, args.remote_dir)
        ftp.cwd("/")   # reset to root before cwd in upload_directory
        upload_directory(ftp, args.local_dir, args.remote_dir)

    elif args.action == "download":
        if args.filename is None:
            parser.error("--filename is required for download")
        download_file(ftp, args.remote_dir, args.filename, args.local_dir)

    ftp.quit()
