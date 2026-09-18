import os
from ftplib import FTP, error_perm


# same layout as final_upload_reformatted.py: {indicator}/ann/{scenario}
INDICATORS = ["tas", "pr"]
LOCAL_BASE = "/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/fastMIP/results"
REMOTE_BASE = "/FASTMIP_phase2/SCALES"


def remote_dir_exists(ftp, remote_dir):
    """Check whether remote_dir exists, without creating it."""
    parent, name = remote_dir.rsplit("/", 1)
    try:
        existing = ftp.nlst(parent or "/")
    except error_perm:
        return False
    return name in {os.path.basename(e) for e in existing}


def list_remote_files(ftp, remote_dir):
    """List filenames present in remote_dir."""
    try:
        entries = ftp.nlst(remote_dir)
    except error_perm:
        return []
    return {os.path.basename(e) for e in entries}


def check_scenario(ftp, local_dir, remote_dir):
    """Compare local_dir's files against remote_dir's files. Returns (missing, extra)."""
    local_files = {
        f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))
    }
    if not remote_dir_exists(ftp, remote_dir):
        return sorted(local_files), []

    remote_files = list_remote_files(ftp, remote_dir)
    missing = sorted(local_files - remote_files)
    extra = sorted(remote_files - local_files)
    return missing, extra


ftp = FTP(HOST)
ftp.login(USER, PASSWORD)
print(f"Connected to {HOST}, current dir: {ftp.pwd()}")

any_missing = False
for indicator in INDICATORS:
    ann_dir = os.path.join(LOCAL_BASE, indicator, "ann")
    scenarios = sorted(
        s for s in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, s))
    )

    for scenario in scenarios:
        local_dir = os.path.join(ann_dir, scenario)
        remote_dir = f"{REMOTE_BASE}/{indicator}/ann/{scenario}"

        missing, extra = check_scenario(ftp, local_dir, remote_dir)
        if missing:
            any_missing = True
            print(f"[MISSING] {remote_dir}: {len(missing)} file(s) not uploaded -> {missing}")
        else:
            print(f"[OK] {remote_dir}: all local files present")
        if extra:
            print(f"[EXTRA] {remote_dir}: {len(extra)} remote file(s) not found locally -> {extra}")

ftp.quit()

print("\nSome files are missing on the remote server." if any_missing
      else "\nAll local files are present on the remote server.")
