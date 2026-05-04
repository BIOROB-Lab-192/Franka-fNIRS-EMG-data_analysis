"""
loader.py — Multi-Modal Data File Discovery
=============================================
Walks the raw data directory and discovers fNIRS, EMG, robot, and video
files for each participant session.

Expected folder structure:
    data/raw/{participant}_{condition}[{num}]/
        {session_date}/           ← SNiRF files here
        *.csv                     ← EMG Trigno CSV (direct child only)
        ...subfolder/             ← robot CSV + video MP4 here

Folder naming: {participant}_{robot|norobot}{num}
    e.g. sam_robot1, clarence_norobot, ronald_robot2

Usage:
    from src.loaders.loader import load_data
    files = load_data("data/raw")
"""

import os
import re
import pprint


def load_data(raw_dir):
    """
    Discover all data files for each participant session in raw_dir.

    Scans for folders matching {participant}_{robot|norobot}{num} and locates
    four file types by their position in the folder tree:
        - fNIRS:  .snirf inside a date-stamped session subfolder
        - EMG:    .csv sitting directly in the participant folder
        - Robot:  .csv in a deeper subfolder that also contains a video file
        - Video:  .mp4/.avi alongside the robot CSV

    Args:
        raw_dir: Path to the raw data directory containing participant folders.

    Returns:
        dict: Keys are "{participant}_{condition}[_{num}]" strings.
              Values are dicts with "fNIRS", "emg", "robot", "video" keys,
              each holding an absolute path or None if not found.
    """
    data_files = {}

    for folder in os.listdir(raw_dir):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        match = re.match(
            r"^(?P<participant>\w+?)_(?P<condition>robot|norobot)(?P<num>\d*)$", folder
        )
        if not match:
            continue

        num = match.group("num")
        key = f"{match.group('participant')}_{match.group('condition')}" + (
            f"_{num}" if num else ""
        )

        entry = {"fNIRS": None, "emg": None, "robot": None, "video": None}

        for session in sorted(os.listdir(folder_path)):
            session_path = os.path.join(folder_path, session)
            if not os.path.isdir(session_path):
                continue

            # fNIRS: inside date-stamped session folder
            for f in os.listdir(session_path):
                if f.endswith(".snirf"):
                    entry["fNIRS"] = os.path.join(session_path, f)
                    break

        # EMG: only CSV at the direct directory level of the participant folder
        for f in os.listdir(folder_path):
            if f.endswith(".csv"):
                entry["emg"] = os.path.join(folder_path, f)
                break

        # Robot + video live together in a subfolder. Walk the tree and
        # stop at the first directory containing both a CSV and a video file.
        for root, dirs, files in os.walk(folder_path):
            # Skip date-stamped session folders (fNIRS) and Trial_number folders
            if "Trial_number" in root or re.search(r"\d{4}-\d{2}-\d{2}", root):
                continue
            csvs = [f for f in files if f.endswith(".csv")]
            videos = [f for f in files if f.endswith((".mp4", ".avi"))]
            if csvs and videos:
                entry["robot"] = os.path.join(root, csvs[0])
                entry["video"] = os.path.join(root, videos[0])
                break

        data_files[key] = entry

    return data_files


if __name__ == "__main__":
    path = "/Users/haider/code/data_analysis/data/raw"
    pprint.pprint(load_data(path))