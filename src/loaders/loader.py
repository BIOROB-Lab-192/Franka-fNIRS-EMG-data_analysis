import os
import re
import pprint


def load_data(raw_dir):
    """
    Discover all data files for a participant-session.

    Returns a dict:
        {key: {"fNIRS": snirf_path, "emg": emg_path, "robot": robot_csv_path, "video": video_path}}
    where key = "{participant}_{condition}[_{num}]"
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

        # Robot + video: in a deeper folder (has a CSV next to a video)
        for root, dirs, files in os.walk(folder_path):
            # Skip session date folders and Trial_number for robot search
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