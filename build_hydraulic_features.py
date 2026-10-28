""" Downloads the UCI "Condition monitoring of hydraulic systems dataset" (a zip file)
    Reads each sensor file (pressure, flow, temperature, etc.).
    Reduces each sensor to two numbers per cycle: mean and std into just 34 features.
    Reads the pump leakage label (0, 1 or 2) from profile.txt and Saves to hydraulic_features.csv.
"""
import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
ZIP_PATH = "hydraulic.zip"

# Download the zip file with raw sensor data (skip if the zip already exists)
if not os.path.exists(ZIP_PATH):
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                bar.update(len(chunk))
else:
    print("Zip file already downloaded, skipping the download.")

# Extract the features into mean and std
sensors = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
           "FS1", "FS2", "TS1", "TS2", "TS3", "TS4",
           "VS1", "CE", "CP", "SE"]

features = {}
with zipfile.ZipFile(ZIP_PATH) as zf:
    for s in tqdm(sensors, desc="Processing sensors data"):
        with zf.open(f"{s}.txt") as f:
            df = pd.read_csv(f, sep="\t", header=None)
        features[f"{s}_mean"] = df.mean(axis=1)
        features[f"{s}_std"] = df.std(axis=1)

    # profile.txt columns: cooler, valve, pump leakage, accumulator, stable flag
    with zf.open("profile.txt") as f:
        profile = pd.read_csv(f, sep="\t", header=None)

# Save the data to "hydraulic_features.csv"
X = pd.DataFrame(features)
X["pump_leakage"] = profile[2]
X.to_csv("hydraulic_features.csv", index=False)
print(f"Saved the features to hydraulic_features.csv, size: {X.shape}")