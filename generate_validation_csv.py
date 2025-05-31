import os
import pandas as pd
import re

# Path to the validation directory
VALIDATION_DIR = "validation"

# List all possible conditions (update this list as needed)
CONDITIONS = set()

# First pass: collect all unique conditions
for folder in sorted(os.listdir(VALIDATION_DIR)):
    folder_path = os.path.join(VALIDATION_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    pngs = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    for png in pngs:
        match = re.match(r".*-(.+)\.png", png)
        if match:
            CONDITIONS.add(match.group(1))

CONDITIONS = sorted(list(CONDITIONS))

# Second pass: build the CSV rows
rows = []
for folder in sorted(os.listdir(VALIDATION_DIR)):
    folder_path = os.path.join(VALIDATION_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    npy_file = f"{folder}.npy"
    npy_path = os.path.join(folder_path, npy_file)
    pngs = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    if pngs and os.path.exists(npy_path):
        match = re.match(r".*-(.+)\.png", pngs[0])
        if match:
            label = match.group(1)
            row = {"filename": npy_file}
            for cond in CONDITIONS:
                row[cond] = int(cond == label)
            rows.append(row)

# Create DataFrame and save
if rows:
    df = pd.DataFrame(rows)
    df.to_csv("validation.csv", index=False)
    print(f"validation.csv generated with columns: {['filename'] + CONDITIONS}")
else:
    print("No validation data found.") 