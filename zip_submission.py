import os
import zipfile
import shutil

TEAM_NUMBER = 10

zip_filename = f"DMA_project2_team{TEAM_NUMBER:02d}.zip"

print(f"Creating submission zip: {zip_filename}...")

# Folders to include
folders_to_include = ["AA", "SE", "CL"]

# If there is an existing zip file, remove it first
if os.path.exists(zip_filename):
    os.remove(zip_filename)

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for folder in folders_to_include:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found! Skipping...")
            continue

        print(f"Adding folder: {folder}...")
        for root, dirs, files in os.walk(folder):
            # Skip cache folders like __pycache__ or .antigravitycli
            if "__pycache__" in root or ".antigravitycli" in root:
                continue

            for file in files:
                filepath = os.path.join(root, file)

                # We only want to zip necessary files
                # For AA: part1.py, horizontal.pkl, association.pkl
                # For SE: make_index.py, CustomScoring.py, QueryResult.py, index folder
                # For CL: clasification.py, nb.pkl, svm.pkl

                # Check for files
                relative_path = os.path.relpath(filepath, ".")

                # Skip temp/log files
                if (
                    file.endswith(".log")
                    or file.endswith(".tmp")
                    or file.startswith("MAIN.tmp")
                ):
                    continue

                zipf.write(filepath, relative_path)
                print(f"  -> Added {relative_path}")

print(f"\nSuccessfully created {zip_filename}!")
print(
    "Please compile your 'DMA_project2_team01_보고서.md' to 'DMA_project2_team01_보고서.pdf' and add it and the presentation slides to this zip before final submission."
)
