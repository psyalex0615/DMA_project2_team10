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
                # For SE: make_index.py, CustomScoring.py, QueryResult.py, se_analyzer.py, index folder
                # For CL: clasification.py, nb.pkl, svm.pkl

                relative_path = os.path.relpath(filepath, ".")
                parts = relative_path.replace("\\", "/").split("/")
                top_folder = parts[0]

                keep = False
                if top_folder == "AA":
                    if file == "part1.py" or "part1_horizontal.pkl" in file or "part1_association.pkl" in file:
                        keep = True
                elif top_folder == "SE":
                    if "index" in parts or file in ["make_index.py", "CustomScoring.py", "QueryResult.py", "se_analyzer.py"]:
                        keep = True
                elif top_folder == "CL":
                    if file == "clasification.py" or file.endswith("_nb.pkl") or file.endswith("_svm.pkl"):
                        keep = True

                if not keep:
                    continue

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
    f"Please compile your 'DMA_project2_team{TEAM_NUMBER:02d}_보고서.md' to 'DMA_project2_team{TEAM_NUMBER:02d}_보고서.pdf' and add it and the presentation slides to this zip before final submission."
)

