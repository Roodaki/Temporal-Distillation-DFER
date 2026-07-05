import argparse
import os
import shutil
import re  # Import regular expressions module
import pathlib


def organize_videos_by_class(source_directory, destination_root_directory):
    """
    Organizes video files from a source directory into class-based subfolders
    within a specified destination root directory.

    Handles filenames in formats:
      1. <original_stem>_trimmed_<number>.mp4
      2. <original_stem>_rnd_<number>.mp4  (The new random sampling format)

    Args:
        source_directory (str): The path to the directory containing the video files.
        destination_root_directory (str): The path where the new root folder
                                          for organized videos will be created.
    """
    source_path = pathlib.Path(source_directory)
    dest_path = pathlib.Path(destination_root_directory)

    try:
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured destination root directory exists: {dest_path}")
    except OSError as e:
        print(f"Error creating destination root directory {dest_path}: {e}")
        return

    # --- REGEX UPDATE ---
    # This pattern now looks for either "_trimmed_" OR "_rnd_" followed by digits
    # Group 1: The original stem (everything before the suffix)
    # Group 2: The extension (e.g., .mp4, .avi)
    filename_pattern = re.compile(r"^(.*?)_(?:trimmed|rnd)_\d+(\.\w+)$", re.IGNORECASE)

    files_processed_count = 0

    for file_path in source_path.iterdir():
        if file_path.is_file():
            filename = file_path.name

            match = filename_pattern.match(filename)
            if match:
                original_stem = match.group(1)  # Get the part before _rnd_ or _trimmed_

                # print(f"Processing: {filename} -> Stem: {original_stem}")

                try:
                    # Extract the class name from the original_stem
                    # Assumes original_stem is like: <id>_<class_name>_...
                    stem_parts = original_stem.split("_")

                    if len(stem_parts) < 2:
                        print(
                            f"  [Skipping] Could not extract class from stem '{original_stem}' in {filename}"
                        )
                        continue

                    # Class name is the second part (index 1), converted to lowercase
                    class_name = stem_parts[1].lower()

                    if not class_name:
                        print(
                            f"  [Skipping] Empty class name extracted from {filename}"
                        )
                        continue

                    # Define Class Folder
                    class_destination_directory = dest_path / class_name
                    class_destination_directory.mkdir(exist_ok=True)

                    # Define Destination File Path
                    destination_filepath = class_destination_directory / filename

                    # Copy the file
                    shutil.copy2(file_path, destination_filepath)
                    files_processed_count += 1

                    # Optional: Print every 100 files to avoid cluttering console
                    if files_processed_count % 100 == 0:
                        print(f"Processed {files_processed_count} files...")

                except Exception as e:
                    print(f"  [Error] Processing {filename}: {e}")
            else:
                # Useful to debug if files aren't matching
                # print(f"  [Ignored] Pattern mismatch: {filename}")
                pass

    print(f"\nFinished. Successfully organized {files_processed_count} videos.")


# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize trimmed clips into class-based subfolders by filename"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing *_trimmed_N.mp4 / *_rnd_N.mp4 clips to organize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory where class-based subfolders will be created.",
    )
    args = parser.parse_args()

    if os.path.exists(args.source_dir):
        organize_videos_by_class(args.source_dir, args.output_dir)
    else:
        print(f"Source directory not found: {args.source_dir}")
