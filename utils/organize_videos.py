import os
import shutil
import re  # Import regular expressions module


def organize_videos_by_class(source_directory, destination_root_directory):
    """
    Organizes video files from a source directory into class-based subfolders
    within a specified destination root directory.

    Handles filenames in the new format: <original_stem>_trimmed_<number>.mp4
    where <original_stem> is assumed to be like <id>_<class_name>_...

    Args:
        source_directory (str): The path to the directory containing the video files.
        destination_root_directory (str): The path where the new root folder
                                          for organized videos will be created.
    """
    try:
        os.makedirs(destination_root_directory, exist_ok=True)
        print(
            f"Ensured destination root directory exists: {destination_root_directory}"
        )
    except OSError as e:
        print(
            f"Error creating destination root directory {destination_root_directory}: {e}"
        )
        return

    # Regex to identify new trimmed video filenames like "..._trimmed_1.mp4", "..._trimmed_12.mp4"
    # It captures the part before "_trimmed_"
    trimmed_video_pattern = re.compile(r"^(.*?)_trimmed_\d+\.mp4$", re.IGNORECASE)

    for filename in os.listdir(source_directory):
        source_filepath = os.path.join(source_directory, filename)

        if os.path.isfile(source_filepath):
            match = trimmed_video_pattern.match(filename)
            if match:
                original_stem = match.group(
                    1
                )  # Get the part before "_trimmed_<number>.mp4"
                print(f"Processing file: {filename} (Original Stem: {original_stem})")

                try:
                    # Extract the class name from the original_stem
                    # Assumes original_stem is like: <id>_<class_name>_optional_other_parts
                    stem_parts = original_stem.split("_")

                    if len(stem_parts) < 2:
                        print(
                            f"Could not extract class name from stem '{original_stem}' of {filename}. Skipping."
                        )
                        continue

                    class_name = stem_parts[
                        1
                    ].lower()  # Class name is the second part, converted to lowercase

                    if (
                        not class_name
                    ):  # Should be caught by len(stem_parts) < 2 already
                        print(f"Extracted empty class name from {filename}. Skipping.")
                        continue

                    class_destination_directory = os.path.join(
                        destination_root_directory, class_name
                    )
                    os.makedirs(class_destination_directory, exist_ok=True)
                    # print(f"Ensured class directory exists: {class_destination_directory}") # Less verbose

                    destination_filepath = os.path.join(
                        class_destination_directory,
                        filename,  # Keep original filename for the copy
                    )

                    shutil.copy2(source_filepath, destination_filepath)
                    print(f"Copied '{filename}' to '{class_destination_directory}'")

                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")
            else:
                (f"Skipping file with unexpected name format: {filename}")
        else:
            print(f"Skipping non-file item: {filename}")


# --- Example Usage ---
# Replace with the actual path to your NEWLY trimmed video files
source_directory_path = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face224_trimmed32"  # MODIFIED Example Path

# Replace with the desired path for the organized videos based on the new naming
destination_root_directory_path = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face224_trimmed32_organized"  # MODIFIED Example Path

organize_videos_by_class(source_directory_path, destination_root_directory_path)

print("\nVideo organization process finished.")
