import os
import cv2
import csv
import sys


def save_video_info_to_csv(directory_path, output_csv_path):
    """
    Gets the name, duration, and frame count of video files in a directory
    using OpenCV and saves the information to a CSV file.

    Args:
        directory_path (str): The path to the directory containing video files.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    print(f"Scanning directory: {directory_path}\n")
    print(f"Writing video information to: {output_csv_path}\n")

    # List of common video file extensions (you can add more if needed)
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".ts"]

    # Open the CSV file for writing
    # newline='' is important to prevent extra blank rows in the CSV
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(
            ["Filename", "Duration (seconds)", "Duration (MM:SS)", "Frame Count"]
        )

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                # Check if the file extension is in our list of video extensions
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in video_extensions:
                    cap = None  # Initialize cap outside the try block
                    try:
                        # Create a VideoCapture object
                        cap = cv2.VideoCapture(file_path)

                        # Check if video file was opened successfully
                        if not cap.isOpened():
                            print(f"Could not open video file: {filename}. Skipping.\n")
                            # Write a row with partial info or error indication
                            csv_writer.writerow(
                                [filename, "Error: Could not open", "", ""]
                            )
                            continue  # Skip to the next file

                        # Get frame count and FPS
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        duration = 0
                        duration_formatted = "N/A"

                        if fps > 0:
                            duration = frame_count / fps  # Duration in seconds
                            minutes = int(duration // 60)
                            seconds = int(duration % 60)
                            duration_formatted = f"{minutes:02d}:{seconds:02d}"
                        else:
                            print(
                                f"Could not get FPS for {filename}. Cannot calculate duration.\n"
                            )
                            # Write a row with frame count but no duration
                            csv_writer.writerow(
                                [filename, "N/A (FPS=0)", "N/A", frame_count]
                            )
                            cap.release()  # Release the capture object
                            continue  # Skip writing the full info row and move to next file

                        # Release the video capture object
                        cap.release()

                        # Write the video information to the CSV file
                        csv_writer.writerow(
                            [
                                filename,
                                f"{duration:.2f}",
                                duration_formatted,
                                frame_count,
                            ]
                        )

                    except Exception as e:
                        print(
                            f"An error occurred with file {filename}: {e}. Skipping.\n"
                        )
                        # Write a row indicating an error occurred
                        csv_writer.writerow([filename, f"Error: {e}", "", ""])
                        if cap is not None:
                            cap.release()  # Ensure release if an error occurs after opening

                # Optional: else block to print files that were skipped
                # else:
                #     print(f"Skipping non-video file: {filename}")

    print(f"Finished processing directory. Information saved to {output_csv_path}")


# --- Replace with your directory path and desired output CSV path ---
video_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face448"
output_csv_file = "./video_length448.csv"  # Change this to your desired output file

# Run the function
save_video_info_to_csv(video_directory, output_csv_file)
