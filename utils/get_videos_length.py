import argparse
import os
import csv
import cv2


def save_imagenet_style_videos_to_csv(dataset_root, output_csv_path):
    """
    Reads an ImageNet-style dataset where each class is a folder,
    but the samples are video files such as .mp4.

    Expected structure:
        dataset_root/
            class_1/
                video001.mp4
                video002.mp4
            class_2/
                video003.mp4

    Output CSV columns:
        Video Path, Filename, Class Name, Class Index,
        Duration, Frame Count, FPS, Width, Height
    """

    print(f"Scanning dataset root: {dataset_root}")
    print(f"Writing video information to: {output_csv_path}\n")

    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".ts"]

    class_names = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    class_names.sort()

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(
            [
                "Video Path",
                "Filename",
                "Class Name",
                "Class Index",
                "Duration (seconds)",
                "Duration (MM:SS)",
                "Frame Count",
                "FPS",
                "Width",
                "Height",
            ]
        )

        total_videos = 0

        for class_name in class_names:
            class_dir = os.path.join(dataset_root, class_name)
            class_idx = class_to_idx[class_name]

            print(f"Processing class: {class_name} -> label {class_idx}")

            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)

                if not os.path.isfile(file_path):
                    continue

                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension not in video_extensions:
                    continue

                cap = None

                try:
                    cap = cv2.VideoCapture(file_path)

                    if not cap.isOpened():
                        print(f"Could not open video: {file_path}")
                        csv_writer.writerow(
                            [
                                file_path,
                                filename,
                                class_name,
                                class_idx,
                                "Error: could not open",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                        continue

                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if fps > 0:
                        duration = frame_count / fps
                        minutes = int(duration // 60)
                        seconds = int(duration % 60)
                        duration_formatted = f"{minutes:02d}:{seconds:02d}"
                    else:
                        duration = "N/A"
                        duration_formatted = "N/A"

                    csv_writer.writerow(
                        [
                            file_path,
                            filename,
                            class_name,
                            class_idx,
                            (
                                f"{duration:.2f}"
                                if isinstance(duration, float)
                                else duration
                            ),
                            duration_formatted,
                            frame_count,
                            f"{fps:.2f}" if fps > 0 else "N/A",
                            width,
                            height,
                        ]
                    )

                    total_videos += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    csv_writer.writerow(
                        [
                            file_path,
                            filename,
                            class_name,
                            class_idx,
                            f"Error: {e}",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )

                finally:
                    if cap is not None:
                        cap.release()

    print(f"\nFinished. Found {total_videos} video files.")
    print(f"CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report per-video duration/resolution stats for an ImageNet-style video dataset"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="ImageNet-style root of class-labeled videos to inspect.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./imagenet_style_video_dataset_info.csv",
        help="Path to write the resulting CSV report to.",
    )
    args = parser.parse_args()

    save_imagenet_style_videos_to_csv(args.dataset_root, args.output_csv)
