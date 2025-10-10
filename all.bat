@echo off
REM Batch script to run Python files in sequence

echo Starting video processing pipeline...

REM Assuming this script is in the project's root directory,
REM and 'utils' and 'models' are subdirectories.

echo ------------------------------------
echo Step 1: Running extract_faces_mediapipe.py...
echo ------------------------------------
python "C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\utils\extract_faces_mediapipe.py"

REM Check if the previous script was successful
IF ERRORLEVEL 1 (
  echo Error: extract_faces_mediapipe.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 2: Running analyze_videos.py...
echo ------------------------------------
python "C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\utils\analyze_videos.py"

REM Check if the previous script was successful
IF ERRORLEVEL 1 (
  echo Error: analyze_videos.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 2: Running trim_videos.py...
echo ------------------------------------
python "C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\utils\trim_videos.py"

IF ERRORLEVEL 1 (
  echo Error: trim_videos.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 3: Running organize_videos.py...
echo ------------------------------------
python "C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\utils\organize_videos.py"

IF ERRORLEVEL 1 (
  echo Error: organize_videos.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 4: Running timesformer.py...
echo ------------------------------------
python "C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\models\timesformer\timesformer-train.py"

IF ERRORLEVEL 1 (
  echo Error: models\timesformer.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo All scripts executed successfully.
echo Pipeline finished.

:eof