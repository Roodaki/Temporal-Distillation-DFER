@echo off
REM Batch script to run the full pipeline in sequence.
REM Assumes this script lives in the project root, with 'utils' as a subdirectory.

SET "ROOT=%~dp0"

echo Starting video processing pipeline...

echo ------------------------------------
echo Step 1: Running extract_faces_mediapipe.py...
echo ------------------------------------
python "%ROOT%utils\extract_faces_mediapipe.py"

REM Check if the previous script was successful
IF ERRORLEVEL 1 (
  echo Error: extract_faces_mediapipe.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 2: Running analyze_videos.py...
echo ------------------------------------
python "%ROOT%utils\analyze_videos.py"

REM Check if the previous script was successful
IF ERRORLEVEL 1 (
  echo Error: analyze_videos.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 3: Running trim_videos_emotion.py...
echo ------------------------------------
python "%ROOT%utils\trim_videos_emotion.py"

IF ERRORLEVEL 1 (
  echo Error: trim_videos_emotion.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 4: Running organize_videos.py...
echo ------------------------------------
python "%ROOT%utils\organize_videos.py"

IF ERRORLEVEL 1 (
  echo Error: organize_videos.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 5: Running timesformer_train_offline.py...
echo ------------------------------------
python "%ROOT%utils\timesformer_train_offline.py"

IF ERRORLEVEL 1 (
  echo Error: timesformer_train_offline.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo Step 6: Running vivit_train_offline.py...
echo ------------------------------------
python "%ROOT%utils\vivit_train_offline.py"

IF ERRORLEVEL 1 (
  echo Error: vivit_train_offline.py failed. Exiting.
  goto :eof
)

echo ------------------------------------
echo All scripts executed successfully.
echo Pipeline finished.

:eof