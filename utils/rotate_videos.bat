@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Directory containing the MP4 files
REM If the script is in the same directory as the videos, you can use:
REM SET "TARGET_DIR=."
REM Otherwise, specify the full path, e.g., SET "TARGET_DIR=C:\path\to\your\videos"
SET "TARGET_DIR=."

REM Suffix to add to the rotated files
SET "SUFFIX=_rotated"

REM Check if FFmpeg is accessible (in PATH or same directory)
ffmpeg -version >nul 2>nul
IF ERRORLEVEL 1 (
    echo FFmpeg could not be found.
    echo Please ensure FFmpeg is installed and in your system's PATH,
    echo or place ffmpeg.exe in the same directory as this script.
    pause
    exit /b
)

REM Loop through all .mp4 files in the target directory
FOR %%F IN ("%TARGET_DIR%\*.mp4") DO (
    SET "FILENAME=%%~nxF"
    SET "BASENAME=%%~nF"
    SET "EXTENSION=%%~xF"
    SET "OUTPUT_FILE=%TARGET_DIR%\!BASENAME!%SUFFIX%!EXTENSION!"

    echo Processing !FILENAME!...
    REM Check if the output file already exists before processing
    IF EXIST "!OUTPUT_FILE!" (
        echo Output file !OUTPUT_FILE! already exists. Skipping !FILENAME!.
    ) ELSE (
        ffmpeg -i "%%F" -vf "transpose=2" -c:a copy "!OUTPUT_FILE!"
        IF !ERRORLEVEL! EQU 0 (
            echo Successfully rotated !FILENAME! to !OUTPUT_FILE!
            REM --- ADDITION FOR DELETION ---
            echo Deleting original file !FILENAME!...
            DEL "%%F"
            IF !ERRORLEVEL! EQU 0 (
                echo Successfully deleted !FILENAME!
            ) ELSE (
                echo Error deleting !FILENAME!
            )
            REM --- END ADDITION ---
        ) ELSE (
            echo Error processing !FILENAME!
        )
    )
)

echo All .mp4 files processed.
pause