@echo off
REM Example usage scripts for preprocessing QXSLAB_SAROPT images

echo ========================================
echo SAR Image Preprocessing Examples
echo ========================================
echo.

REM Example 1: Preprocess single image
echo Example 1: Preprocess a single SAR image
echo Command:
echo python preprocess_qxslab.py --input "path/to/sar_image.png" --output "path/to/output_image.png"
echo.

REM Example 2: Batch preprocess directory
echo Example 2: Batch preprocess all images in a directory
echo Command:
echo python preprocess_qxslab.py --input "E:/SAR_Dataset/QXSLAB_SAROPT/sar/" --output "E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed/"
echo.

REM Example 3: Preprocess with custom parameters
echo Example 3: Preprocess with custom filter settings
echo Command:
echo python preprocess_qxslab.py ^
    --input "E:/SAR_Dataset/QXSLAB_SAROPT/sar/" ^
    --output "E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed/" ^
    --filter lee ^
    --window-size 7 ^
    --percentile-low 2.0 ^
    --percentile-high 98.0 ^
    --gamma 0.9
echo.

REM Example 4: Batch preprocess and run inference
echo Example 4: Batch preprocess and run SAR2Optical inference
echo Command:
echo python batch_preprocess_and_infer.py ^
    --input "E:/SAR_Dataset/QXSLAB_SAROPT/sar/" ^
    --output "E:/SAR_Dataset/QXSLAB_SAROPT/output_optical/" ^
    --save-preprocessed ^
    --filter lee ^
    --window-size 5
echo.

echo ========================================
echo To run any of these commands, copy and paste them into your terminal
echo Remember to replace paths with your actual file/directory paths
echo ========================================
echo.

pause
