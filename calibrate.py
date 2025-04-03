#!/usr/bin/env python3
"""
Camera Calibration Script

This script performs camera calibration using a checkerboard pattern.
It calculates the camera matrix and distortion coefficients and saves them to files.

Usage:
    python calibrate.py [--board (X,Y)] [--data_in DATA_IN] [--data_set DATA_SET] [--data_set_ext DATA_SET_EXT] [--data_out DATA_OUT]

Example:
    python calibrate.py --board 6,8 --data_in data --data_set otter --data_set_ext JPG --data_out data/otter
"""

import argparse
import glob
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Script")
    parser.add_argument(
        "--board",
        type=str,
        default="6,8",
        help="Checkerboard dimensions as X,Y",
    )
    parser.add_argument(
        "--data_in", type=str, default="data", help="Input data directory"
    )
    parser.add_argument(
        "--data_set", type=str, default="otter", help="Dataset name or subdirectory"
    )
    parser.add_argument(
        "--data_set_ext", type=str, default="JPG", help="Dataset file extension"
    )
    parser.add_argument(
        "--data_out", type=str, default="data/otter", help="Output directory"
    )
    args = parser.parse_args()

    CHECKERBOARD = tuple(map(int, args.board.split(",")) if args.board else (6, 8))
    print(f"Checkerboard dimensions: {CHECKERBOARD}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

    points_3d = []
    points_2d = []

    image_paths = os.path.join(
        args.data_in, args.data_set, "calibration", f"*.{args.data_set_ext}"
    )
    images = glob.glob(image_paths)

    if not images:
        print(f"No images found at {image_paths}")
        return

    print(f"Found {len(images)} images for calibration")

    for fname in images:
        print(f"Processing {fname}...")

        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        b, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if b:
            points_3d.append(objp)

            image_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            points_2d.append(image_corners)
            print(f"Found corners in {fname}")

    if not points_3d:
        print("No chessboard patterns found in any images. Calibration failed.")
        return

    _, K, D, _, _ = cv2.calibrateCamera(
        points_3d, points_2d, gray.shape[::-1], None, None
    )

    print("Camera matrix (K):\n", K)
    print("\nDistortion coefficients (D):\n", D)

    os.makedirs(args.data_out, exist_ok=True)

    K_path = os.path.join(args.data_out, "K.txt")
    D_path = os.path.join(args.data_out, "D.txt")

    np.savetxt(K_path, K, fmt="%f", delimiter=" ")

    if D.shape[1] > 5:
        D = D[:, :5]
    np.savetxt(D_path, D, fmt="%f", delimiter=" ")

    print(f"\nCalibration complete. Results saved to {K_path} and {D_path}")


if __name__ == "__main__":
    main()
