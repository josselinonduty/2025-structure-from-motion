#!/usr/bin/env python3
"""
Structure from Motion (SfM) Pipeline

This script demonstrates how to generate a 3D point cloud from a set of images
using Structure from Motion (SfM) techniques. It uses OpenCV for feature extraction,
matching, and triangulation.

Usage:
    python main.py [--data_in DATA_IN] [--data_set DATA_SET] [--data_set_ext DATA_SET_EXT] [--data_out DATA_OUT]

Example:
    python main.py --data_in data --data_set castle-P30 --data_set_ext jpg --data_out out
"""

import cv2
import numpy as np
import os
import glob
import argparse
from matplotlib import pyplot as plt
import open3d as o3d


def get_original_image_id(id):
    """Convert internal image index to original image index"""
    return 31 - id if id < 16 else id - 16


def count_positive_depth(P1, P2, pts1, pts2, K):
    """Count the number of triangulated points with positive depth in both camera frames"""
    points_4d = cv2.triangulatePoints(K @ P1, K @ P2, pts1.T, pts2.T)
    points_3d = (
        points_4d[:3] / points_4d[3]
    )  # Convert to 3D by dividing by homogeneous coord

    # Convert to second camera's coordinate system
    points_cam2 = P2[:, :3] @ points_3d + P2[:, 3].reshape(3, 1)

    # Count points where Z > 0 in both camera frames
    return np.sum((points_3d[2] > 0) & (points_cam2[2] > 0))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Structure from Motion Pipeline")
    parser.add_argument(
        "--data_in", type=str, default="data", help="Input data directory"
    )
    parser.add_argument("--data_set", type=str, default="globe", help="Dataset name")
    parser.add_argument(
        "--data_set_ext", type=str, default="JPG", help="Dataset file extension"
    )
    parser.add_argument("--data_out", type=str, default="out", help="Output directory")
    parser.add_argument(
        "--show_plots", action="store_true", help="Display matplotlib plots"
    )
    args = parser.parse_args()

    # Set paths
    data_in = args.data_in
    data_set = args.data_set
    data_set_ext = args.data_set_ext
    data_out = args.data_out

    # Create output directory if it doesn't exist
    os.makedirs(data_out, exist_ok=True)

    # Load images
    image_paths = sorted(
        glob.glob(os.path.join(data_in, data_set, f"*.{data_set_ext}"))
    )

    if not image_paths:
        print(
            f"Error: No images found in {os.path.join(data_in, data_set)} with extension .{data_set_ext}"
        )
        return

    print(f"Loading {len(image_paths)} images...")
    images = [cv2.imread(img_path, cv2.IMREAD_COLOR_RGB) for img_path in image_paths]

    if data_set == "globe":
        print("Rearranging images for globe dataset...")
        images = images[31:15:-1] + images[0:16]
    elif len(images) < 10:
        print(f"Warning: Expected at least 10 images, but only found {len(images)}.")
        print("Proceeding with the available images without rearrangement.")

    print(f"Number of images: {len(images)}")

    # ------------------------------------------------------------------------
    # Step 1: Feature Extraction and Matching
    # ------------------------------------------------------------------------
    print("Step 1: Extracting and matching features...")

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Extract features
    keypoints = []
    descriptors = []
    for i, img in enumerate(images):
        if img is None:
            print(f"Warning: Image {i} failed to load. Skipping.")
            continue
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Match features between images
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = {}
    for i in range(len(images) - 1):
        if descriptors[i] is None or descriptors[i + 1] is None:
            print(
                f"Warning: Descriptors missing for image pair ({i}, {i+1}). Skipping."
            )
            continue
        matches[(i, i + 1)] = bf.match(descriptors[i], descriptors[i + 1])
        print(
            f"Number of matches between images {get_original_image_id(i)} and {get_original_image_id(i + 1)}: {len(matches[(i, i + 1)])}"
        )

    # Sort matches by distance
    matches = {k: sorted(v, key=lambda x: x.distance) for k, v in matches.items()}

    # Plot matches
    if args.show_plots:
        for i in range(min(3, len(images) - 1)):
            if (i, i + 1) not in matches:
                continue
            img_i = cv2.drawMatches(
                images[i],
                keypoints[i],
                images[i + 1],
                keypoints[i + 1],
                matches[(i, i + 1)],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            plt.figure(figsize=(12, 6))
            plt.imshow(img_i)
            plt.axis("off")
            plt.title(
                f"Initial matches: images {get_original_image_id(i)} and {get_original_image_id(i+1)}"
            )
            plt.show()

    # ------------------------------------------------------------------------
    # Step 2: Eliminate Outliers
    # ------------------------------------------------------------------------
    print("Step 2: Eliminating outliers...")

    # Remove outliers from matches using cv2.findHomography
    for i, j in list(matches.keys()):  # Create a copy of keys to avoid runtime error
        try:
            kp1 = np.array([keypoints[i][m.queryIdx].pt for m in matches[(i, j)]])
            kp2 = np.array([keypoints[j][m.trainIdx].pt for m in matches[(i, j)]])

            if len(kp1) < 4 or len(kp2) < 4:
                print(
                    f"Warning: Not enough keypoints for homography estimation between images {i} and {j}. Skipping."
                )
                continue

            H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)
            if H is None:
                print(
                    f"Warning: Homography estimation failed for images {i} and {j}. Skipping."
                )
                continue

            matches[(i, j)] = [m for m, msk in zip(matches[(i, j)], mask) if msk]
        except Exception as e:
            print(f"Error in homography estimation for images {i} and {j}: {e}")

    # Plot matches after outlier removal
    if args.show_plots:
        for i in range(min(3, len(images) - 1)):
            if (i, i + 1) not in matches:
                continue
            img_i = cv2.drawMatches(
                images[i],
                keypoints[i],
                images[i + 1],
                keypoints[i + 1],
                matches[(i, i + 1)],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            plt.figure(figsize=(12, 6))
            plt.title(
                f"Filtered matches between images {get_original_image_id(i)} and {get_original_image_id(i+1)}"
            )
            plt.imshow(img_i)
            plt.axis("off")
            plt.show()

    # ------------------------------------------------------------------------
    # Step 3: Essential Matrix Estimation
    # ------------------------------------------------------------------------
    print("Step 3: Estimating essential matrix...")

    if not matches:
        print("Error: No valid matches found between any image pairs. Exiting.")
        return

    # Find the best image pair (with the most matches)
    try:
        best_pair_idx = np.argmax([len(m) for m in matches.values()])
        best_matches = list(matches.items())[best_pair_idx][1]

        print(f"Number of matches in best pair: {len(best_matches)}")
        best_pair_i, best_pair_j = list(matches.keys())[best_pair_idx]
        print(
            f"Best pair: ({get_original_image_id(best_pair_i)}, {get_original_image_id(best_pair_j)})"
        )

        # Get points from matches
        pts1 = np.float32([keypoints[best_pair_i][m.queryIdx].pt for m in best_matches])
        pts2 = np.float32([keypoints[best_pair_j][m.trainIdx].pt for m in best_matches])

        # Compute the essential matrix
        # Camera intrinsic matrix (typically should be calibrated per camera)
        K = np.array(
            [
                [1698.873755, 0.000000, 971.7497705],
                [0.000000, 1698.8796645, 647.7488275],
                [0.000000, 0.000000, 1.000000],
            ]
        )
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=5.0
        )

        num_outliers = np.sum(mask == 0)
        print(f"Number of outliers: {num_outliers}")

        num_inliers = np.sum(mask == 1)
        print(f"Number of inliers: {num_inliers}")

        # Plot the best pair of images
        if args.show_plots:
            img_best_pair = cv2.drawMatches(
                images[best_pair_i],
                keypoints[best_pair_i],
                images[best_pair_j],
                keypoints[best_pair_j],
                best_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            plt.figure(figsize=(12, 6))
            plt.title(
                f"Best matches between images {get_original_image_id(best_pair_i)} and {get_original_image_id(best_pair_j)}"
            )
            plt.imshow(img_best_pair)
            plt.axis("off")
            plt.show()

        # ------------------------------------------------------------------------
        # Step 4: Camera Pose Estimation
        # ------------------------------------------------------------------------
        print("Step 4: Estimating camera poses...")

        R1, R2, t = cv2.decomposeEssentialMat(E)

        # Select the correct rotation and translation
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at (I|0)
        P2s = [
            np.hstack((R1, t)),
            np.hstack((R1, -t)),
            np.hstack((R2, t)),
            np.hstack((R2, -t)),
        ]

        # Select the best P2 based on positive depth count
        P2 = max(P2s, key=lambda P2_i: count_positive_depth(P1, P2_i, pts1, pts2, K))

        print("R:", P2[:, :3])
        print("t:", P2[:, 3])

        # Double check the results
        _, R_, t_, mask = cv2.recoverPose(E, pts1, pts2, K)
        assert np.allclose(R_, P2[:, :3])
        assert np.allclose(t_.reshape(-1), P2[:, 3])

        # ------------------------------------------------------------------------
        # Step 5: 3D Reconstruction
        # ------------------------------------------------------------------------
        print("Step 5: Reconstructing 3D points...")

        # Reconstruct 3D points using cv2.triangulatePoints
        points = cv2.triangulatePoints(K @ P1, K @ P2, pts1.T, pts2.T)
        points /= points[3]

        # Create point cloud and save to file
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.T[:, :3])

        output_file = os.path.join(data_out, f"{data_set}-2.ply")
        o3d.io.write_point_cloud(output_file, cloud)
        print(f"Point cloud saved to {output_file}")

        # TODO: Implement Step 5 (Growing step) from the notebook if needed

    except Exception as e:
        print(f"Error during reconstruction: {e}")


if __name__ == "__main__":
    main()
