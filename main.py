#!/usr/bin/env python3
"""
Structure from Motion (SfM) Pipeline

This script demonstrates how to generate a 3D point cloud from a set of images
using Structure from Motion (SfM) techniques. It uses OpenCV for feature extraction,
matching, and triangulation.

Usage:
    python main.py
        [--data_in DATA_IN]
        [--data_set DATA_SET]
        [--data_set_ext DATA_SET_EXT]
        [--data_out DATA_OUT]
        [--data_k DATA_K]
        [--data_d DATA_D]
        [--show_plots]
        [--color_mode COLOR_MODE]

Example:
    python main.py --data_in data --data_set otter --data_set_ext JPG --data_out out --data_k K.txt --data_d D.txt --show_plots --color_mode rgb
"""


import argparse
import glob
import os
import random

import cv2
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def load_K(file_path: str) -> np.array:
    """
    Reads the Camera intrinsic parameters from the file

    Returns
    -------
    np.array:
        Camera intrinsic parameters
    """

    with open(file_path) as f:
        K = np.array(
            list(
                (
                    map(
                        lambda x: list(map(lambda x: float(x), x.strip().split(" "))),
                        f.read().split(None),
                    )
                )
            )
        ).reshape(3, 3)
    return K


def load_D(file_path: str) -> np.array:
    """
    Reads the Camera distortion parameters from the file
    If the file does not exist, it returns an array of zeros

    Returns
    -------
    np.array:
        Camera distortion parameters
    """

    if not os.path.exists(file_path):
        print(f"Warning: No distortion file found at {file_path}")
        return np.zeros((1, 5), dtype=np.float32)

    with open(file_path) as f:
        D = np.array(
            list(
                (
                    map(
                        lambda x: list(map(lambda x: float(x), x.strip().split(" "))),
                        f.read().split(None),
                    )
                )
            )
        ).reshape(1, 5)
    return D


def load_images(
    data_path: str, data_set: str, data_set_ext: str, color_mode: str
) -> tuple[list, list]:
    """
    Reads the set of images from the file

    Parameters
    ----------
    data_path : str
        Path to the data directory
    data_set : str
        Name of the dataset
    data_set_ext : str
        File extension of the images
    color_mode : str
        Color mode to use ('bgr' or 'rgb')

    Returns
    -------
    list:
        List of images
    list:
        List of image paths
    """

    images = []
    image_paths = []

    image_paths = sorted(
        glob.glob(os.path.join(data_path, data_set, f"*.{data_set_ext}"))
    )

    if data_set == "globe":
        image_paths = image_paths[31:15:-1] + image_paths[:16]

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_mode.lower() == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images, image_paths


def compute_features(images: list) -> tuple:
    """
    Computes the features from the images

    Returns
    -------
    list:
        List of keypoints
    list:
        List of descriptors
    """

    sift = cv2.SIFT_create()

    keypoints = [None] * len(images)
    descriptors = [None] * len(images)

    for i, image in enumerate(images):
        keypoints[i], descriptors[i] = sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
        )
        print(f"Image {i}: {len(keypoints[i])} keypoints")

    return keypoints, descriptors


def compute_matches(keypoints: list, descriptors: list) -> list:
    """
    Computes the matches between the keypoints and descriptors

    Returns
    -------
    dict:
        Dict of matches
    """

    matcher = cv2.BFMatcher()
    matches = {}

    for i in range(len(keypoints) - 1):
        # Lowe's ratio test
        res = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)
        features = [m for m, n in res if m.distance < 0.70 * n.distance]

        matches[(i, i + 1)] = (
            np.float32([keypoints[i][m.queryIdx].pt for m in features]),
            np.float32(
                [keypoints[i + 1][m.trainIdx].pt for m in features],
            ),
        )

        print(f"Matches {i}-{i+1}: {len(features)}")

    return matches


def compute_shared(points_1, points_2, points_3):
    """
    Finds shared points between points_1 and points_2, and returns the corresponding points in points_3
    Returns:
        indices_1: Indices in points_1 that are common with points_2
        indices_2: Corresponding indices in points_2
        mask_points_2: points_2 excluding shared points
        mask_points_3: points_3 excluding corresponding shared points
    """
    _, id_1, id_2 = np.intersect1d(
        points_1.view([("", points_1.dtype)] * 2),
        points_2.view([("", points_2.dtype)] * 2),
        return_indices=True,
    )

    # Remove the matching entries from points_2 and points_3
    mask = np.ones(len(points_2), dtype=bool)
    mask[id_2] = False
    mask_points_2 = points_2[mask]
    mask_points_3 = points_3[mask]
    return id_1, id_2, mask_points_2, mask_points_3


def save_points(file_path: str, points: np.array, colors: np.array) -> None:
    """
    Saves the 3D points to a file

    Parameters
    ----------
    file_path : str
        File path
    points : np.array
        3D points
    colors : np.array
        Colors

    Returns
    -------
    o3d.geometry.PointCloud:
        Point cloud
    """

    # Remove outliers using clustering
    clustering = DBSCAN(eps=0.25, min_samples=10).fit(points)
    labels = clustering.labels_
    mask = labels != -1
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(filtered_points)
    cloud.colors = o3d.utility.Vector3dVector(filtered_colors / 255)

    o3d.io.write_point_cloud(os.path.join(file_path), cloud, write_ascii=True)

    return cloud


def create_camera_mesh(position, rotation, scale=0.25):
    """
    Creates a triangle mesh box representing the camera at the given position and rotation.

    Parameters
    ----------
    position : np.array
        Camera position (3D coordinates).
    rotation : np.array
        Camera rotation matrix (3x3).
    scale : float
        Scale of the camera mesh.

    Returns
    -------
    o3d.geometry.TriangleMesh:
        Triangle mesh box representing the camera.
    """
    camera_mesh = o3d.geometry.TriangleMesh.create_box(
        width=2 * scale, height=scale, depth=0.1 * scale
    )
    camera_mesh.compute_vertex_normals()
    camera_mesh.paint_uniform_color([1, 0, 0])

    camera_mesh.rotate(rotation.T, center=(0, 0, 0))
    camera_mesh.translate(-rotation.T @ position)

    return camera_mesh


def compute_camera_pose(
    features_i: np.array, features_j: np.array, K: np.array, E: np.array
) -> tuple:
    """
    Computes the camera pose using the essential matrix.
    Parameters
    ----------
    features_i : np.array
        Features from the first image.
    features_j : np.array
        Features from the second image.
    K : np.array
        Camera intrinsic matrix.
    E : np.array
        Essential matrix.
    Returns
    -------
    tuple:
        Tuple containing the rotation and translation vectors.
    """

    R1, R2, t = cv2.decomposeEssentialMat(E)

    P_i = np.hstack((np.eye(3), np.zeros((3, 1))))
    P_js = [
        np.hstack((R1, t)),
        np.hstack((R1, -t)),
        np.hstack((R2, t)),
        np.hstack((R2, -t)),
    ]

    P_j = None
    for P_j_k in P_js:
        points_4d = cv2.triangulatePoints(P_i, P_j_k, features_i.T, features_j.T)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)
        if np.all(points_3d[:, 2] > 0):
            P_j = P_j_k
            break
    if P_j is None:
        raise ValueError("No valid camera pose found.")

    return P_j[:3, :3], P_j[:3, 3].reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="Structure from Motion Pipeline")
    parser.add_argument(
        "--data_in", type=str, default="data", help="Input data directory"
    )
    parser.add_argument("--data_set", type=str, default="otter", help="Dataset name")
    parser.add_argument(
        "--data_set_ext", type=str, default="JPG", help="Dataset file extension"
    )
    parser.add_argument("--data_out", type=str, default="out", help="Output directory")
    parser.add_argument(
        "--data_k", type=str, default="K.txt", help="Camera intrinsic file"
    )
    parser.add_argument(
        "--data_d",
        type=str,
        default="D.txt",
        help="Camera distortion file",
        required=False,
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Display matplotlib plots"
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        default="rgb",
        choices=["bgr", "rgb"],
        help="Color mode to use (bgr or rgb)",
    )
    args = parser.parse_args()

    # Set paths
    data_in = args.data_in
    data_set = args.data_set
    data_set_ext = args.data_set_ext
    data_out = args.data_out
    data_k = args.data_k
    data_d = args.data_d
    color_mode = args.color_mode

    os.makedirs(data_out, exist_ok=True)

    K = load_K(os.path.join(data_in, data_set, data_k))
    D = load_D(os.path.join(data_in, data_set, data_d))
    images, _ = load_images(data_in, data_set, data_set_ext, color_mode)

    if args.show_plots and len(images) >= 2 and D is not None and np.any(D):
        idx_samples = random.sample(range(len(images)), 2)
        _, axs = plt.subplots(2, 2, figsize=(10, 8))

        for i, idx in enumerate(idx_samples):
            img = images[idx]
            undistorted = cv2.undistort(img, K, D)

            axs[i][0].imshow(img)
            axs[i][0].set_title(f"Original Image {idx}")
            axs[i][0].axis("off")

            axs[i][1].imshow(undistorted)
            axs[i][1].set_title(f"Undistorted Image {idx}")
            axs[i][1].axis("off")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------
    # Step 1: Feature Extraction and Matching
    # ---------------------------------------

    keypoints, descriptors = compute_features(images)
    matches = compute_matches(keypoints, descriptors)

    if args.show_plots:
        image_keypoints = cv2.drawKeypoints(
            images[0],
            keypoints[0],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        plt.figure(figsize=(15, 8))
        plt.title("Keypoints on First Image")
        plt.imshow(image_keypoints)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    id_i, id_j = 0, 1
    features_i, features_j = matches[(id_i, id_j)]

    if args.show_plots:
        image_matches = cv2.drawMatches(
            images[id_i],
            keypoints[id_i],
            images[id_j],
            keypoints[id_j],
            [cv2.DMatch(i, i, 0) for i in range(len(features_i))],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )

        plt.figure(figsize=(15, 8))
        plt.title("Feature Matches Between First Two Images")
        plt.imshow(image_matches)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------
    # Step 2: Essential Matrix Estimation
    # ---------------------------------------

    E, mask = cv2.findEssentialMat(
        features_i,
        features_j,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.5,
    )

    features_i = features_i[mask.ravel() == 1]
    features_j = features_j[mask.ravel() == 1]

    # ---------------------------------------
    # Step 3: Camera Pose Estimation
    # ---------------------------------------

    _, R, t, mask = cv2.recoverPose(E, features_i, features_j, K)
    features_i = features_i[mask.ravel() > 0]
    features_j = features_j[mask.ravel() > 0]

    # Double check the result
    R_prime, t_prime = compute_camera_pose(features_i, features_j, K, E)
    assert np.allclose(R, R_prime)
    assert np.allclose(t, t_prime)

    P_i = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_j = K @ np.hstack((R, t))

    camera_meshes = []
    camera_meshes.append(create_camera_mesh(np.zeros(3), np.eye(3)))  # First camera
    camera_meshes.append(create_camera_mesh(t.ravel(), R))  # Second camera

    # ---------------------------------------
    # Step 4: Triangulation
    # ---------------------------------------

    points = cv2.triangulatePoints(P_i, P_j, features_i.T, features_j.T)
    points = cv2.convertPointsFromHomogeneous(points.T)
    points = points.reshape(-1, 3)

    points = points[:, :3]

    _, rodrigues_vector, t, mask = cv2.solvePnPRansac(
        points, features_j, K, None, cv2.SOLVEPNP_ITERATIVE
    )
    R, _ = cv2.Rodrigues(rodrigues_vector)

    if mask is not None:
        features_i = features_i[mask[:, 0]]
        points = points[mask[:, 0]]
        features_j = features_j[mask[:, 0]]

    camera_meshes.append(create_camera_mesh(t.ravel(), R))

    points_cumulative = points
    colors_cumulative = np.array(
        [images[id_j][x, y] for (y, x) in np.array(features_j, dtype=np.int32)]
    )

    os.makedirs(os.path.join(data_out, f"{data_set}"), exist_ok=True)
    cloud = save_points(
        os.path.join(data_out, data_set, f"matching-2.ply"),
        points_cumulative,
        colors_cumulative,
    )

    if args.show_plots:
        o3d.visualization.draw_geometries([cloud, *camera_meshes])

    # ---------------------------------------
    # Step 5: Structure from Motion
    # ---------------------------------------

    features_prev = features_j
    for id_k in range(2, len(images)):
        print(f"Processing image {id_k+1}")
        features_j, features_k = matches[(id_j, id_k)]

        shared_id_prev, shared_id_j, shared_j, shared_k = compute_shared(
            features_prev, features_j, features_k
        )
        points_shared_k = features_k[shared_id_j]

        points = points[shared_id_prev]
        _, rodrigues_vector, t, _ = cv2.solvePnPRansac(
            points, points_shared_k, K, None, cv2.SOLVEPNP_ITERATIVE
        )
        R, _ = cv2.Rodrigues(rodrigues_vector)

        P_k = K @ np.hstack((R, t))

        camera_meshes.append(create_camera_mesh(t.ravel(), R))

        points = cv2.triangulatePoints(P_j, P_k, shared_j.T, shared_k.T)
        points = cv2.convertPointsFromHomogeneous(points.T).reshape(-1, 3)
        points_cumulative = np.vstack((points_cumulative, points))

        image_k = images[id_k]
        colors = np.array(
            [image_k[y, x] for (x, y) in np.array(shared_k, dtype=np.int32)]
        )
        colors_cumulative = np.vstack((colors_cumulative, colors))

        id_j = id_k
        features_i = features_j
        features_prev = features_k
        P_i = P_j
        P_j = P_k

        os.makedirs(os.path.join(data_out, f"{data_set}"), exist_ok=True)
        cloud = save_points(
            os.path.join(data_out, data_set, f"matching-{id_k+1}.ply"),
            points_cumulative,
            colors_cumulative,
        )

        if id_k < len(images) - 1:
            points = cv2.triangulatePoints(P_i, P_j, features_i.T, features_prev.T)
            points = cv2.convertPointsFromHomogeneous(points.T).reshape(-1, 3)

    # ---------------------------------------
    # Step 6: Point Cloud Visualization
    # ---------------------------------------

    if args.show_plots:
        o3d.visualization.draw_geometries([cloud, *camera_meshes])


if __name__ == "__main__":
    main()
