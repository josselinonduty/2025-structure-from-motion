import cv2
import numpy as np
import glob

# Define checkerboard size (number of inner corners per row and column)
CHECKERBOARD = (6, 8)  # Adjust to match your pattern

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load images
images = glob.glob("calibration/*.JPG")  # Adjust folder path if needed

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)

# Save results as txt files (matrix, no brackets, spaces as delimiters, new lines)
np.savetxt("calibration/K.txt", mtx, delimiter=" ")
np.savetxt("calibration/distortion.txt", dist, delimiter=" ")
