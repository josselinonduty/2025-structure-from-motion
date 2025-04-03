# SfM

## Requirements

Tested against Python 3.12

```bash
pip install matplotlib
pip install opencv-python
pip install open3d
```

## Available Datasets

- `otter`: A dataset containing images of a plush otter toy.
- `globe`: A dataset containing images of a globe.

## Usage

### Camera Calibration (`calibrate.py`)

This script performs camera calibration using a checkerboard pattern. It calculates the camera matrix and distortion coefficients and saves them to files.

#### Examples

```bash
python calibrate.py
```

```bash
python calibrate.py --board 6,8 --data_in data --data_set otter --data_set_ext JPG --data_out data/otter
```

> The script will look for calibration images in the `$data_in/$data_set/calibration` directory.

#### Arguments:

- `--board`: Checkerboard dimensions (default: `6,8`).
- `--data_in`: Input data directory (default: `data`).
- `--data_set`: Dataset name or subdirectory (default: `otter`).
- `--data_set_ext`: Dataset file extension (default: `JPG`).
- `--data_out`: Output directory (default: `data/otter`).

> (6,8) should yield the same results as (8,6).

### Structure from Motion (`main.py`)

This script generates a 3D point cloud from a set of images using Structure from Motion (SfM) techniques.

#### Examples

```bash
python main.py
```

```bash
python main.py --data_in data --data_set globe --data_set_ext JPG --data_out out --data_k K.txt --data_d D.txt --show_plots --color_mode rgb
```

#### Arguments:

- `--data_in`: Input data directory (default: `data`).
- `--data_set`: Dataset name (default: `otter`).
- `--data_set_ext`: Dataset file extension (default: `JPG`).
- `--data_out`: Output directory (default: `out`).
- `--data_k`: Camera intrinsic file (default: `K.txt`).
- `--data_d`: Camera distortion file (default: `D.txt`).
- `--show_plots`: Display matplotlib plots (optional).
- `--color_mode`: Color mode to use (`bgr` or `rgb`, default: `rgb`).

## Sources

[1]
J. Cohen, “Photogrammetry Explained: From Multi-View Stereo to Structure from Motion - PyImageSearch,” PyImageSearch, Oct. 14, 2024. https://pyimagesearch.com/2024/10/14/photogrammetry-explained-from-multi-view-stereo-to-structure-from-motion/ (accessed Mar. 19, 2025).
‌

[2]
“Lecture 16: Structure from Motion.” Accessed: Mar. 19, 2025. [Online]. Available: https://www.cs.unc.edu/~ronisen/teaching/spring_2023/web_materials/lecture_16_sfm.pdf
‌
