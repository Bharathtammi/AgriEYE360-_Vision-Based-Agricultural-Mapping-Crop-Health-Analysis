"""
Split a 360 equirectangular image into 4 POV images
(front, right, back, left) using OpenCV and NumPy.

Requirements:
  pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import math


def equirectangular_to_perspective(img, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    """
    Convert equirectangular panorama to a single perspective view.

    img       : input equirectangular image (H x W x 3, BGR)
    fov_deg   : horizontal field of view in degrees (e.g. 90)
    yaw_deg   : left-right angle, 0 = front, 90 = right, 180 = back, -90 = left
    pitch_deg : up-down angle, 0 = horizon, positive = look up, negative = look down
    out_w     : output width
    out_h     : output height
    """
    h, w = img.shape[:2]

    # Convert degrees to radians
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Focal length based on fov and output width
    f = (out_w / 2) / math.tan(fov / 2)

    # Build pixel grid in output image
    xs = np.linspace(-out_w / 2, out_w / 2, out_w)
    ys = np.linspace(-out_h / 2, out_h / 2, out_h)
    x_grid, y_grid = np.meshgrid(xs, ys)

    # Direction in camera space
    z = f * np.ones_like(x_grid)
    x = x_grid
    y = -y_grid  # flip y so that positive is up

    # Normalize direction vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    # Rotate by pitch (around x axis) then yaw (around y axis)
    # Pitch
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    y2 = y * cp - z * sp
    z2 = y * sp + z * cp

    # Yaw
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x3 = x * cy + z2 * sy
    z3 = -x * sy + z2 * cy
    y3 = y2

    # Convert to spherical coordinates
    lon = np.arctan2(x3, z3)          # -pi to pi
    lat = np.arcsin(y3)               # -pi/2 to pi/2

    # Map lon/lat to equirectangular pixel coordinates
    u = (lon + math.pi) / (2 * math.pi) * (w - 1)
    v = (math.pi / 2 - lat) / math.pi * (h - 1)

    # Remap
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    perspective = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)

    return perspective


def split_360_into_4_pov(input_path, output_dir,
                         fov_deg=90, out_w=800, out_h=600, pitch_deg=0):
    """
    Load a 360 equirectangular image and generate 4 POV images:
    front, right, back, left.

    input_path : path to 360 image (e.g. .jpg or .png)
    output_dir : folder where output images will be saved
    fov_deg    : horizontal field of view for each POV
    out_w/h    : output resolution
    pitch_deg  : vertical angle for all POVs (0 = horizon)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Define yaw angles for 4 POVs
    views = {
        "front": 0,
        "right": 90,
        "back": 180,
        "left": -90
    }

    for name, yaw in views.items():
        pov_img = equirectangular_to_perspective(
            img,
            fov_deg=fov_deg,
            yaw_deg=yaw,
            pitch_deg=pitch_deg,
            out_w=out_w,
            out_h=out_h
        )
        out_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(out_path, pov_img)
        print(f"Saved {name} to {out_path}")


if __name__ == "__main__":
    # Example usage
    input_360 = "input_360.jpg"   # change to your 360 image path
    output_folder = "pov_outputs" # folder for 4 POV images

    split_360_into_4_pov(
        input_path=input_360,
        output_dir=output_folder,
        fov_deg=90,   # 90° horizontal FOV for each POV
        out_w=800,
        out_h=600,
        pitch_deg=0   # looking at horizon; change if you want to look up/down
    )

