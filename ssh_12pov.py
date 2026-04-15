"""
Generate 12 POV images from a 360 equirectangular panorama:

4 views at pitch 0°   : front, right, back, left
4 views at pitch +30° : front_up30, right_up30, back_up30, left_up30
4 views at pitch -30° : front_down30, right_down30, back_down30, left_down30

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import math


def equirectangular_to_perspective(img, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    h, w = img.shape[:2]

    # Convert to radians
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Focal length
    f = (out_w / 2) / math.tan(fov / 2)

    # Output pixel grid
    xs = np.linspace(-out_w / 2, out_w / 2, out_w)
    ys = np.linspace(-out_h / 2, out_h / 2, out_h)
    x_grid, y_grid = np.meshgrid(xs, ys)

    z = f * np.ones_like(x_grid)
    x = x_grid
    y = -y_grid

    # Normalize direction vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    # Pitch rotation
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    y2 = y * cp - z * sp
    z2 = y * sp + z * cp

    # Yaw rotation
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x3 = x * cy + z2 * sy
    z3 = -x * sy + z2 * cy
    y3 = y2

    # Convert to spherical coordinates
    lon = np.arctan2(x3, z3)
    lat = np.arcsin(y3)

    # Map to equirectangular coordinates
    u = (lon + math.pi) / (2 * math.pi) * (w - 1)
    v = (math.pi / 2 - lat) / math.pi * (h - 1)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )


def generate_12_pov(input_path, output_dir,
                    fov_deg=90,
                    out_w=800,
                    out_h=600):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    # Yaw directions
    yaws = {
        "front": 0,
        "right": 90,
        "back": 180,
        "left": -90
    }

    # Pitch levels
    pitches = {
        "up30": 30,
        "mid": 0,
        "down30": -30
    }

    for pitch_name, pitch_val in pitches.items():
        for yaw_name, yaw_val in yaws.items():

            pov = equirectangular_to_perspective(
                img,
                fov_deg,
                yaw_val,
                pitch_val,
                out_w,
                out_h
            )

            if pitch_name == "mid":
                filename = f"{yaw_name}.png"
            else:
                filename = f"{yaw_name}_{pitch_name}.png"

            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, pov)

            print(f"Saved: {filename}")


if __name__ == "__main__":
    input_image = "input_360.jpg"   # ← change this
    output_folder = "pov_12_outputs"

    generate_12_pov(
        input_image,
        output_folder,
        fov_deg=90,
        out_w=800,
        out_h=600
    )
