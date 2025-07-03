import random
import math
import cv2 as cv
import numpy as np
from albucore import add_weighted

def get_circles(num_circles: int,
                flare_center_x: int,
                flare_center_y: int,
                height: int,
                width: int,
                angle_range: list[float, float],
                src_color: int,):
    diagonal = math.sqrt(height**2 + width**2)
    angle = 2 * math.pi * random.uniform(angle_range[0], angle_range[1])

    step_size = max(1, int(diagonal * 0.01))
    max_radius = max(2, int(height * 0.01))
    color_range = int(src_color * 0.2)
    # Generate points along the flare line
    t_range = range(-flare_center_x, width - flare_center_x, step_size)
    def line(t: float) -> tuple[float, float]:
        return (
            flare_center_x + t * math.cos(angle),
            flare_center_y + t * math.sin(angle),
        )
    
    # Generate points along the flare line
    points = [line(t) for t in t_range]

    circles = []
    for i in range(num_circles):
        alpha = random.uniform(0.05, 0.2)
        point = points[random.randint(0, len(points) - 1)]
        rad = random.randint(1, max_radius)

        colors = random.randint(src_color - color_range, src_color)
        
        circles.append(
            (
                alpha,
                (int(point[0]), int(point[1])),
                pow(rad, 3),
                colors,
            )
        )
    return circles

def add_sun_flare_overlay(img: np.ndarray,
                          flare_center: tuple[int, int],
                          src_radius: int,
                          src_color: int,
                          circles):
    overlay = img.copy()
    output = img.copy()

    weighted_brightness = 0.0
    total_radius_length = 0.0

    for alpha, (x, y), rad3, circle_color in circles:
        weighted_brightness += alpha * rad3
        total_radius_length += rad3
        cv.circle(overlay, (x, y), rad3, circle_color, -1)
        output = cv.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    point = [int(x) for x in flare_center]

    overlay = output.copy()
    num_times = src_radius // 10

    # max_alpha is calculated using weighted_brightness and total_radii_length times 5
    # meaning the higher the alpha with larger area, the brighter the bright spot will be
    # for list of alphas in range [0.05, 0.2], the max_alpha should below 1
    max_alpha = weighted_brightness / total_radius_length * 5
    alpha = np.linspace(0.0, min(max_alpha, 1.0), num=num_times)

    rad = np.linspace(1, src_radius, num=num_times)

    for i in range(num_times):
        cv.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        output = add_weighted(overlay, alp, output, 1 - alp)
        
    return output


def add_sun_flare_overlayv2(img: np.ndarray,
                            flare_center: tuple[int, int],
                            src_radius: int,
                            src_color: int):
    overlay = img.copy()
    output = img.copy()

    weighted_brightness = 0.0
    total_radius_length = 0.0

    point = [int(x) for x in flare_center]

    overlay = output.copy()
    num_times = src_radius // 10

    # max_alpha is calculated using weighted_brightness and total_radii_length times 5
    # meaning the higher the alpha with larger area, the brighter the bright spot will be
    # for list of alphas in range [0.05, 0.2], the max_alpha should below 1
    alpha = np.linspace(0.0, random.uniform(0.5, 0.9), num=num_times)

    rad = np.linspace(1, src_radius, num=num_times)

    for i in range(num_times):
        cv.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1]**3
        output = add_weighted(overlay, alp, output, 1 - alp)
        
    return output

def add_sun_flare_physics_v2(img: np.ndarray,
                             flare_center: tuple[int, int],
                             src_radius: int,
                             src_color: int):

    output = img.copy().astype(np.float32)
    height, width = img.shape[:2]

    # Create a separate flare layer
    flare_layer = np.zeros_like(img, dtype=np.float32)

    # Add the main sun
    cv.circle(flare_layer, flare_center, src_radius, src_color, -1)

    # Add lens diffraction spikes
    # for angle in [0, 45, 90, 135]:
    #     end_point = (
    #         int(flare_center[0] + np.cos(np.radians(angle)) * max(width, height)),
    #         int(flare_center[1] + np.sin(np.radians(angle)) * max(width, height)),
    #     )
    #     cv.line(flare_layer, flare_center, end_point, src_color, 2)

    # Apply gaussian blur to soften the flare
    flare_layer = cv.GaussianBlur(flare_layer, (9, 9), sigmaX=15, sigmaY=15)

    # Create a radial gradient mask
    y, x = np.ogrid[:height, :width]
    mask = np.sqrt((x - flare_center[0]) ** 2 + (y - flare_center[1]) ** 2)
    mask = 1 - np.clip(mask / (max(width, height) * 0.7), 0, 1)

    # Apply the mask to the flare layer
    flare_layer *= mask

    output = 255 - ((255 - output) * (255 - flare_layer) / 255)
    # output = cv.cvtColor(output, cv.COLOR_RGB2GRAY)
    return output.astype(np.uint8)

def sun_flare(image: np.ndarray,
              flare_center: tuple[int, int],
              src_radius: int = 500,
              src_color = 255,
              angle_range: tuple[float, float] = (0, 1),
              num_circles: int = 10,):
    height, width = image.shape[:2]

    image = add_sun_flare_physics_v2(
        img=image,
        flare_center=flare_center,
        src_radius=src_radius,
        src_color=src_color
    )
    return image