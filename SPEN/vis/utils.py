import cv2 as cv
import numpy as np

def show_image(image: np.ndarray, winname: str = "Image") -> None:
    """
    Display the image.

    Args:
        image (np.ndarray): The image to display.
        winname (str): The name of the window. Default: "Image".

    Returns:
        None
    """
    cv.imshow(winname, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None