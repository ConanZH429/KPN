import numpy as np


class Camera:
    def __init__(self):
        self.K: np.ndarray
        self.K_inv: np.ndarray


class SPEEDCamera(Camera):
    def __init__(self, shape):
        super().__init__()
        self.fwx = 0.0176  # focal length[m]
        self.fwy = 0.0176  # focal length[m]
        self.width = 1920  # number of horizontal[pixels]
        self.height = 1200  # number of vertical[pixels]
        self.shape = shape
        self.scale = self.height / shape[0]
        self.ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
        self.ppy = self.ppx  # vertical pixel pitch[m / pixel]
        self.fx = self.fwx / self.ppx  # horizontal focal length[pixels]
        self.fy = self.fwy / self.ppy  # vertical focal length[pixels]
        self.K = np.array([[self.fx, 0, self.width / 2], [0, self.fy, self.height / 2], [0, 0, 1]])
        self.K[:2] = self.K[:2] / self.scale
        self.K_inv = np.linalg.inv(self.K)
        # self.K_label = np.array([[self.fx, 0, self.width / 2], [0, self.fy, self.height / 2], [0, 0, 1]])
        # self.K_label_inv = np.linalg.inv(self.K_label)
        # self.K_image = np.array([[self.fx, 0, self.width / 2], [0, self.fy, self.height / 2], [0, 0, 1]])
        # self.K_image[:2] = self.K_image[:2] / self.scale
        # self.K_image_inv = np.linalg.inv(self.K_image)

class SPEEDplusCamera(Camera):
    def __init__(self, shape):
        super().__init__()
        self.height = 1200
        self.width = 1920
        self.scale = self.height / shape[0]
        self.K = np.array([
            [2988.5795163815555, 0, 960.0],
            [0, 2988.3401159176124, 600.0],
            [0, 0, 1]
        ])
        self.K[:2] = self.K[:2] / self.scale
        self.K_inv = np.linalg.inv(self.K)
        # self.K_label = np.array([
        #     [2988.5795163815555, 0, 960.0],
        #     [0, 2988.3401159176124, 600.0],
        #     [0, 0, 1]
        # ])
        # self.K_label_inv = np.linalg.inv(self.K_label)
        # self.K_image = np.array([
        #     [2988.5795163815555, 0, 960.0],
        #     [0, 2988.3401159176124, 600.0],
        #     [0, 0, 1]
        # ])
        # self.K_image[:2] = self.K_image[:2] / self.scale
        # self.K_image_inv = np.linalg.inv(self.K_image)