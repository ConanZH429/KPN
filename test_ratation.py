import numpy as np
from scipy.spatial.transform import Rotation as R



# 旋转向量
r_vect = R.from_rotvec(45 * np.array([0, 0, 1]), degrees=True)      # 绕Z轴旋转45度
print(r_vect.as_matrix())
r_euler = R.from_euler("yxz", [0, 0, 45], degrees=True)      # 绕Z轴旋转45度
print(r_euler.as_matrix())