import matplotlib.pyplot as plt
import numpy as np
position = [0,0]
velocity = [1,0.5]
traj = []
trajx = []
trajy = []
obsx = []
obsy = []
obs=[]
filteredx = []
filteredy = []
x = np.array([0, 0, 1, 0.5])  # 初始状态
P = np.eye(4)                  # 初始协方差
Q = np.eye(4) * 0.1  # 4×4单位矩阵×0.1
R = np.eye(2) * 1    # 2×2单位矩阵×1
F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
H = np.array([[1,0,0,0],[0,1,0,0]])
filtered= []
#真实
for i in range(100):
    position[0] = position[0]+velocity[0]
    position[1] = position[1]+velocity[1]
    traj.append([position[0],position[1]])
# 观测
for i in range(100):
    
    position[0] = traj[i][0]+np.random.normal(0,1)
    position[1] = traj[i][1]+np.random.normal(0,1)
    
    obs.append([position[0],position[1]])
# 滤波
for i in range(100):
    x_pred = F@x
    P_pred = F@P@F.T+Q
    px,py = x_pred[0],x_pred[1]
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x = x_pred + K @ (np.array(obs[i]) - H @ x_pred)
    P = (np.eye(4) - K @ H) @ P_pred
    filtered.append(x)
# 准备绘图
for i in range(100):
    trajx.append(traj[i][0])
    trajy.append(traj[i][1])
    obsx.append(obs[i][0])
    obsy.append(obs[i][1])
    filteredx.append(filtered[i][0])
    filteredy.append(filtered[i][1])
plt.plot(trajx, trajy, label='True Trajectory', color='green', linewidth=2)
plt.scatter(obsx, obsy, label='Noisy Observations', color='red', s=10, alpha=0.5)
plt.plot(filteredx, filteredy, label='Kalman Filter', color='blue', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
