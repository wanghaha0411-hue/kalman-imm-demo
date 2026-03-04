# 2026.03.04 匀速直线运动小球 学习EKF滤波
# 两个缺点：
# 1.EKF用一阶泰勒展开近似非线性函数，当非线性程度很强的时候，一阶近似误差很大，滤波结果会不准甚至发散。
# 2.每换一个系统，都要重新推导h(x)和f(x)的偏导数，复杂系统很容易推错，工程上很麻烦。
# 相应的，UKF诞生：不做线性化，用一组采样点来实现非线性传播，对比来看，对强非线性系统效果好于EKF
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
P = np.eye(4)                  # 初始状态估计误差协方差矩阵
Q = np.eye(4) * 0.1  # 4×4单位矩阵×0.1
R = np.array([[1,0],[0,0.01]])    # 2×2单位矩阵×1
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
    noisedis = np.random.normal(0,1)
    noiseang = np.random.normal(0,0.01)
    position[0] = traj[i][0]+noisedis
    position[1] = traj[i][1]+noisedis
    distance = np.sqrt(traj[i][0]**2+traj[i][1]**2)
    angle = np.arctan2(traj[i][1],traj[i][0])
    obs.append([distance+noisedis,angle+noiseang])
# 滤波
for i in range(100):
    x_pred = F@x
    P_pred = F@P@F.T+Q
    px,py = x_pred[0],x_pred[1]
    dist = np.sqrt(px**2+py**2)
    z_pred = np.array([dist,np.arctan2(py,px)])
    H = np.array([[px/dist,py/dist,0,0],[-py/dist**2,px/dist**2,0,0]])

    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    #新息计算
    innovation = np.array(obs[i])-z_pred

    x = x_pred + K @ innovation
    P = (np.eye(4) - K @ H) @ P_pred
    filtered.append(x)
# 准备绘图
for i in range(100):
    trajx.append(traj[i][0])
    trajy.append(traj[i][1])
    
    filteredx.append(filtered[i][0])
    filteredy.append(filtered[i][1])
plt.plot(trajx,trajy)
plt.plot(filteredx,filteredy)
plt.legend()
plt.show()
