# 2026.3.2 匀速直线运动的小球，实现了对状态的简单卡尔曼滤波，探索了R矩阵对滤波效果的影响
# 2026.3.3 匀加速直线运动的小球，实现了交互式多模型加权实现状态融合
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
mu = np.array([0.5, 0.5])
mu_history = []
# 定义变量
position = [0,0]
velocity = [1,0.5]
traj = []
trajx = []
trajy = []
obsx = []
obsy = []
obs=[]
fused = []

filtered0x = []
filtered0y = []
filtered1x = []
filtered1y = []
fusedx = []
x1 = np.array([0, 0, 1, 0.5,0.5,0.5])  # 初始状态
P1 = np.eye(6)                  # 初始协方差
Q1 = np.eye(6) * 0.1
R1 = np.eye(2) * 25    # 2×2单位矩阵×1
F1 = np.array([[1,0,1,0,0.5,0],[0,1,0,1,0,0.5],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]])
H1 = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])

x0 = np.array([0, 0, 1, 0.5])  # 初始状态

P0 = np.eye(4)                  # 初始协方差
Q0 = np.eye(4) * 0.1  # 4×4单位矩阵×0.1
R0 = np.eye(2) * 25    # 2×2单位矩阵×1
F0 = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
H0 = np.array([[1,0,0,0],[0,1,0,0]])
filtered0= []
filtered1= []
# 真实轨迹
for i in range(100):
    if i < 50:
        acceleration = [0, 0]  # 前50步匀速
    else:
        acceleration = [0.5, 0.5]  # 后50步匀加速
    position[0] = position[0]+velocity[0]+ 0.5*acceleration[0]
    position[1] = position[1]+velocity[1]+ 0.5*acceleration[1]
    velocity[0] = velocity[0] + acceleration[0]
    velocity[1] = velocity[1] + acceleration[1]
    traj.append([position[0],position[1]])
#观测信息
for i in range(100):
    noise_x = np.random.normal(0, 5)
    noise_y = np.random.normal(0, 5)
    position[0] = traj[i][0] + noise_x
    position[1] = traj[i][1] + noise_y
    obs.append([position[0],position[1]])

for i in range(100):
    # 交互
    # x0_expanded = np.append(x0, [0, 0])
    # x_mixed = mu[0]*x0_expanded + mu[1]*x1
    # x0 = x_mixed[:4]  # 取前4维给x0
    # x1 = x_mixed      # 全6维给x1
    # 0 卡尔曼滤波
    # 预测
    x_pred = F0@x0
    P_pred = F0@P0@F0.T+Q0
    # 更新
    K = P_pred @ H0.T @ np.linalg.inv(H0 @ P_pred @ H0.T + R0) # 计算卡尔曼增益，括号里是新息协方差
    x0 = x_pred + K @ (np.array(obs[i]) - H0 @ x_pred) #括号里是新息
    x0_expanded = np.append(x0, [0, 0])
    v0 = np.array(obs[i]) - H0 @ x_pred
    S0 = H0 @ P_pred @ H0.T + R0
    P0 = (np.eye(4) - K @ H0) @ P_pred #状态协方差矩阵
    filtered0.append(x0)
    
    # 1卡尔曼滤波
    # 预测
    x_pred = F1@x1
    P_pred = F1@P1@F1.T+Q1
    # 更新
    K = P_pred @ H1.T @ np.linalg.inv(H1 @ P_pred @ H1.T + R1) # 计算卡尔曼增益
    x1 = x_pred + K @ (np.array(obs[i]) - H1 @ x_pred) #新息
    v1 = np.array(obs[i]) - H1 @ x_pred
    S1 = H1 @ P_pred @ H1.T + R1
    P1 = (np.eye(6) - K @ H1) @ P_pred #状态协方差矩阵
    filtered1.append(x1)

    L0 = multivariate_normal.pdf(v0, mean=np.zeros(2), cov=S0)
    L1 = multivariate_normal.pdf(v1, mean=np.zeros(2), cov=S1)
    mu[0] = L0 * mu[0]
    mu[1] = L1 * mu[1]
    mu = mu / mu.sum()  # 归一化
    mu = np.clip(mu, 0.01, 0.99)
    x_fused = mu[0] * x0_expanded + mu[1] * x1
    mu_history.append(mu.copy())
    fused.append(x_fused)
    

for i in range(100):
    trajx.append(traj[i][0])
    trajy.append(traj[i][1])
    obsx.append(obs[i][0])
    obsy.append(obs[i][1])
    filtered0x.append(filtered0[i][0])
    filtered0y.append(filtered0[i][1])
    filtered1x.append(filtered1[i][0])
    filtered1y.append(filtered1[i][1])
    fusedx.append(fused[i][0])
rmse0 = np.sqrt(np.mean((np.array(filtered0x) - np.array(trajx))**2))
rmse1 = np.sqrt(np.mean((np.array(filtered1x) - np.array(trajx))**2))
rmsefused = np.sqrt(np.mean((np.array(fusedx) - np.array(trajx))**2))
print(rmse0)
print(rmse1)
print(rmsefused)
plt.plot(mu_history, label=['CV', 'CA'])
plt.legend()
plt.show()
