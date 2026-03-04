#2026.03.04 今天学习了UKF滤波
import matplotlib.pyplot as plt
import numpy as np
n = 4  # 状态维度
alpha = 0.001
beta = 2
kappa = 0
lambda_ = alpha**2 * (n + kappa) - n
F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
Wm = [lambda_/(n+lambda_)]
Wc = [lambda_/(n+lambda_)+1-alpha**2+beta]

for i in range(2*n):
    Wm.append(1/(2*(n+lambda_)))
    Wc.append(1/(2*(n+lambda_)))

#生成sigma点
def generate_sigma_points(x, P):
    sigma_points = np.zeros((2*n+1, n))# 2n+1行   n列矩阵
    sigma_points[0] = x
    S = np.linalg.cholesky((n + lambda_) * P)
    for i in range(n):
        sigma_points[i+1] = x + S[:, i]
        sigma_points[i+1+n] = x - S[:, i]
    return sigma_points
#预测
#传播sigma点（状态转移）
def sigma_points_pass(sigma_points):
    sigma_points_pred = []
    for i in range(2*n+1):
        sigma_points_pred.append(F @sigma_points[i])
    return sigma_points_pred
#重构预测均值
def resturcct_x_pred(sigma_points_pred,Wm):
    x_pred = np.zeros(n)
    for j in range(2*n+1):
        x_pred += Wm[j]*sigma_points_pred[j] 
    return x_pred
#重构预测协方差
def resturcct_P_pred(sigma_points_pred,Wc,x_pred):
    P_pred = np.zeros((n,n))
    for i in range(2*n+1):
        P_pred += Wc[i] * np.outer(sigma_points_pred[i]-x_pred, sigma_points_pred[i]-x_pred)
    P_pred = P_pred+Q
    return P_pred
#更新
def sigma_points_passbyh(sigma_points_pred):
    z_sigma =[]
    for i in range(2*n+1):
        dist = np.sqrt(sigma_points_pred[i][0]**2+sigma_points_pred[i][1]**2)
        ang = np.arctan2(sigma_points_pred[i][1],sigma_points_pred[i][0])
        z_sigma.append([dist,ang])
    return z_sigma
def resturcct_z_pred(z_sigma):
    z_pred =np.zeros(2)
    for j in range(2*n+1):
        z_pred += Wm[j]*np.array(z_sigma[j])
    return z_pred
def resturcct_zz_pred(z_sigma,Wc,z_pred):
    Pzz = np.zeros((2,2))
    for i in range(2*n+1):
        Pzz += Wc[i] * np.outer(z_sigma[i]-z_pred, z_sigma[i]-z_pred)
    Pzz = Pzz+R
    return Pzz
def resturcct_xz_pred(x_pred,z_sigma,Wc,z_pred,sigma_points_pred):
    Pxz = np.zeros((4,2))
    for i in range(2*n+1):
        Pxz += Wc[i] * np.outer(sigma_points_pred[i]-x_pred, z_sigma[i]-z_pred)
    return Pxz

filtered =[]
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
for i in range(100):
    # 预测步
    sigma_points = generate_sigma_points(x, P)
    sigma_points_pred = sigma_points_pass(sigma_points)
    x_pred = resturcct_x_pred(sigma_points_pred, Wm)
    P_pred = resturcct_P_pred(sigma_points_pred, Wc, x_pred)
    
    # 更新步
    z_sigma = sigma_points_passbyh(sigma_points_pred)
    z_pred = resturcct_z_pred(z_sigma)
    Pzz = resturcct_zz_pred(z_sigma, Wc, z_pred)
    Pxz = resturcct_xz_pred(x_pred, z_sigma, Wc, z_pred, sigma_points_pred)
    
    # 卡尔曼增益和状态更新
    K = Pxz @ np.linalg.inv(Pzz)
    x = x_pred + K @ (np.array(obs[i]) - z_pred)
    P = P_pred - K @ Pzz @ K.T
    
    filtered.append(x)
for i in range(100):
    trajx.append(traj[i][0])
    trajy.append(traj[i][1])
    filteredx.append(filtered[i][0])
    filteredy.append(filtered[i][1])

plt.plot(trajx, trajy, label='truth')
plt.plot(filteredx, filteredy, label='UKF')
plt.legend()
plt.show()
