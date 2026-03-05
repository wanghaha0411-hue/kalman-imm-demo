# 粒子滤波算法：开局在初始状态附近瞎猜很多个点，然后根据观测值计算每个粒子的权重，根据权重保留部分粒子，不断循环
# 粒子滤波算法的缺点：粒子退化问题，越到后面粒子越趋向于一致，跟踪效果就很差
# 解决办法：更新粒子时加小扰动避免同质化，增大粒子数量，增大过程噪声
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# 粒子滤波
def predict_particles(particles):
    # 每个粒子按匀速运动移动，加上过程噪声
    particles_pred = []
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    for i in range(N):
        noise = np.random.normal(0, 0.1, 4)
        particles_pred.append(F@particles[i]+noise)
    return np.array(particles_pred)
    
def observe(particle):
    # 给定一个粒子[x,y,vx,vy]，返回[距离, 角度]

    distance = np.sqrt(particle[0]**2+particle[1]**2)
    angle = np.arctan2(particle[1],particle[0])
    return [distance,angle]

def compute_weights(particles, obs_i):
    weights = []
    R = np.diag([1, 0.01])
    for i in range(N):
        z_pred = observe(particles[i])
        w = multivariate_normal.pdf(obs_i, mean=z_pred, cov=R)
        weights.append(w)
    weights = np.array(weights)
    weights /= weights.sum()  # 归一化
    return weights

def resample(particles, weights):
    indices = np.random.choice(N, size=N, replace=True, p=weights)
    return particles[indices]
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
N = 500
filtered= []
particles = np.random.normal(x, [1, 1, 0.1, 0.1], size=(N, 4))
#真实/观测生成
for i in range(100):
    position[0] = position[0]+velocity[0]
    position[1] = position[1]+velocity[1]
    traj.append([position[0],position[1]])
for i in range(100):
    noisedis = np.random.normal(0,1)
    noiseang = np.random.normal(0,0.01)
    position[0] = traj[i][0]+noisedis
    position[1] = traj[i][1]+noisedis
    distance = np.sqrt(traj[i][0]**2+traj[i][1]**2)
    angle = np.arctan2(traj[i][1],traj[i][0])
    obs.append([distance+noisedis,angle+noiseang])

for i in range(100):
    particles = predict_particles(particles)
    weights = compute_weights(particles, obs[i])
    particles = resample(particles, weights)
    particles += np.random.normal(0, 0.1, particles.shape) 
    x = np.mean(particles,axis=0)
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
#plt.scatter(obsx, obsy, label='Noisy Observations', color='red', s=10, alpha=0.5)
plt.plot(filteredx, filteredy, label=' Filter', color='blue', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
