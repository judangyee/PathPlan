# sac_simplified.py
import math, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from track import create_map, show_map

# -------------------------
# 간단 차량 환경 (상태: [x,y,theta,v], 액션: [v_cmd_norm, phi_cmd_norm])
# -------------------------
class VehicleEnvSimple:
    def __init__(self, grid, start, goal, dt=0.2, L=0.5, v_max=2.0, phi_max=math.radians(25)):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.dt = dt; self.L = L
        self.v_max = v_max; self.phi_max = phi_max
        self.sdim = 7
        self.adim = 2
        self.reset()

    def reset(self):
        sx, sy = float(self.start[0]), float(self.start[1])
        gx, gy = float(self.goal[0]), float(self.goal[1])
        theta = math.atan2(gy - sy, gx - sx)
        self.state = np.array([sx, sy, theta, 0.0], dtype=np.float32)
        self.steps = 0
        return self._obs()

    def _obs(self):
        x,y,th,v = self.state
        rel = np.array([self.goal[0]-x, self.goal[1]-y], dtype=np.float32)
        return np.array([x,y,math.sin(th),math.cos(th),v,rel[0],rel[1]], dtype=np.float32)

    def _in_bounds(self,x,y):
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < self.grid.shape[1] and 0 <= yi < self.grid.shape[0]

    def _collision(self, x, y):
        r = 0.5  # 차량 반지름
        for dx in np.linspace(-r, r, 5):
            for dy in np.linspace(-r, r, 5):
                if dx * dx + dy * dy <= r * r:
                    xx = x + dx
                    yy = y + dy
                    if not self._in_bounds(xx, yy): return True
                    if self.grid[int(round(yy)), int(round(xx))] == 1:
                        return True
        return False
    def step(self, action):
        # action in [-1,1]^2 -> scale
        v_cmd = (action[0]+1)/2 * self.v_max
        phi = action[1] * self.phi_max

        x,y,th,v = self.state
        # simple first-order speed
        v = v_cmd
        x_new = x + v * math.cos(th) * self.dt
        y_new = y + v * math.sin(th) * self.dt
        th_new = th + (v / max(1e-3,self.L)) * math.tan(phi) * self.dt

        self.steps += 1
        done = False
        reward = -0.01  # step penalty

        if self._collision(x_new, y_new):
            reward -= 50.0
            done = True
            self.state = np.array([x_new, y_new, th_new, 0.0], dtype=np.float32)
            return self._obs(), reward, done, {}

        self.state = np.array([x_new, y_new, th_new, v], dtype=np.float32)
        prev_dist = math.hypot(self.goal[0]-x, self.goal[1]-y)
        new_dist = math.hypot(self.goal[0]-x_new, self.goal[1]-y_new)
        reward += (prev_dist - new_dist) * 3.0

        if new_dist < 0.6:
            reward += 100.0
            done = True

        if self.steps > 500:
            done = True
        return self._obs(), float(reward), done, {}

    def render_path(self, path, title="Path"):
        plt.figure(figsize=(8,5))
        plt.imshow(self.grid, cmap='gray_r', origin='lower')
        if path:
            xs, ys = zip(*path)
            plt.plot(xs, ys, '-r')
        plt.scatter(self.start[0], self.start[1], c='b', s=80, label='Start')
        plt.scatter(self.goal[0], self.goal[1], c='g', s=80, label='Goal')
        plt.legend(); plt.grid(True); plt.title(title); plt.show()

# -------------------------
# Replay buffer (간단)
# -------------------------
Transition = namedtuple('Transition', ('s','a','r','ns','d'))
class Replay:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Transition(*args))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s = np.stack([b.s for b in batch])
        a = np.stack([b.a for b in batch])
        r = np.stack([b.r for b in batch])
        ns = np.stack([b.ns for b in batch])
        d = np.stack([b.d for b in batch])

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)
        return s, a, r, ns, d

    def __len__(self): return len(self.buf)

# -------------------------
# 간단한 네트워크들
# -------------------------
def mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,out_dim))

class Policy(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        self.net = mlp(sdim, adim*2)  # mean, log_std
    def forward(self, s):
        x = self.net(s)
        mean, logstd = x.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        return mean, logstd
    def sample(self, s):
        mean, logstd = self.forward(s)
        std = logstd.exp()
        z = torch.randn_like(mean)
        x = mean + std * z
        y = torch.tanh(x)
        a = y
        logp = -0.5*((x-mean)/(std+1e-8)).pow(2) - logstd - 0.5*math.log(2*math.pi)
        logp = logp.sum(1, keepdim=True) - torch.log(1 - y.pow(2) + 1e-6).sum(1, keepdim=True)
        return a, logp, torch.tanh(mean)

class QNet(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        self.net = mlp(sdim+adim, 1)
    def forward(self, s,a):
        return self.net(torch.cat([s,a], dim=1))

# -------------------------
# 간단 SAC 에이전트 (고정 alpha)
# -------------------------
class SimpleSAC:
    def __init__(self, sdim, adim, device='cpu'):
        self.device = torch.device(device)
        self.policy = Policy(sdim, adim).to(self.device)
        self.q1 = QNet(sdim, adim).to(self.device)
        self.q2 = QNet(sdim, adim).to(self.device)
        self.q1_t = QNet(sdim, adim).to(self.device); self.q2_t = QNet(sdim, adim).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict()); self.q2_t.load_state_dict(self.q2.state_dict())

        self.pi_opt = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=1e-3)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=1e-3)

        self.gamma = 0.99; self.tau = 0.02; self.alpha = 0.3

    def select(self, s, eval=False):
        t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
        if eval:
            with torch.no_grad():
                _,_,mean = self.policy.sample(t)
            return mean.cpu().numpy()[0]
        else:
            with torch.no_grad():
                a,_,_ = self.policy.sample(t)
            return a.cpu().numpy()[0]

    def update(self, replay, batch=128):
        s,a,r,ns,d = replay.sample(batch)
        s,a,r,ns,d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)

        with torch.no_grad():
            na, nlogp, _ = self.policy.sample(ns)
            tq1 = self.q1_t(ns, na); tq2 = self.q2_t(ns, na)
            tq = torch.min(tq1, tq2) - self.alpha * nlogp
            target = r + (1-d) * self.gamma * tq

        q1_loss = nn.MSELoss()(self.q1(s,a), target)
        q2_loss = nn.MSELoss()(self.q2(s,a), target)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # policy
        new_a, logp, _ = self.policy.sample(s)
        q_pi = torch.min(self.q1(s,new_a), self.q2(s,new_a))
        pi_loss = (self.alpha * logp - q_pi).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        # soft update targets
        for p,tg in zip(self.q1.parameters(), self.q1_t.parameters()):
            tg.data.copy_(self.tau*p.data + (1-self.tau)*tg.data)
        for p,tg in zip(self.q2.parameters(), self.q2_t.parameters()):
            tg.data.copy_(self.tau*p.data + (1-self.tau)*tg.data)

        return {'q1': q1_loss.item(), 'q2': q2_loss.item(), 'pi': pi_loss.item()}

# -------------------------
# 학습 루프 (간단)
# -------------------------
def train_simple(grid, start, goal, episodes=200, init_steps=5000, batch=256, device='cpu'):
    env = VehicleEnvSimple(grid, start, goal)
    sdim = 7; adim = 2
    agent = SimpleSAC(sdim, adim, device=device)
    replay = Replay(200000)

    # 초기 랜덤 데이터 수집
    s = env.reset()
    for _ in range(init_steps):
        a = np.random.uniform(-1, 1, 2)
        ns, r, done, _ = env.step(a)
        r = np.clip(r / 10.0, -10, 10)  # 보상 스케일 안정화
        replay.push(s, a, r, ns, float(done))
        s = ns
        if done:
            s = env.reset()

    rewards = []
    ema_reward = None
    update_per_step = 2  # 한 step에 2번 업데이트

    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_r = 0.0
        for t in range(500):
            # 탐험 유지: 학습 후반에도 약간의 노이즈
            if len(replay) > batch and random.random() > 0.05:
                a = agent.select(s, eval=False)
            else:
                a = np.random.uniform(-1, 1, 2)

            ns, r, done, _ = env.step(a)
            r = np.clip(r / 10.0, -10, 10)
            replay.push(s, a, r, ns, float(done))
            s = ns
            ep_r += r

            # 업데이트를 여러 번 수행
            if len(replay) > batch:
                for _ in range(update_per_step):
                    agent.update(replay, batch=batch)

            if done:
                break

        # EMA 리워드 업데이트
        if ema_reward is None:
            ema_reward = ep_r
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * ep_r

        rewards.append(ep_r)
        if ep % 10 == 0 or ep <= 5:
            avg = np.mean(rewards[-10:])
            print(f"Ep {ep}/{episodes} | avg10={avg:.2f} | ema={ema_reward:.2f} | buffer={len(replay)}")

    return agent, env, rewards

# -------------------------
# 경로 추출(평가)
# -------------------------
def eval_path(agent, env, max_steps=500):
    s = env.reset()
    path=[(float(s[0]), float(s[1]))]
    for _ in range(max_steps):
        a = agent.select(s, eval=True)
        s,_,done,_ = env.step(a)
        path.append((float(s[0]), float(s[1])))
        if done: break
    return path

# -------------------------
# 실행 예제
# -------------------------
if __name__ == "__main__":
    grid, start, goal = create_map(width=36, height=22, obstacle_ratio=0.10)
    # show_map(grid, start, goal)
    agent, env, rewards = train_simple(grid, start, goal, episodes=100, init_steps=1500, batch=256, device='cpu')

    # 모델 저장
    torch.save({
        'policy': agent.policy.state_dict(),
        'q1': agent.q1.state_dict(),
        'q2': agent.q2.state_dict(),
        'alpha': agent.alpha
    }, "sac_model.pth")

    print("모델 저장 완료: sac_model.pth")

    plt.plot(rewards); plt.title("Rewards"); plt.show()
    path = eval_path(agent, env)
    env.render_path(path, title="Simplified SAC Path")
