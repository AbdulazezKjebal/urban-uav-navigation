"""Urban UAV navigation training in Isaac Lab.

Experimental V12 implementation."""
from __future__ import annotations
import argparse, os, sys, math
import json, time, csv, zipfile, warnings
from collections import deque
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description='Urban UAV navigation training')
parser.add_argument('--algo', type=str, default='ea_vla_nav', choices=['td3', 'sac', 'ea_vla_nav'])
parser.add_argument('--scenario', type=str, default='s1', choices=['s1', 's2', 's3'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--total_steps', type=int, default=2000000)
parser.add_argument('--save_interval', type=int, default=100000)
parser.add_argument('--results_dir', type=str, default=os.path.expanduser('~/IsaacLab/results/v7_training'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, CameraCfg, patterns
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
sys.path.insert(0, os.path.expanduser('~/Desktop/ETD3'))
from env_builder_v7 import spawn_environment, SCENARIO_PARAMS, START_X_RANGE, START_Y_RANGE, GOAL_X_RANGE, GOAL_Y_RANGE, CRUISE_ALT, ALT_MIN, ALT_MAX, WORKSPACE
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False
try:
    import open_clip
    _CLIP = True
except ImportError:
    _CLIP = False
    warnings.warn('[VLA] pip install open-clip-torch')
try:
    from stable_baselines3 import SAC as SB3_SAC
    _SB3 = True
except ImportError:
    _SB3 = False
    warnings.warn('[SAC] pip install stable-baselines3')
GRAVITY = 9.81
MASS_NOM = 1.5
MASS_MIN = 1.35
MASS_MAX = 1.65
DRAG_NOM = 0.47
DRAG_MIN = 0.423
DRAG_MAX = 0.517
ROTOR_AREA = 0.04
AIR_DENSITY = 1.225
KI = 1.15
DEPTH_H = 84
DEPTH_W = 84
LIDAR_BEAMS = 64
GOAL_DIM = 3
SOC_DIM = 1
VLA_DIM = 64
ICM_FEAT = 64
ACT_DIM = 4
OBS_BASE = DEPTH_H * DEPTH_W + LIDAR_BEAMS + GOAL_DIM + SOC_DIM
OBS_VLA = OBS_BASE + VLA_DIM
CURRIC = [(15.0, 40.0, 0.7, 50, False), (40.0, 80.0, 0.65, 60, False), (80.0, 150.0, 0.6, 80, False), (150.0, 230.0, 0.55, 100, False), (230.0, 290.0, 2.0, 999, True)]
MAX_EP_STEPS = 4000
HER_K = 4
HER_ENABLE = True
W_GOAL = 20.0
W_ARRIVE = 1000.0
W_HEADING = 1.5
W_VEL = 3.0
W_DIST = 2.0
W_PROX = 4.0
PROX_SIGMA = 8.0
W_SMOOTH = 0.5
W_COLL = -200.0
W_ENERGY = -0.0005
W_STEP = -0.5
W_SEM = 0.5
W_ALT = 0.1
W_INTR = 1.0
GOAL_THRESH = 30.0
PROX_THRESH = 5.0
COLL_THRESH = 1.0
ALT_SIGMA = 10.0
OU_THETA = 0.15
NOISE_START = 0.3
NOISE_END = 0.04
NOISE_DECAY = 0.999997
BUFFER_CAP = 1000000
BATCH = 256
GAMMA = 0.99
TAU = 0.005
LR_A = 0.0003
LR_C = 0.0003
POL_DELAY = 2
TGT_NOISE = 0.2
NOISE_CLIP = 0.5
WARMUP = 25000
PER_ALPHA = 0.6
PER_BETA0 = 0.4
PER_EPS = 1e-06
PER_MAX_P = 10000.0
ICM_FWD_W = 0.8
ICM_INV_W = 0.2
ICM_ETA_0 = 0.8
ICM_ETA_END = 0.05
ICM_DECAY = 0.99999
MMD_W = 0.005
MMD_SIGMA = 1.0
SAC_LR = 0.0003
SAC_BUF = 300000
SAC_BATCH = 256
WIND_STD = 2.0
LABELS = ['building', 'tree', 'vehicle', 'open sky', 'unknown']
RISK = [1.0, 1.2, 2.5, 0.0, 1.0]

def get_obs_dim(algo):
    return OBS_VLA if algo == 'ea_vla_nav' else OBS_BASE
UAV_USD = os.path.expanduser('~/IsaacLab/assets/robots/quadrotor/quadrotor.usd')
UAV_CFG = ArticulationCfg(spawn=sim_utils.UsdFileCfg(usd_path=UAV_USD, activate_contact_sensors=True, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.1, angular_damping=0.1, max_linear_velocity=20.0, max_angular_velocity=40.0, max_depenetration_velocity=1.0), articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0, fix_root_link=False)), init_state=ArticulationCfg.InitialStateCfg(pos=(-105.0, -100.0, CRUISE_ALT), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0, 0, 0), ang_vel=(0, 0, 0), joint_pos={'.*': 0.0}, joint_vel={'.*': 0.0}), actuators={'rotors': ImplicitActuatorCfg(joint_names_expr=['.*'], effort_limit=1.0, velocity_limit=1500.0, stiffness=0.0, damping=0.0)})

@configclass
class SceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path='/World/DomeLight', spawn=sim_utils.DomeLightCfg(intensity=2800.0, color=(0.9, 0.93, 1.0)))
    sun = AssetBaseCfg(prim_path='/World/Sun', spawn=sim_utils.DistantLightCfg(intensity=6000.0, color=(1.0, 0.97, 0.9), angle=0.53))
    ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.GroundPlaneCfg(size=(800.0, 800.0), color=(0.18, 0.36, 0.14)))
    robot: ArticulationCfg = UAV_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    lidar = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/body', update_period=1 / 20, offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0)), mesh_prim_paths=['/World'], pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0, 0), horizontal_fov_range=(-180, 180), horizontal_res=360 / 128), max_distance=50.0, debug_vis=False)
    depth_cam = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/body/depth_sensor', update_period=1 / 20, offset=CameraCfg.OffsetCfg(pos=(0.15, 0, 0), rot=(1, 0, 0, 0)), data_types=['distance_to_image_plane'], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)), width=DEPTH_W, height=DEPTH_H)
    rgb_cam = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/body/rgb_sensor', update_period=1 / 10, offset=CameraCfg.OffsetCfg(pos=(0.15, 0, 0), rot=(1, 0, 0, 0)), data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 1000.0)), width=224, height=224)

class LightweightVLA(nn.Module):
    """Encode RGB observations and compare them with fixed CLIP text labels."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ok = False
        self.proj = nn.Linear(1, VLA_DIM)
        if not _CLIP:
            print('[VLA] CLIP unavailable — zero embeddings')
            return
        try:
            m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.clip = m.to(device).eval()
            for p in self.clip.parameters():
                p.requires_grad = False
            self.proj = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, VLA_DIM))
            tok = open_clip.get_tokenizer('ViT-B-32')
            with torch.no_grad():
                self.txt = F.normalize(self.clip.encode_text(tok(LABELS).to(device)).float(), dim=-1)
            self.ok = True
            n = sum((p.numel() for p in self.proj.parameters()))
            print(f'[VLA] ViT-B/32 frozen | proj={n:,} params')
        except Exception as e:
            warnings.warn(f'[VLA] Load failed: {e}')

    def _norm(self, x):
        m = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        s = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        return (x - m) / s

    def encode(self, rgb):
        if not self.ok:
            return torch.zeros(rgb.shape[0], VLA_DIM, device=self.device)
        with torch.no_grad():
            h = F.normalize(self.clip.encode_image(self._norm(rgb)).float(), dim=-1)
        return self.proj(h)

    @torch.no_grad()
    def classify(self, rgb):
        if not self.ok:
            return torch.zeros(rgb.shape[0], dtype=torch.long, device=self.device)
        h = F.normalize(self.clip.encode_image(self._norm(rgb)).float(), dim=-1)
        return (h @ self.txt.T).argmax(dim=-1)

    def forward(self, rgb):
        return self.encode(rgb)

class ICM(nn.Module):
    """Predict state features and actions to compute a curiosity reward."""
    GD = LIDAR_BEAMS + GOAL_DIM + SOC_DIM

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.enc = nn.Sequential(nn.Linear(self.GD, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, ICM_FEAT), nn.LayerNorm(ICM_FEAT), nn.ReLU())
        self.fwd = nn.Sequential(nn.Linear(ICM_FEAT + ACT_DIM, 128), nn.ReLU(), nn.Linear(128, ICM_FEAT))
        self.inv = nn.Sequential(nn.Linear(ICM_FEAT * 2, 128), nn.ReLU(), nn.Linear(128, ACT_DIM), nn.Tanh())
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _geo(self, obs):
        s = DEPTH_H * DEPTH_W
        return obs[:, s:s + self.GD]

    def forward(self, obs, act, nobs):
        φ = self.enc(self._geo(obs))
        φn = self.enc(self._geo(nobs))
        φn_hat = self.fwd(torch.cat([φ, act], -1))
        l_fwd = F.mse_loss(φn_hat, φn.detach(), reduction='none').mean(-1)
        a_hat = self.inv(torch.cat([φ, φn], -1))
        l_inv = F.mse_loss(a_hat, act.detach(), reduction='none').mean(-1)
        return (l_fwd.detach(), l_fwd.mean(), l_inv.mean())

def mmd_rbf(x, y, σ=MMD_SIGMA):
    """Compute the experimental critic-distribution penalty."""

    def K(a, b):
        r = (a ** 2).sum(1, keepdim=True)
        return torch.exp(-(r + r.T - 2 * a @ b.T) / (2 * σ ** 2))
    B = x.shape[0]
    Kxx = K(x, x)
    Kyy = K(y, y)
    Kxy = K(x, y)
    return ((Kxx.sum() - Kxx.trace()) / (B * (B - 1)) + (Kyy.sum() - Kyy.trace()) / (B * (B - 1)) - 2 * Kxy.mean()).clamp(min=0.0)

class DCEEncoder(nn.Module):
    """Combine depth-image features with LiDAR, goal, and battery inputs."""

    def __init__(self, obs_dim):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            co = self.cnn(torch.zeros(1, 1, DEPTH_H, DEPTH_W)).shape[1]
        self.vp = nn.Sequential(nn.Linear(co, 256), nn.LayerNorm(256), nn.ReLU())
        self.gp = nn.Sequential(nn.Linear(LIDAR_BEAMS + GOAL_DIM + SOC_DIM, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU())
        self.out_dim = 384

    def forward(self, obs):
        d = obs[:, :DEPTH_H * DEPTH_W].view(-1, 1, DEPTH_H, DEPTH_W)
        s = DEPTH_H * DEPTH_W
        l = obs[:, s:s + LIDAR_BEAMS]
        g = obs[:, s + LIDAR_BEAMS:s + LIDAR_BEAMS + GOAL_DIM]
        sc = obs[:, s + LIDAR_BEAMS + GOAL_DIM:s + LIDAR_BEAMS + GOAL_DIM + SOC_DIM]
        return torch.cat([self.vp(self.cnn(d)), self.gp(torch.cat([l, g, sc], -1))], -1)

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Actor(nn.Module):

    def __init__(self, obs_dim):
        super().__init__()
        self.enc = DCEEncoder(obs_dim)
        extra = obs_dim - OBS_BASE
        enc_out = self.enc.out_dim + max(extra, 0)
        self.vla = nn.Linear(VLA_DIM, VLA_DIM) if extra > 0 else None
        self.fc = nn.Sequential(nn.Linear(enc_out, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, ACT_DIM), nn.Tanh())
        self.apply(_init_weights)

    def forward(self, obs):
        f = self.enc(obs)
        if self.vla is not None:
            f = torch.cat([f, self.vla(obs[:, OBS_BASE:])], -1)
        return self.fc(f)

class Critic(nn.Module):

    def __init__(self, obs_dim):
        super().__init__()
        self.e1 = DCEEncoder(obs_dim)
        self.e2 = DCEEncoder(obs_dim)
        extra = obs_dim - OBS_BASE
        eo = self.e1.out_dim + max(extra, 0)
        self.v1 = nn.Linear(VLA_DIM, VLA_DIM) if extra > 0 else None
        self.v2 = nn.Linear(VLA_DIM, VLA_DIM) if extra > 0 else None
        self.q1 = nn.Sequential(nn.Linear(eo + ACT_DIM, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), spectral_norm(nn.Linear(256, 1)))
        self.q2 = nn.Sequential(nn.Linear(eo + ACT_DIM, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), spectral_norm(nn.Linear(256, 1)))
        self.apply(_init_weights)

    def _enc(self, obs, e, v):
        f = e(obs)
        if v is not None:
            f = torch.cat([f, v(obs[:, OBS_BASE:])], -1)
        return f

    def forward(self, obs, act):
        f1 = self._enc(obs, self.e1, self.v1)
        f2 = self._enc(obs, self.e2, self.v2)
        return (self.q1(torch.cat([f1, act], -1)), self.q2(torch.cat([f2, act], -1)))

    def q1_val(self, obs, act):
        return self.q1(torch.cat([self._enc(obs, self.e1, self.v1), act], -1))

class SumTree:

    def __init__(self, cap):
        self.cap = cap
        self.tree = np.zeros(2 * cap - 1, np.float64)
        self.data = [None] * cap
        self.ptr = 0
        self.size = 0

    def _prop(self, i, d):
        p = (i - 1) // 2
        self.tree[p] += d
        if p:
            self._prop(p, d)

    def update(self, i, p):
        d = p - self.tree[i]
        self.tree[i] = p
        self._prop(i, d)

    def add(self, p, data):
        i = self.ptr + self.cap - 1
        self.data[self.ptr] = data
        self.update(i, p)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def get(self, s):
        i = 0
        while i < self.cap - 1:
            l = 2 * i + 1
            if s <= self.tree[l]:
                i = l
            else:
                s -= self.tree[l]
                i = 2 * i + 2
        return (i, self.tree[i], self.data[i - (self.cap - 1)])

    @property
    def total(self):
        return float(self.tree[0])

    @property
    def max_p(self):
        lo = self.cap - 1
        return float(self.tree[lo:lo + self.size].max()) if self.size else 1.0

class PERBuffer:

    def __init__(self, cap, obs_dim, device='cuda'):
        self.tree = SumTree(cap)
        self.beta = PER_BETA0
        self.device = device

    def add(self, o, a, r, no, d):
        p = min(max(self.tree.max_p, 1.0) ** PER_ALPHA, PER_MAX_P)
        self.tree.add(p, (o.astype(np.float32), a.astype(np.float32), float(r), no.astype(np.float32), float(d)))

    def sample(self, bs, t, T):
        self.beta = min(1.0, PER_BETA0 + (1 - PER_BETA0) * t / T)
        idx, pri, bat = ([], [], [])
        total = self.tree.total
        if not np.isfinite(total) or total <= 0:
            total = float(self.tree.size) if self.tree.size else 1.0
        seg = total / bs
        for i in range(bs):
            lo, hi = (seg * i, seg * (i + 1))
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = (0.0, max(total, 1e-06))
            s = np.random.uniform(lo, hi)
            i2, p, d = self.tree.get(s)
            if d is None:
                continue
            idx.append(i2)
            pri.append(p)
            bat.append(d)
        if len(bat) < bs:
            return None
        pr = np.clip(np.array(pri) / self.tree.total, 1e-10, None)
        w = (self.tree.size * pr) ** (-self.beta)
        w = (w / w.max()).astype(np.float32)
        T_ = lambda a: torch.from_numpy(a).to(self.device)
        return (T_(np.stack([b[0] for b in bat])), T_(np.stack([b[1] for b in bat])), T_(np.array([b[2] for b in bat], np.float32)).unsqueeze(1), T_(np.stack([b[3] for b in bat])), T_(np.array([b[4] for b in bat], np.float32)).unsqueeze(1), T_(w), idx)

    def update_p(self, idx, tde):
        for i, e in zip(idx, tde):
            self.tree.update(i, min((abs(float(e)) + PER_EPS) ** PER_ALPHA, PER_MAX_P))

    def __len__(self):
        return self.tree.size

class UniformBuffer:

    def __init__(self, cap, obs_dim, device='cuda'):
        self.cap = cap
        self.ptr = 0
        self.size = 0
        self.device = device
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros((cap, ACT_DIM), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.no = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros((cap, 1), np.float32)

    def add(self, o, a, r, no, d):
        self.o[self.ptr] = o
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.no[self.ptr] = no
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs, *_, **__):
        i = np.random.randint(0, self.size, bs)
        T = lambda x: torch.from_numpy(x).to(self.device)
        return (T(self.o[i]), T(self.a[i]), T(self.r[i]), T(self.no[i]), T(self.d[i]), torch.ones(bs, device=self.device), i)

    def update_p(self, *_):
        pass

    def __len__(self):
        return self.size

class OUNoise:

    def __init__(self, n, dim, device):
        self.n = n
        self.dim = dim
        self.dev = device
        self.sigma = NOISE_START
        self.state = torch.zeros(n, dim, device=device)

    def reset(self, ids):
        self.state[ids] = 0.0

    def sample(self):
        dx = -OU_THETA * self.state + self.sigma * torch.randn_like(self.state)
        self.state = self.state + dx
        return self.state.clone()

    def anneal(self, n_steps: int=1):
        self.sigma = max(NOISE_END, self.sigma * NOISE_DECAY ** n_steps)

class EAVLANavAgent:
    """Experimental TD3 agent with prioritized replay and curiosity."""

    def __init__(self, obs_dim, device, n_envs):
        self.device = device
        self.step = 0
        self.eta = ICM_ETA_0
        self.actor = Actor(obs_dim).to(device)
        self.actor_t = Actor(obs_dim).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim).to(device)
        self.critic_t = Critic(obs_dim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.icm = ICM(device).to(device)
        self.ou = OUNoise(n_envs, ACT_DIM, device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=LR_A, weight_decay=1e-05)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=LR_C, weight_decay=1e-05)
        self.opt_icm = torch.optim.Adam(self.icm.parameters(), lr=0.0003)
        na = sum((p.numel() for p in self.actor.parameters() if p.requires_grad))
        nc = sum((p.numel() for p in self.critic.parameters() if p.requires_grad))
        ni = sum((p.numel() for p in self.icm.parameters() if p.requires_grad))
        print(f'[NET] EA-VLA-Nav | Actor:{na:,} Critic:{nc:,} ICM:{ni:,}')

    def select_action(self, obs, explore=True):
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(obs)
        self.actor.train()
        if explore:
            a = a + self.ou.sample()
            self.ou.anneal(n_steps=self.ou.n)
        return a.clamp(-1.0, 1.0)

    def update(self, buf, total_steps):
        if len(buf) < BATCH:
            return {}
        s = buf.sample(BATCH, total_steps, args.total_steps)
        if s is None:
            return {}
        obs, act, rew, nobs, done, w, idx = s
        r_i, l_fwd, l_inv = self.icm(obs, act, nobs)
        l_icm = ICM_FWD_W * l_fwd + ICM_INV_W * l_inv
        self.opt_icm.zero_grad()
        l_icm.backward()
        nn.utils.clip_grad_norm_(self.icm.parameters(), 10.0)
        self.opt_icm.step()
        eta = self.eta
        self.eta = max(ICM_ETA_END, self.eta * ICM_DECAY)
        rew_aug = rew + W_INTR * eta * r_i.unsqueeze(1).detach()
        with torch.no_grad():
            noise = (torch.randn_like(act) * TGT_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            na = (self.actor_t(nobs) + noise).clamp(-1, 1)
            q1t, q2t = self.critic_t(nobs, na)
            tgt = rew_aug + GAMMA * (1 - done) * torch.min(q1t, q2t)
        q1, q2 = self.critic(obs, act)
        tde = ((q1 - tgt).abs() + (q2 - tgt).abs()).squeeze().detach().cpu().numpy() / 2.0
        l_c = (w * ((q1 - tgt).pow(2) + (q2 - tgt).pow(2)).squeeze()).mean()
        l_c = l_c + MMD_W * mmd_rbf(q1, q2)
        self.opt_c.zero_grad()
        l_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()
        buf.update_p(idx, tde)
        l_a_val = 0.0
        if self.step % POL_DELAY == 0:
            a_pred = self.actor(obs)
            q_val = self.critic.q1_val(obs, a_pred)
            ldr = obs[:, DEPTH_H * DEPTH_W:DEPTH_H * DEPTH_W + LIDAR_BEAMS].detach()
            d_min = ldr.min(-1).values.clamp(min=0.05)
            safe = (1.0 / d_min).mean()
            l_a = -q_val.mean() + 0.05 * safe
            self.opt_a.zero_grad()
            l_a.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_a.step()
            l_a_val = l_a.item()
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
        self.step += 1
        return {'loss_c': l_c.item(), 'loss_a': l_a_val, 'loss_icm': l_icm.item(), 'mmd': (MMD_W * mmd_rbf(q1, q2)).item(), 'tde': float(tde.mean()), 'noise': self.ou.sigma, 'icm_eta': self.eta}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'icm': self.icm.state_dict(), 'step': self.step, 'noise': self.ou.sigma, 'icm_eta': self.eta}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(c['actor'])
        self.critic.load_state_dict(c['critic'])
        if 'icm' in c:
            self.icm.load_state_dict(c['icm'])
        self.step = c.get('step', 0)
        self.ou.sigma = c.get('noise', NOISE_START)
        self.eta = c.get('icm_eta', ICM_ETA_0)

class TD3Agent:
    """TD3 agent with twin critics and delayed policy updates."""

    def __init__(self, obs_dim, device, n_envs):
        self.device = device
        self.step = 0
        self.actor = Actor(obs_dim).to(device)
        self.actor_t = Actor(obs_dim).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim).to(device)
        self.critic_t = Critic(obs_dim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=LR_C)
        na = sum((p.numel() for p in self.actor.parameters() if p.requires_grad))
        print(f'[NET] TD3 baseline | Actor:{na:,}')

    def select_action(self, obs, explore=True):
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(obs)
        self.actor.train()
        if explore:
            a = a + torch.randn_like(a) * 0.1
        return a.clamp(-1.0, 1.0)

    def update(self, buf, total_steps):
        if len(buf) < BATCH:
            return {}
        s = buf.sample(BATCH, total_steps, args.total_steps)
        if s is None:
            return {}
        obs, act, rew, nobs, done, w, idx = s
        with torch.no_grad():
            noise = (torch.randn_like(act) * TGT_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            na = (self.actor_t(nobs) + noise).clamp(-1, 1)
            q1t, q2t = self.critic_t(nobs, na)
            tgt = rew + GAMMA * (1 - done) * torch.min(q1t, q2t)
        q1, q2 = self.critic(obs, act)
        l_c = ((q1 - tgt).pow(2) + (q2 - tgt).pow(2)).mean()
        self.opt_c.zero_grad()
        l_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()
        l_a_val = 0.0
        if self.step % POL_DELAY == 0:
            l_a = -self.critic.q1_val(obs, self.actor(obs)).mean()
            self.opt_a.zero_grad()
            l_a.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_a.step()
            l_a_val = l_a.item()
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
        self.step += 1
        return {'loss_c': l_c.item(), 'loss_a': l_a_val, 'noise': 0.1}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'step': self.step}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(c['actor'])
        self.critic.load_state_dict(c['critic'])
        self.step = c.get('step', 0)

class SACWrapper:
    """Connect Stable-Baselines3 SAC to the simulation loop."""

    def __init__(self, obs_dim, device, n_envs):
        self.device = device
        self.n_envs = n_envs
        self.step = 0
        if not _SB3:
            raise RuntimeError('pip install stable-baselines3')
        import gymnasium as gym

        class _E(gym.Env):

            def __init__(s, od):
                s.observation_space = gym.spaces.Box(-1, 1, (od,), np.float32)
                s.action_space = gym.spaces.Box(-1, 1, (ACT_DIM,), np.float32)

            def reset(s, **k):
                return (np.zeros(s.observation_space.shape, np.float32), {})

            def step(s, a):
                return (np.zeros(s.observation_space.shape, np.float32), 0.0, False, False, {})
        self._m = SB3_SAC('MlpPolicy', _E(obs_dim), learning_rate=SAC_LR, buffer_size=SAC_BUF, batch_size=SAC_BATCH, learning_starts=WARMUP, ent_coef='auto', verbose=0, device=device)
        np_ = sum((p.numel() for p in self._m.policy.parameters() if p.requires_grad))
        print(f'[NET] SAC | Policy:{np_:,}')

    def select_action(self, obs, explore=True):
        a, _ = self._m.predict(obs.detach().cpu().numpy(), deterministic=not explore)
        return torch.from_numpy(a.astype(np.float32)).to(self.device)

    def add_to_buffer(self, obs, act, rew, nobs, done):
        for i in range(self.n_envs):
            self._m.replay_buffer.add(obs[i:i + 1].detach().cpu().numpy(), nobs[i:i + 1].detach().cpu().numpy(), act[i:i + 1].detach().cpu().numpy(), np.array([[float(rew[i])]]), np.array([[float(done[i])]]), [{}])

    def update(self, buf, total_steps):
        if self._m.replay_buffer.size() < WARMUP:
            return {}
        self._m.train(gradient_steps=1, batch_size=SAC_BATCH)
        self.step += 1
        return {'noise': 0.0}

    def save(self, path):
        self._m.save(str(path))

    def load(self, path):
        self._m = SB3_SAC.load(str(path), device=self.device)

def _depth(scene, N, dev):
    try:
        d = scene['depth_cam'].data.output['distance_to_image_plane']
        if d.dim() == 4:
            d = d.squeeze(-1)
        d = d.to(dev).float().clamp(0.3, 10.0) / 10.0
        return (d + torch.randn_like(d) * 0.01).clamp(0, 1)
    except Exception:
        return torch.full((N, DEPTH_H, DEPTH_W), 0.5, device=dev)

def _rgb(scene, N, dev):
    try:
        r = scene['rgb_cam'].data.output['rgb'].to(dev).float()
        return r[..., :3].permute(0, 3, 1, 2) / 255.0
    except Exception:
        return torch.full((N, 3, 224, 224), 0.5, device=dev)

def _lidar(scene, N, dev):
    try:
        pos = scene['robot'].data.root_pos_w
        hits = scene['lidar'].data.ray_hits_w
        r = torch.norm(hits - pos.unsqueeze(1), dim=-1).clamp(0, 50) / 50.0
        return r.view(N, 64, 2).mean(-1).to(dev)
    except Exception:
        return torch.full((N, LIDAR_BEAMS), 1.0, device=dev)

def get_obs(scene, goal, soc, dev, vla=None):
    """Assemble sensor observations and the simulator-derived relative goal."""
    N = scene.num_envs
    pos = scene['robot'].data.root_pos_w
    depth = _depth(scene, N, dev).view(N, -1)
    lidar = _lidar(scene, N, dev)
    delta = goal - pos
    dist = torch.norm(delta, dim=-1, keepdim=True).clamp(0.001)
    diag = math.sqrt(WORKSPACE ** 2 * 3)
    phi = torch.atan2(delta[:, 1:2], delta[:, 0:1]) / math.pi
    theta = torch.atan2(delta[:, 2:3], dist) / (math.pi / 2)
    gp = torch.cat([dist / diag, phi, theta], -1)
    sc = soc.unsqueeze(-1).to(dev)
    base = torch.cat([depth, lidar, gp, sc], -1)
    if vla is not None:
        rgb = _rgb(scene, N, dev)
        return (torch.cat([base, vla.encode(rgb)], -1), rgb)
    return (base, None)
_GOAL_VEC_START = OBS_BASE - GOAL_DIM - SOC_DIM
_GOAL_VEC_END = OBS_BASE - SOC_DIM

def relabel_goal_vec(obs_np, pos_np, virtual_goal_np):
    obs_new = obs_np.copy()
    delta = virtual_goal_np - pos_np
    dist = max(float(np.linalg.norm(delta)), 0.001)
    diag = math.sqrt(WORKSPACE ** 2 * 3)
    phi = math.atan2(delta[1], delta[0]) / math.pi
    theta = math.atan2(delta[2], dist) / (math.pi / 2)
    obs_new[_GOAL_VEC_START:_GOAL_VEC_END] = [dist / diag, phi, theta]
    return obs_new

def her_reward(pos_np, virtual_goal_np, collided):
    dist = float(np.linalg.norm(virtual_goal_np - pos_np))
    arrived = dist < GOAL_THRESH
    diag = math.sqrt(WORKSPACE ** 2 * 2)
    r_dist = -W_DIST * (dist / diag)
    r_arrive = W_ARRIVE if arrived else 0.0
    return (r_dist, arrived, r_arrive)

class EpisodeBuffer:

    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.reset_all()

    def reset_all(self):
        self.data = [[] for _ in range(self.n_envs)]

    def reset_env(self, i):
        self.data[i] = []

    def append(self, i, obs, pos, act, rew, nobs, npos, done):
        self.data[i].append((obs, pos, act, rew, nobs, npos, done))

    def __len__(self):
        return sum((len(d) for d in self.data))

class Curriculum:

    def __init__(self):
        self.stage = 0
        self.window = deque(maxlen=200)
        self.patience = 0

    def update(self, success):
        self.window.append(float(success))

    @property
    def sr(self):
        return float(np.mean(self.window)) if self.window else 0.0

    @property
    def dist_range(self):
        s = min(self.stage, len(CURRIC) - 1)
        return (CURRIC[s][0], CURRIC[s][1])

    @property
    def is_final_stage(self):
        s = min(self.stage, len(CURRIC) - 1)
        return CURRIC[s][4]

    def maybe_advance(self):
        if self.stage >= len(CURRIC) - 1:
            return
        thr = CURRIC[self.stage][2]
        pat = CURRIC[self.stage][3]
        if self.sr >= thr:
            self.patience += 1
        else:
            self.patience = 0
        if self.patience >= pat:
            self.stage += 1
            self.patience = 0
            self.window.clear()
            d0, d1 = self.dist_range
            print(f'\n[CURRIC] Stage {self.stage} | dist {d0:.0f}–{d1:.0f} m | final_geometry={self.is_final_stage}\n')

def reset_envs(scene, ids, goal, mass, drag, energy, soc, dev, cur, rng, ou=None):
    """Reset selected UAVs and sample goal and model parameters."""
    n = ids.numel()
    if n == 0:
        return (goal, mass, drag, energy, soc)
    d0, d1 = cur.dist_range
    sz = torch.full((n,), CRUISE_ALT, device=dev)
    gz = torch.full((n,), CRUISE_ALT, device=dev)
    MAX_TRIES = 50
    if cur.is_final_stage:
        sx = torch.tensor(rng.uniform(*START_X_RANGE, n), device=dev, dtype=torch.float32)
        sy = torch.tensor(rng.uniform(*START_Y_RANGE, n), device=dev, dtype=torch.float32)
        sx_np = sx.cpu().numpy()
        sy_np = sy.cpu().numpy()
        gx_out = np.empty(n, dtype=np.float32)
        gy_out = np.empty(n, dtype=np.float32)
        for i in range(n):
            best_gx, best_gy, best_miss = (None, None, float('inf'))
            for _ in range(MAX_TRIES):
                gx_c = rng.uniform(*GOAL_X_RANGE)
                gy_c = rng.uniform(*GOAL_Y_RANGE)
                d = math.sqrt((gx_c - sx_np[i]) ** 2 + (gy_c - sy_np[i]) ** 2)
                if d0 <= d <= d1:
                    best_gx, best_gy = (gx_c, gy_c)
                    break
                miss = max(d0 - d, d - d1, 0.0)
                if miss < best_miss:
                    best_miss, best_gx, best_gy = (miss, gx_c, gy_c)
            gx_out[i], gy_out[i] = (best_gx, best_gy)
        gx = torch.tensor(gx_out, device=dev)
        gy = torch.tensor(gy_out, device=dev)
    else:
        margin = 20.0
        gx_np = rng.uniform(-WORKSPACE + margin, WORKSPACE - margin, n)
        gy_np = rng.uniform(-WORKSPACE + margin, WORKSPACE - margin, n)
        sx_out = np.empty(n, dtype=np.float32)
        sy_out = np.empty(n, dtype=np.float32)
        for i in range(n):
            dist = rng.uniform(d0, d1)
            ang = rng.uniform(0, 2 * math.pi)
            sx_c = gx_np[i] + dist * math.cos(ang)
            sy_c = gy_np[i] + dist * math.sin(ang)
            for _ in range(10):
                if -WORKSPACE <= sx_c <= WORKSPACE and -WORKSPACE <= sy_c <= WORKSPACE:
                    break
                ang = rng.uniform(0, 2 * math.pi)
                sx_c = gx_np[i] + dist * math.cos(ang)
                sy_c = gy_np[i] + dist * math.sin(ang)
            sx_out[i] = np.clip(sx_c, -WORKSPACE, WORKSPACE)
            sy_out[i] = np.clip(sy_c, -WORKSPACE, WORKSPACE)
        sx = torch.tensor(sx_out, device=dev, dtype=torch.float32)
        sy = torch.tensor(sy_out, device=dev, dtype=torch.float32)
        gx = torch.tensor(gx_np, device=dev, dtype=torch.float32)
        gy = torch.tensor(gy_np, device=dev, dtype=torch.float32)
    rs = scene['robot'].data.default_root_state.clone()[ids]
    rs[:, 0] = sx
    rs[:, 1] = sy
    rs[:, 2] = sz
    rs[:, 3] = 1.0
    rs[:, 4:7] = 0.0
    rs[:, 7:] = 0.0
    scene['robot'].write_root_state_to_sim(rs, env_ids=ids)
    goal[ids] = torch.stack([gx, gy, gz], -1)
    mass[ids] = torch.zeros(n, device=dev).uniform_(MASS_MIN, MASS_MAX)
    drag[ids] = torch.zeros(n, device=dev).uniform_(DRAG_MIN, DRAG_MAX)
    energy[ids] = 0.0
    soc[ids] = 1.0
    if ou is not None:
        ou.reset(ids)
    return (goal, mass, drag, energy, soc)

def compute_reward(scene, goal, prev_dist, mass, drag, energy, dt, dev, scenario, use_energy=True, vla=None, rgb=None, prev_act=None, path_len=None):
    """Combine goal progress, clearance, altitude, and optional power costs."""
    N = scene.num_envs
    pos = scene['robot'].data.root_pos_w
    vel = scene['robot'].data.root_lin_vel_w
    curr_dist = torch.norm(goal - pos, dim=-1)
    arrived = curr_dist < GOAL_THRESH
    r_goal = W_GOAL * (prev_dist - curr_dist) + arrived.float() * W_ARRIVE
    gdir = F.normalize(goal - pos + 1e-08, dim=-1)
    vnrm = F.normalize(vel + 1e-08, dim=-1)
    r_head = W_HEADING * (gdir * vnrm).sum(-1).clamp(-1, 1)
    cos_theta = (vnrm * gdir).sum(-1).clamp(0, 1)
    r_vel = W_VEL * cos_theta ** 2
    DIAG_MAX = math.sqrt(WORKSPACE ** 2 * 2)
    r_dist = -W_DIST * (curr_dist / DIAG_MAX)
    ldr = _lidar(scene, N, dev) * 50.0
    d_min = ldr.min(-1).values.clamp(min=0.1)
    if vla is not None and rgb is not None:
        cls = vla.classify(rgb)
        risk = torch.tensor(RISK, device=dev)[cls]
    else:
        risk = torch.ones(N, device=dev)
    r_prox = -W_PROX * torch.exp(-d_min / PROX_SIGMA) * risk
    r_sem = W_SEM * (risk == 0.0).float() if vla is not None and rgb is not None else torch.zeros(N, device=dev)
    if prev_act is not None:
        r_smooth = torch.zeros(N, device=dev)
    else:
        r_smooth = torch.zeros(N, device=dev)
    alt_err = (pos[:, 2] - CRUISE_ALT).abs()
    r_alt = W_ALT * torch.exp(-alt_err ** 2 / (2 * ALT_SIGMA ** 2))
    oob_alt = ((pos[:, 2] < ALT_MIN) | (pos[:, 2] > ALT_MAX)).float()
    r_alt = r_alt - 0.5 * oob_alt
    spd = torch.norm(vel, -1).clamp(0)
    P_ind = KI * (mass * GRAVITY).pow(1.5) / math.sqrt(2 * AIR_DENSITY * ROTOR_AREA)
    P_par = 0.5 * drag * ROTOR_AREA * AIR_DENSITY * spd ** 3
    P_tot = P_ind + P_par
    r_E = W_ENERGY * P_tot if use_energy else torch.zeros(N, device=dev)
    energy = energy + P_tot * dt
    r_step = torch.full((N,), W_STEP, device=dev)
    if path_len is not None:
        path_len += spd * dt
    collision = d_min < COLL_THRESH
    oob = (pos[:, 0].abs() > WORKSPACE + 10) | (pos[:, 1].abs() > WORKSPACE + 10) | (pos[:, 2] < ALT_MIN - 5) | (pos[:, 2] > ALT_MAX + 5)
    done = collision | oob | arrived
    if arrived.any():
        print(f'[ARRIVE] n={int(arrived.sum())}  dist={curr_dist[arrived].mean():.1f}m  thresh={GOAL_THRESH}m')
    reward = r_goal + r_head + r_vel + r_dist + r_prox + r_smooth + r_alt + r_E + r_step + r_sem + collision.float() * W_COLL
    return (reward, curr_dist, energy, done, arrived, collision)

class DynamicManager:
    """Update obstacle position arrays and sample horizontal wind forces."""

    def __init__(self, device, seed):
        rng = np.random.default_rng(seed + 9999)
        self.device = device
        self.uav_pos = torch.tensor([[float(rng.uniform(-100, 100)), float(rng.uniform(-100, 0)), CRUISE_ALT + float(rng.uniform(-8, 8))] for _ in range(5)], device=device, dtype=torch.float32)
        self.uav_vel = torch.zeros(5, 3, device=device)
        for i in range(5):
            self.uav_vel[i, 0] = float(rng.choice([-3.0, -2.0, 2.0, 3.0]))
            self.uav_vel[i, 1] = float(rng.choice([-3.0, -2.0, 2.0, 3.0]))
        sx = 32.0 + 12.0
        self.car_pos = torch.tensor([[float(rng.choice([-120 + int(rng.integers(0, 5)) * sx + 6])), float(rng.uniform(-110, 110)), 0.73] for _ in range(5)], device=device, dtype=torch.float32)
        self.car_vel = torch.zeros(5, 3, device=device)
        for i in range(5):
            self.car_vel[i, 1] = float(rng.choice([-3.0, -2.0, 2.0, 3.0]))

    def step(self, dt):
        nu = torch.randn_like(self.uav_vel) * 0.05
        self.uav_pos = self.uav_pos + (self.uav_vel + nu) * dt
        for ax in [0, 1]:
            self.uav_pos[:, ax] = torch.where(self.uav_pos[:, ax] > 120, torch.full_like(self.uav_pos[:, ax], -120.0), self.uav_pos[:, ax])
            self.uav_pos[:, ax] = torch.where(self.uav_pos[:, ax] < -120, torch.full_like(self.uav_pos[:, ax], 120.0), self.uav_pos[:, ax])
        nc = torch.randn_like(self.car_vel) * 0.02
        self.car_pos = self.car_pos + (self.car_vel + nc) * dt
        self.car_pos[:, 1] = torch.where(self.car_pos[:, 1] > 110, torch.full_like(self.car_pos[:, 1], -110.0), self.car_pos[:, 1])
        self.car_pos[:, 1] = torch.where(self.car_pos[:, 1] < -110, torch.full_like(self.car_pos[:, 1], 110.0), self.car_pos[:, 1])

    def get_wind(self, N):
        w = torch.zeros(N, 3, device=self.device)
        w[:, 0] = torch.randn(N, device=self.device) * WIND_STD
        w[:, 1] = torch.randn(N, device=self.device) * WIND_STD
        return w

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng_np = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = args.num_envs
    algo = args.algo
    scenario = args.scenario
    use_E = algo == 'ea_vla_nav'
    use_V = algo == 'ea_vla_nav'
    obs_dim = get_obs_dim(algo)
    scfg = SCENARIO_PARAMS[scenario]
    run = f'{algo}_{scenario}_seed{args.seed}'
    root = Path(args.results_dir) / algo / scenario / f'seed_{args.seed}'
    ckpt = root / 'checkpoints'
    log = root / 'logs'
    ckpt.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    cf = open(root / 'training_metrics.csv', 'w', newline='')
    cw = csv.writer(cf)
    cw.writerow(['step', 'reward', 'sr', 'energy_J', 'cr', 'noise', 'icm_eta', 'loss_c', 'loss_a', 'loss_icm', 'mmd', 'stage', 'altitude', 'spl'])
    writer = SummaryWriter(log_dir=str(log)) if _TB else None
    hp = {'algo': algo, 'scenario': scenario, 'scenario_name': scfg['name'], 'seed': args.seed, 'num_envs': N, 'total_steps': args.total_steps, 'obs_dim': obs_dim, 'act_dim': ACT_DIM, 'gamma': GAMMA, 'tau': TAU, 'batch': BATCH, 'buffer': BUFFER_CAP, 'lr_actor': LR_A, 'lr_critic': LR_C, 'policy_delay': POL_DELAY, 'target_noise_std': TGT_NOISE, 'target_noise_clip': NOISE_CLIP, 'warmup_steps': WARMUP, 'ou_theta': OU_THETA, 'noise_start': NOISE_START, 'noise_end': NOISE_END, 'noise_decay': NOISE_DECAY, 'per_alpha': PER_ALPHA, 'per_beta0': PER_BETA0, 'per_eps': PER_EPS, 'icm_fwd_w': ICM_FWD_W, 'icm_inv_w': ICM_INV_W, 'icm_eta_0': ICM_ETA_0, 'icm_eta_end': ICM_ETA_END, 'mmd_weight': MMD_W, 'mmd_sigma': MMD_SIGMA, 'w_goal': W_GOAL, 'w_arrive': W_ARRIVE, 'w_heading': W_HEADING, 'w_vel': W_VEL, 'w_dist': W_DIST, 'w_prox': W_PROX, 'w_collision': W_COLL, 'w_energy': W_ENERGY, 'w_step': W_STEP, 'w_altitude': W_ALT, 'goal_thresh_m': GOAL_THRESH, 'prox_thresh_m': PROX_THRESH, 'coll_thresh_m': COLL_THRESH, 'max_ep_steps': MAX_EP_STEPS, 'her_k': HER_K, 'her_enable': HER_ENABLE, 'curriculum_stages': CURRIC, 'v12_fixes': {'STRUCT1_HER': f'HER_K={HER_K}, future strategy (Andrychowicz 2017)', 'STRUCT2_CURRIC': 'inverted: 15m→290m near-goal-out (was 230m→360m far-out)', 'BUGFIX_GOAL_THRESH': 'duplicate constant block was silently reverting GOAL_THRESH 30m→15m in v10; removed'}, 'mass_nom_kg': MASS_NOM, 'mass_dr': f'U({MASS_MIN},{MASS_MAX})', 'drag_nom': DRAG_NOM, 'drag_dr': f'U({DRAG_MIN},{DRAG_MAX})', 'rotor_area_m2': ROTOR_AREA, 'air_density': AIR_DENSITY, 'ki': KI, 'cruise_alt_m': CRUISE_ALT, 'alt_min_m': ALT_MIN, 'alt_max_m': ALT_MAX, 'start_zone_x': list(START_X_RANGE), 'start_zone_y': list(START_Y_RANGE), 'goal_zone_x': list(GOAL_X_RANGE), 'goal_zone_y': list(GOAL_Y_RANGE), 'curriculum_stages': CURRIC, 'n_buildings': scfg['n_blocks_x'] * scfg['n_blocks_y'] * scfg['bld_per_block'], 'dr_height': '±20%', 'dr_position': '±3m', 'sensor_noise': 'N(0,0.01²) depth', 'wind_std_N': WIND_STD if scenario == 's3' else 0}
    with open(root / 'hyperparameters.json', 'w') as f:
        json.dump(hp, f, indent=2)
    print(f"\n{'═' * 65}")
    print(f'  EA-VLA-Nav v12  —  IEEE RA-L')
    print(f'  Algorithm  : {algo.upper()}')
    print(f"  Scenario   : {scenario.upper()} — {scfg['name']}")
    print(f"  Grid       : {scfg['n_blocks_x']}×{scfg['n_blocks_y']} | {scfg['n_blocks_x'] * scfg['n_blocks_y'] * scfg['bld_per_block']} buildings")
    print(f'  Flight     : SW→NE diagonal (non-trivial)')
    print(f'  Altitude   : {ALT_MIN}–{ALT_MAX}m (pref {CRUISE_ALT}m)')
    print(f'  DR         : h±20% | pos±3m | mass±10% | drag±10%')
    print(f'  Seed       : {args.seed}')
    print(f'  Steps      : {args.total_steps:,}')
    print(f'  Device     : {device}')
    print(f"{'═' * 65}\n")
    vla = LightweightVLA(device).to(device) if use_V else None
    sim_cfg = SimulationCfg(dt=0.02, gravity=(0, 0, -GRAVITY), render_interval=4)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=500.0))
    env_stats = spawn_environment(scenario, seed=args.seed)
    sim.reset()
    print(f"[INFO] Environment ready | {env_stats['n_buildings']} buildings")
    body_ids = None
    try:
        body_ids, _ = scene['robot'].find_bodies(['body'])
    except Exception:
        pass
    dyn = DynamicManager(device, args.seed) if scenario == 's3' else None
    if algo == 'ea_vla_nav':
        agent = EAVLANavAgent(obs_dim, device, N)
        buf = PERBuffer(BUFFER_CAP, obs_dim, device=device)
    elif algo == 'td3':
        agent = TD3Agent(obs_dim, device, N)
        buf = UniformBuffer(BUFFER_CAP, obs_dim, device=device)
    else:
        agent = SACWrapper(obs_dim, device, N)
        buf = None
    cur = Curriculum()
    goal = torch.zeros(N, 3, device=device)
    mass = torch.full((N,), MASS_NOM, device=device)
    drag = torch.full((N,), DRAG_NOM, device=device)
    energy = torch.zeros(N, device=device)
    soc = torch.ones(N, device=device)
    all_ids = torch.arange(N, device=device)
    ou = getattr(agent, 'ou', None)
    goal, mass, drag, energy, soc = reset_envs(scene, all_ids, goal, mass, drag, energy, soc, device, cur, rng_np, ou)
    obs, rgb = get_obs(scene, goal, soc, device, vla)
    prev_d = torch.norm(goal - scene['robot'].data.root_pos_w, -1)
    ep_r = torch.zeros(N, device=device)
    ep_s = torch.zeros(N, device=device)
    ep_e = torch.zeros(N, device=device)
    ep_c = torch.zeros(N, device=device)
    ep_t = torch.zeros(N, device=device)
    ep_a = torch.zeros(N, device=device)
    ep_path = torch.zeros(N, device=device)
    ep_d0 = torch.zeros(N, device=device)
    ep_n = 0
    wr = deque(maxlen=200)
    wsr = deque(maxlen=200)
    we = deque(maxlen=200)
    wcr = deque(maxlen=200)
    wt = deque(maxlen=200)
    wa = deque(maxlen=200)
    wspl = deque(maxlen=200)
    best_sr = 0.0
    gs = 0
    t0 = time.time()
    prev_act = torch.zeros(N, ACT_DIM, device=device)
    epbuf = EpisodeBuffer(N) if HER_ENABLE and algo != 'sac' else None
    her_count = 0
    print(f'[INFO] Training — warmup {WARMUP:,} steps\n')
    while gs < args.total_steps:
        if gs < WARMUP:
            act = torch.zeros(N, ACT_DIM, device=device).uniform_(-1.0, 1.0)
        else:
            act = agent.select_action(obs, explore=True)
        act = torch.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
        act = torch.clamp(act, -1.0, 1.0)
        jerk = -W_SMOOTH * (act - prev_act).pow(2).sum(-1)
        vel = scene['robot'].data.root_lin_vel_w
        vel = torch.nan_to_num(vel, nan=0.0, posinf=50.0, neginf=-50.0)
        vel = torch.clamp(vel, -50.0, 50.0)
        tv = act[:, :3] * 8.0
        fc = (tv - vel) * mass.unsqueeze(-1) * 20.0
        fg = torch.zeros(N, 3, device=device)
        fg[:, 2] = mass * GRAVITY
        wind = dyn.get_wind(N) if dyn is not None else torch.zeros(N, 3, device=device)
        nf = (fc + fg + wind).unsqueeze(1)
        nf = torch.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0)
        nf = torch.clamp(nf, -2000.0, 2000.0)
        if body_ids is not None:
            scene['robot'].set_external_force_and_torque(forces=nf, torques=torch.zeros(N, 1, 3, device=device), body_ids=body_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        if dyn is not None:
            dyn.step(sim_cfg.dt)
        rew, prev_d, energy, done, arrived, collision = compute_reward(scene, goal, prev_d, mass, drag, energy, sim_cfg.dt, device, scenario, use_E, vla, rgb, prev_act=prev_act, path_len=ep_path)
        rew = rew + jerk
        prev_act = act.detach().clone()
        soc = (1.0 - energy / 100000.0).clamp(0, 1)
        nobs, nrgb = get_obs(scene, goal, soc, device, vla)
        on = obs.detach().cpu().numpy()
        an = act.detach().cpu().numpy()
        rn = rew.detach().cpu().numpy()
        non = nobs.detach().cpu().numpy()
        dn = done.float().detach().cpu().numpy()
        pos_now = scene['robot'].data.root_pos_w.detach().cpu().numpy()
        if algo == 'sac':
            agent.add_to_buffer(obs, act, rew, nobs, done.float())
        else:
            for i in range(N):
                buf.add(on[i], an[i], float(rn[i]), non[i], float(dn[i]))
                if epbuf is not None:
                    epbuf.append(i, on[i], pos_now[i], an[i], float(rn[i]), non[i], pos_now[i], bool(dn[i]))
        ep_r += rew
        ep_e += energy
        ep_s = torch.maximum(ep_s, arrived.float())
        ep_c += collision.float()
        ep_t += 1.0
        ep_a += scene['robot'].data.root_pos_w[:, 2]
        timeout = ep_t >= MAX_EP_STEPS
        done = done | timeout
        info = {}
        if gs >= WARMUP:
            info = agent.update(None, gs) if algo == 'sac' else agent.update(buf, gs)
            if writer and info:
                for k, v in info.items():
                    writer.add_scalar(f'train/{k}', v, gs)
        dids = done.nonzero(as_tuple=False).squeeze(-1)
        if dids.numel() > 0:
            for i in dids.cpu().tolist():
                wr.append(float(ep_r[i]))
                wsr.append(float(ep_s[i]))
                we.append(float(ep_e[i]))
                wcr.append(float(ep_c[i] > 0))
                wt.append(float(ep_t[i]))
                wa.append(float(ep_a[i] / max(ep_t[i], 1)))
                L_star = float(ep_d0[i])
                L_agent = float(ep_path[i])
                spl_i = float(ep_s[i]) * L_star / max(L_agent, L_star, 1.0)
                wspl.append(spl_i)
                cur.update(bool(ep_s[i]))
                ep_n += 1
                if epbuf is not None and len(epbuf.data[i]) >= 2:
                    traj = epbuf.data[i]
                    T = len(traj)
                    for _ in range(HER_K):
                        t_idx = int(rng_np.integers(0, T - 1))
                        f_idx = int(rng_np.integers(t_idx + 1, T))
                        o_t, p_t, a_t, _r_t, no_t, np_t, _d_t = traj[t_idx]
                        _, virt_goal, *_ = traj[f_idx]
                        o_relab = relabel_goal_vec(o_t, p_t, virt_goal)
                        no_relab = relabel_goal_vec(no_t, np_t, virt_goal)
                        r_relab, arrived_v, arrive_bonus = her_reward(np_t, virt_goal, collided=False)
                        old_dist_to_real_goal = None
                        her_final_reward = r_relab + (W_ARRIVE if arrived_v else 0.0) + W_STEP
                        d_relab = 1.0 if arrived_v else 0.0
                        buf.add(o_relab, a_t, her_final_reward, no_relab, d_relab)
                        her_count += 1
                if epbuf is not None:
                    epbuf.reset_env(i)
            ep_r[dids] = 0.0
            ep_s[dids] = 0.0
            ep_e[dids] = 0.0
            ep_c[dids] = 0.0
            ep_t[dids] = 0.0
            ep_a[dids] = 0.0
            ep_path[dids] = 0.0
            prev_act[dids] = 0.0
            cur.maybe_advance()
            goal, mass, drag, energy, soc = reset_envs(scene, dids, goal, mass, drag, energy, soc, device, cur, rng_np, ou)
            prev_d[dids] = torch.norm(goal[dids] - scene['robot'].data.root_pos_w[dids], dim=-1)
            ep_d0[dids] = prev_d[dids]
        obs = nobs
        rgb = nrgb
        gs += N
        if gs % 5000 < N and len(wr) > 0:
            mr = float(np.mean(wr))
            msr = float(np.mean(wsr))
            me = float(np.mean(we))
            mcr = float(np.mean(wcr))
            mt = float(np.mean(wt))
            malt = float(np.mean(wa))
            mspl = float(np.mean(wspl)) if wspl else 0.0
            t = (time.time() - t0) / 60.0
            ns = info.get('noise', 0)
            eta = info.get('icm_eta', 0)
            lc = info.get('loss_c', 0)
            la = info.get('loss_a', 0)
            li = info.get('loss_icm', 0)
            mm = info.get('mmd', 0)
            _pos = scene['robot'].data.root_pos_w
            _gd = torch.norm(goal - _pos, dim=-1)
            print(f"[{gs:>10,}]  R={mr:+8.1f}  SR={msr:.1%}  SPL={mspl:.3f}  CR={mcr:.1%}  E={me:8.0f}J  Alt={malt:.1f}m  GDist={_gd.mean():.0f}m  Stage={cur.stage}{('*' if cur.is_final_stage else ' ')}  HER={her_count:>7,}  Noise={ns:.3f}  t={t:.1f}m")
            cw.writerow([gs, mr, msr, me, mcr, ns, eta, lc, la, li, mm, cur.stage, malt, mspl])
            cf.flush()
            if writer:
                for k, v in [('SR', msr), ('SPL', mspl), ('R', mr), ('E', me), ('CR', mcr), ('ASG', mt), ('Alt', malt), ('stage', cur.stage)]:
                    writer.add_scalar(f'metrics/{k}', v, gs)
            if msr > best_sr and gs > WARMUP:
                best_sr = msr
                agent.save(ckpt / 'best.pt')
        if gs % args.save_interval < N:
            agent.save(ckpt / f'step_{gs:010d}.pt')
    agent.save(ckpt / 'final.pt')
    if writer:
        writer.close()
    cf.close()
    last = min(50, len(wr))
    metrics = {'algo': algo, 'scenario': scenario, 'scenario_name': scfg['name'], 'seed': args.seed, 'total_steps': gs, 'episodes': ep_n, 'final_SR': float(np.mean(list(wsr)[-last:])), 'best_SR': best_sr, 'final_CR': float(np.mean(list(wcr)[-last:])), 'final_reward': float(np.mean(list(wr)[-last:])), 'final_energy_J': float(np.mean(list(we)[-last:])), 'final_ASG': float(np.mean(list(wt)[-last:])), 'mean_altitude': float(np.mean(list(wa)[-last:])), 'curriculum_stage': cur.stage, 'training_min': (time.time() - t0) / 60.0, 'hyperparameters': hp}
    with open(root / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n{'═' * 65}")
    print(f'  COMPLETE | {algo.upper()} | {scenario.upper()} | seed={args.seed}')
    print(f"  Final SR   : {metrics['final_SR']:.1%}")
    print(f"  Best SR    : {metrics['best_SR']:.1%}")
    print(f"  Final CR   : {metrics['final_CR']:.1%}")
    print(f"  Energy     : {metrics['final_energy_J']:.0f} J")
    print(f"  Mean Alt   : {metrics['mean_altitude']:.1f} m")
    print(f'  Curriculum : Stage {cur.stage}')
    print(f"  Wall time  : {metrics['training_min']:.1f} min")
    print(f"{'═' * 65}\n")
    zp = root / f'{run}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for fp in root.rglob('*'):
            if fp.is_file() and str(fp) != str(zp):
                z.write(fp, fp.relative_to(root))
    print(f'[INFO] Archive → {zp}')
if __name__ == '__main__':
    main()
    sim_app.close()
