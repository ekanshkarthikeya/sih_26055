import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EWReceiverGymEnv(gym.Env):
    """
    Gymnasium-compatible Electronic Support Receiver Environment.
    State: [Belief (K), Starvation (K), Normalized LO Frequency (1)]
    Action: Choose Sub-band [0 .. K-1]
    Reward: +Threat Weight on Hit, -Hop Penalty for large PLL frequency jumps
    """
    def __init__(self, pdw_df, num_bands=16, dwell_time_sec=50e-6):
        super().__init__()
        self.df = pdw_df.sort_values('toa').reset_index(drop=True)
        self.num_bands = num_bands
        self.dwell_time = dwell_time_sec
        
        self.toas = self.df['toa'].to_numpy()
        self.bands = self.df['band_idx'].to_numpy()
        self.threats = self.df['threat'].to_numpy()
        self.total_pulses = len(self.df)

        # Action: tune to one of the K sub-bands
        self.action_space = spaces.Discrete(self.num_bands)
        
        # Observation space: 
        # [0..K-1]: Sub-band activity beliefs [0, 1]
        # [K..2K-1]: Normalized starvation time [0, 1]
        # [2K]: Current tuner frequency [0, 1]
        obs_dim = 2 * self.num_bands + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        starvation = np.clip((self.current_time - self.last_visited) / 0.05, 0.0, 1.0)
        norm_band = np.array([self.current_band / float(self.num_bands)], dtype=np.float32)
        return np.concatenate([self.belief, starvation, norm_band]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.current_band = 0
        self.belief = np.full(self.num_bands, 0.5, dtype=np.float32)
        self.last_visited = np.zeros(self.num_bands, dtype=np.float32)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        from_band = self.current_band
        to_band = action
        
        # PLL settling delay
        hop_dist = abs(to_band - from_band)
        settle_sec = (5.0 + 15.0 * np.log2(1.0 + hop_dist)) * 1e-6 if hop_dist > 0 else 0.0
        
        dwell_start = self.current_time + settle_sec
        dwell_end = dwell_start + self.dwell_time

        # Slice ground truth
        i_start = np.searchsorted(self.toas, dwell_start)
        i_end = np.searchsorted(self.toas, dwell_end)

        hits = 0
        reward = 0.0
        if i_end > i_start:
            window_bands = self.bands[i_start:i_end]
            valid_hits = (window_bands == action)
            hits = int(np.sum(valid_hits))
            if hits > 0:
                reward = float(np.sum(self.threats[i_start:i_end][valid_hits]))

        # Synthesizer switching penalty: discourage unnecessary frequency hopping
        hop_cost = 0.1 * (hop_dist / float(self.num_bands))
        step_reward = reward - hop_cost if hits > 0 else -0.05 - hop_cost

        # Belief updates
        self.last_visited[action] = self.current_time
        if hits > 0:
            self.belief[action] = 0.95
        else:
            self.belief[action] = 0.05

        # Passive decay for unobserved channels
        for k in range(self.num_bands):
            if k != action:
                self.belief[k] = self.belief[k] * 0.90 + 0.05

        self.current_time = dwell_end
        self.current_band = action
        self.step_count += 1
        
        terminated = self.current_time >= self.toas[-1] or self.step_count >= 15000
        truncated = False
        info = {"hits": hits, "dwell_start": dwell_start}

        return self._get_obs(), step_reward, terminated, truncated, info