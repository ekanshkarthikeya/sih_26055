import numpy as np

class RoundRobinScheduler:
    """Sequential deterministic stepping across all sub-bands."""
    def __init__(self, num_bands=16):
        self.num_bands = num_bands
        self.current_band = 0

    def select_band(self) -> int:
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band

    def update(self, *args, **kwargs):
        pass


class RandomScheduler:
    """Memoryless uniform random channel selection."""
    def __init__(self, num_bands=16):
        self.num_bands = num_bands

    def select_band(self) -> int:
        return int(np.random.randint(0, self.num_bands))

    def update(self, *args, **kwargs):
        pass


class GreedyScheduler:
    """Exploitation-only: locks onto highest expected belief, ignores starvation."""
    def __init__(self, num_bands=16):
        self.num_bands = num_bands
        self.alphas = np.ones(num_bands)
        self.betas = np.ones(num_bands)

    def select_band(self) -> int:
        beliefs = self.alphas / (self.alphas + self.betas)
        return int(np.argmax(beliefs))

    def update(self, band: int, hits: int):
        if hits > 0:
            self.alphas[band] += min(hits, 10)
        else:
            self.betas[band] += 2.0


class EpsilonGreedyScheduler(GreedyScheduler):
    """Exploits highest belief with (1 - eps); takes unconstrained random hops with eps."""
    def __init__(self, num_bands=16, epsilon=0.15):
        super().__init__(num_bands)
        self.epsilon = epsilon

    def select_band(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.num_bands))
        return super().select_band()