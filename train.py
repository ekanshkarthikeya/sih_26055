import os
import numpy as np
from rich.console import Console
from rich.progress import track

from src.utils.turing_loader import TuringDatasetLoader
from src.env.gym_ew_env import EWReceiverGymEnv
from src.schedulers.dqn_agent import DQNAgent

def main():
    console = Console()
    console.print("\n[bold cyan]=== DRDO EW SMART SCAN: REINFORCEMENT LEARNING TRAINER ===[/bold cyan]\n")

    loader = TuringDatasetLoader(num_bands=16)
    
    # Check if official Turing train parquet was downloaded; fall back if not
    train_parquet = "data/stare/train_0.parquet"
    if os.path.exists(train_parquet):
        console.print(f"[green]Loading official Turing dataset: {train_parquet}[/green]")
        df = loader.load_from_parquet(train_parquet)
    else:
        console.print("[yellow]Using generated Turing-standard stare train for training...[/yellow]")
        df = loader.load_or_generate(duration_sec=1.0)

    # Fast training slice (first 50,000 pulses)
    train_df = df.iloc[:50000].copy()

    env = EWReceiverGymEnv(train_df, num_bands=16, dwell_time_sec=50e-6)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)
    num_episodes = 25

    console.print(f"State Dimension: {state_dim} | Action Dimension: {action_dim}")
    console.print(f"Training for {num_episodes} episodes...\n")

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        total_hits = 0
        loss_val = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, terminated)
            
            loss = agent.train_step()
            if loss > 0:
                loss_val = loss

            state = next_state
            ep_reward += reward
            total_hits += info["hits"]

            if terminated or truncated:
                break

        if ep % 3 == 0:
            agent.update_target_network()

        p_int = (total_hits / env.total_pulses) * 100.0
        console.print(
            f"Episode {ep:02d}/{num_episodes} | "
            f"Reward: {ep_reward:8.1f} | "
            f"Hits: {total_hits:5d} ({p_int:5.2f}%) | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Loss: {loss_val:.4f}"
        )

    agent.save("models/dqn_ew_scheduler.pt")
    console.print("\n[bold green]Training complete! Model saved to models/dqn_ew_scheduler.pt[/bold green]\n")

if __name__ == "__main__":
    main()