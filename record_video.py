from grid_maze_env import GridMazeEnv
from mdp_wrapper import GridMazeMDP
from policy_iteration import load_policy, load_values
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime
from pathlib import Path

# Configuration
GRID_SIZE = 5
ARTIFACTS_DIR = Path("artifacts")
POLICY_PATH = ARTIFACTS_DIR / f"policy_g{GRID_SIZE}.json"
VALUES_PATH = ARTIFACTS_DIR / f"values_g{GRID_SIZE}.json"
ACTIONS = [0, 1, 2, 3]
GAMMA = 0.99


def main():
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Missing policy file: {POLICY_PATH}. Run training first (set TRAIN=True in train_and_record.py)."
        )

    policy = load_policy(str(POLICY_PATH), ACTIONS)
    values = load_values(str(VALUES_PATH)) if VALUES_PATH.exists() else None

    env = GridMazeEnv(grid_size=GRID_SIZE, render_mode="rgb_array")
    mdp = GridMazeMDP(env)

    # Create a unique subfolder/name so no overwrite happens
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    video_dir = os.path.join("videos", f"run_{run_id}")
    os.makedirs(video_dir, exist_ok=True)
    video_env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda e: True,
        name_prefix=f"policy_iter_{run_id}",
        disable_logger=True,
    )

    def select_action(obs):
        state = (
            obs["agent"][0], obs["agent"][1],
            obs["target"][0], obs["target"][1],
            obs["mine1"][0], obs["mine1"][1],
            obs["mine2"][0], obs["mine2"][1]
        )
        # Use learned policy when available
        if state in policy.policy:
            return policy.policy[state]
        # Fallback: one-step greedy with values (if available)
        if values is not None:
            best_a, best_q = None, float("-inf")
            for a in ACTIONS:
                q = 0.0
                for prob, next_state, reward in mdp.get_transition_prob_and_reward(state, a):
                    next_val = 0.0
                    if hasattr(mdp, 'is_terminal') and not mdp.is_terminal(next_state):
                        next_val = values.get_value(next_state)
                    q += prob * (reward + GAMMA * next_val)
                if q > best_q:
                    best_q, best_a = q, a
            if best_a is not None:
                return best_a
        # Ultimate fallback
        return ACTIONS[0]

    # Record a single episode
    obs, _ = video_env.reset()
    done = False
    while not done:
        action = select_action(obs)
        obs, reward, done, _, _ = video_env.step(action)
        video_env.render()

    video_env.close()
    env.close()


if __name__ == "__main__":
    main()
