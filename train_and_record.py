from grid_maze_env import GridMazeEnv
from mdp_wrapper import GridMazeMDP
from policy_iteration import PolicyIteration, Policy, save_policy, load_policy, save_values, load_values
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime
from pathlib import Path


GRID_SIZE = 5
TRAIN = False  # set to True to (re)train and overwrite artifacts
ARTIFACTS_DIR = Path("artifacts")
POLICY_PATH = ARTIFACTS_DIR / f"policy_g{GRID_SIZE}.json"
VALUES_PATH = ARTIFACTS_DIR / f"values_g{GRID_SIZE}.json"

env = GridMazeEnv(grid_size=GRID_SIZE, render_mode="rgb_array")
mdp = GridMazeMDP(env)
actions = [0, 1, 2, 3]

if TRAIN or not POLICY_PATH.exists():
    policy = Policy(actions=actions)
    trainer = PolicyIteration(mdp, policy, gamma=0.99, verbose=False)
    iterations = trainer.policy_iteration(max_iterations=50, theta=0.01, sample_size=100)
    print(f"Policy converged in {iterations} iterations.")
    # Save artifacts
    save_policy(policy, str(POLICY_PATH))
    # Note: values are computed during evaluation; not directly returned. Recreate final values sweep for saving.
    # Run one evaluation pass to capture values for current policy
    from policy_iteration import TabularValueFunction
    values = TabularValueFunction()
    states = mdp.get_states(sample_size=1000)
    trainer.policy_evaluation(policy, values, theta=0.01, states=states)
    save_values(values, str(VALUES_PATH))
else:
    policy = load_policy(str(POLICY_PATH), actions)
    # Values are optional for better fallback on unseen states
    if VALUES_PATH.exists():
        values = load_values(str(VALUES_PATH))
    else:
        from policy_iteration import TabularValueFunction
        values = TabularValueFunction()

# --- Record Video ---
# Create a unique subfolder and prefix per run so previous videos are never overwritten
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

def policy_agent(obs):
    state = (
        obs["agent"][0], obs["agent"][1],
        obs["target"][0], obs["target"][1],
        obs["mine1"][0], obs["mine1"][1],
        obs["mine2"][0], obs["mine2"][1]
    )
    # Prefer learned policy; if unseen, fall back to one-step greedy using saved values
    if state in policy.policy:
        return policy.policy[state]
    # Fallback: choose action with highest expected value using saved values
    best_a, best_q = None, float("-inf")
    gamma = 0.99
    for a in actions:
        q = 0.0
        for prob, next_state, reward in mdp.get_transition_prob_and_reward(state, a):
            next_val = 0.0
            # Terminal next state has value 0 by design
            if hasattr(mdp, 'is_terminal') and not mdp.is_terminal(next_state):
                next_val = values.get_value(next_state)
            q += prob * (reward + gamma * next_val)
        if q > best_q:
            best_q, best_a = q, a
    return best_a if best_a is not None else actions[0]

# Record a single episode without retraining
obs, _ = video_env.reset()
done = False
while not done:
    action = policy_agent(obs)
    obs, reward, done, _, _ = video_env.step(action)
    video_env.render()

video_env.close()
env.close()

