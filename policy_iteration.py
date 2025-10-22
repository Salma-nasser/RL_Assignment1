import random
import json
import os

class TabularValueFunction:
    def __init__(self):
        self.values = {}

    def get_value(self, state):
        return self.values.get(state, 0.0)

    def add(self, state, value):
        self.values[state] = value

    def get_q_value(self, mdp, state, action, gamma=0.99):
        q = 0.0
        for prob, next_state, reward in mdp.get_transition_prob_and_reward(state, action):
            # Terminal states have value 0
            next_value = 0.0 if mdp.is_terminal(next_state) else self.get_value(next_state)
            q += prob * (reward + gamma * next_value)
        return q
class QTable:
    def __init__(self, alpha=1.0):
        self.q = {}

    def update(self, state, action, value):
        self.q[(state, action)] = value

    def get_argmax_q(self, state, actions):
        return max(actions, key=lambda a: self.q.get((state, a), 0.0))
class Policy:
    def __init__(self, actions):
        self.policy = {}  # state → action
        self.actions = actions

    def select_action(self, state, actions):
        # Return a deterministic action for unseen states to keep policy fixed during evaluation
        if state not in self.policy:
            # Default to the first available action to avoid stochasticity during evaluation
            self.policy[state] = actions[0]
        return self.policy[state]

    def update(self, state, action):
        self.policy[state] = action

    # ---- Serialization helpers ----
    def to_list(self):
        """Return a JSON-serializable list of [state_list, action]."""
        return [[list(state), action] for state, action in self.policy.items()]

    @staticmethod
    def from_list(items, actions):
        p = Policy(actions)
        for state_list, action in items:
            p.policy[tuple(state_list)] = action
        return p
class PolicyIteration:
    def __init__(self, mdp, policy, gamma=0.99, verbose=False):
        self.mdp = mdp
        self.policy = policy
        self.gamma = gamma
        self.verbose = verbose

    def policy_evaluation(self, policy, values, theta=0.001, states=None, max_eval_iters=2000):
        if states is None:
            states = self.mdp.get_states()
        
        iteration = 0
        while True:
            iteration += 1
            delta = 0.0
            new_values = TabularValueFunction()
            for state in states:
                # Calculate the value of V(s)
                actions = self.mdp.get_actions(state)
                old_value = values.get_value(state)
                new_value = values.get_q_value(
                    self.mdp, state, policy.select_action(state, actions), self.gamma
                )
                new_values.add(state, new_value)  # Use new_values instead of values
                delta = max(delta, abs(old_value - new_value))

            # Update values after completing the full sweep
            values = new_values
            
            # terminate if the value function has converged
            if delta < theta:
                if self.verbose:
                    print(f"  Policy evaluation converged in {iteration} iterations (delta={delta:.6f})")
                break
            
            if iteration >= max_eval_iters:
                if self.verbose:
                    print(f"  Stopping policy evaluation after {iteration} iterations (delta={delta:.6f})")
                break

            if self.verbose and iteration % 50 == 0:
                print(f"  Policy evaluation iteration {iteration}, delta={delta:.4f}")

        return values

    """ Implmentation of policy iteration iteration. Returns the number of iterations executed """

    def policy_iteration(self, max_iterations=100, theta=0.001, sample_size=100):

        # create a value function to hold details
        values = TabularValueFunction()
        
        # Generate states once and reuse them
        if self.verbose:
            print(f"Generating {sample_size} sample states...")
        states = self.mdp.get_states(sample_size=sample_size)
        if self.verbose:
            print(f"Generated {len(states)} states. Starting policy iteration...")

        # Initialize a deterministic starting policy over the sampled states
        for s in states:
            if s not in self.policy.policy:
                self.policy.update(s, self.mdp.get_actions(s)[0])

        for i in range(1, max_iterations + 1):
            if self.verbose:
                print(f"\nPolicy Iteration {i}:")
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta, states)
            for state in states:

                actions = self.mdp.get_actions(state)
                old_action = self.policy.select_action(state, actions)

                q_values = QTable(alpha=1.0)
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = values.get_q_value(self.mdp, state, action, self.gamma)
                    q_values.update(state, action, new_value)
                # V(s) = argmax_a Q(s,a)
                new_action = q_values.get_argmax_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)
                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )

            if not policy_changed:
                if self.verbose:
                    print(f"Policy converged. No changes in iteration {i}")
                return i
            else:
                if self.verbose:
                    print(f"  Policy changed, continuing...")

        return max_iterations


# ---- JSON save/load utilities ----
def save_policy(policy: Policy, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'items': policy.to_list()}, f)

def load_policy(path: str, actions):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Policy.from_list(data.get('items', []), actions)

def save_values(values: TabularValueFunction, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # store as list of [state_list, value]
    items = [[list(state), float(val)] for state, val in values.values.items()]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'items': items}, f)

def load_values(path: str) -> TabularValueFunction:
    tv = TabularValueFunction()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for state_list, val in data.get('items', []):
        tv.add(tuple(state_list), float(val))
    return tv
