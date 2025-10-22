import random
import numpy as np

class GridMazeMDP:
    def __init__(self, env):
        self.env = env
        self.grid_size = env.size

    def get_states(self, sample_size=5000):
        states = set()
        while len(states) < sample_size:
            flat_positions = random.sample(range(self.grid_size ** 2), 4)
            coords = [(p // self.grid_size, p % self.grid_size) for p in flat_positions]
            state = tuple([*coords[0], *coords[1], *coords[2], *coords[3]])
            states.add(state)
        return list(states)


    def get_actions(self, state):
        return [0, 1, 2, 3]  # Right, Up, Left, Down
    
    def is_terminal(self, state):
        """Check if a state is terminal (goal reached or mine hit)"""
        ax, ay, gx, gy, m1x, m1y, m2x, m2y = state
        agent_pos = (ax, ay)
        goal_pos = (gx, gy)
        mines = {(m1x, m1y), (m2x, m2y)}
        return agent_pos == goal_pos or agent_pos in mines
    
    def get_transition_prob_and_reward(self, state, action):
        ax, ay, gx, gy, m1x, m1y, m2x, m2y = state
        grid_size = self.grid_size
        
        # If current state is terminal, no transitions (absorbing state)
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]

        # Define movement vectors
        moves = {
            0: (0, 1),   # Right
            1: (-1, 0),  # Up
            2: (0, -1),  # Left
            3: (1, 0)    # Down
        }

        # Define stochastic outcomes
        perpendiculars = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [0, 2]
        }

        outcomes = [
            (0.7, action),
            (0.15, perpendiculars[action][0]),
            (0.15, perpendiculars[action][1])
        ]

        transitions = []
        for prob, actual_action in outcomes:
            dx, dy = moves[actual_action]
            new_ax = max(0, min(grid_size - 1, ax + dx))
            new_ay = max(0, min(grid_size - 1, ay + dy))

            new_state = (new_ax, new_ay, gx, gy, m1x, m1y, m2x, m2y)
            agent_pos = (new_ax, new_ay)
            goal_pos = (gx, gy)
            mines = {(m1x, m1y), (m2x, m2y)}

            # Determine reward
            if agent_pos == goal_pos:
                reward = 1.0
            elif agent_pos in mines:
                reward = -1.0
            else:
                reward = -0.01

            transitions.append((prob, new_state, reward))

        return transitions
