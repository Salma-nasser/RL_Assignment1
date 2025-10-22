import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.wrappers import RecordVideo
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import random


class GridMazeEnv(gym.Env):
  
  def __init__(self, grid_size=5, render_mode="human"):
    self.size=grid_size
    self.grid_size = grid_size
    self.action_space = spaces.Discrete(4)
    self._agent_location = np.array([-1, -1], dtype=np.int32)
    self._target_location = np.array([-1, -1], dtype=np.int32)
    self.observation_space = gym.spaces.Dict(
      {
          "agent": gym.spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
          "target": gym.spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
          "mine1": gym.spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
          "mine2": gym.spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
      }
    )
    
    self.cell_size = 60  # pixels per grid cell
    self.width = self.cell_size * self.grid_size
    self.height = self.cell_size * self.grid_size
    self.render_mode = render_mode
    self.metadata = {"render_fps": 4}

    self.screen = None
    self.clock = None
    self.running = False

  def render(self):
    """Render the environment.
    - If render_mode == 'rgb_array': returns an RGB numpy array (H, W, 3)
    - If render_mode == 'human': draws using pygame window
    """
    # Colors
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    BLACK = np.array([0, 0, 0], dtype=np.uint8)
    RED = np.array([220, 20, 60], dtype=np.uint8)
    GREEN = np.array([34, 139, 34], dtype=np.uint8)
    BLUE = np.array([30, 144, 255], dtype=np.uint8)

    # Prepare positions
    agent = tuple(self.agent_pos.tolist()) if hasattr(self, 'agent_pos') else (0, 0)
    goal = tuple(self.goal_pos.tolist()) if hasattr(self, 'goal_pos') else (self.grid_size-1, self.grid_size-1)
    mines = getattr(self, 'mines', set())

    if self.render_mode == "rgb_array":
      # Create white canvas
      img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

      # Draw grid lines
      for i in range(1, self.grid_size):
        y = i * self.cell_size
        x = i * self.cell_size
        img[y-1:y+1, :, :] = 0
        img[:, x-1:x+1, :] = 0

      # Helper to fill a cell
      def fill_cell(pos, color):
        r, c = pos
        y0, y1 = r * self.cell_size, (r + 1) * self.cell_size
        x0, x1 = c * self.cell_size, (c + 1) * self.cell_size
        img[y0:y1, x0:x1, :] = color

      # Draw entities (order: mines, goal, agent)
      for m in mines:
        fill_cell(m, RED)
      fill_cell(goal, GREEN)
      fill_cell(agent, BLUE)

      return img

    # Human rendering with pygame
    if self.render_mode == "human":
      if self.screen is None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid Maze")
        self.clock = pygame.time.Clock()

      self.screen.fill((255, 255, 255))

      # Draw grid
      for i in range(1, self.grid_size):
        pygame.draw.line(self.screen, (0, 0, 0), (0, i * self.cell_size), (self.width, i * self.cell_size), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.height), 2)

      # Helper to draw a filled rect
      def draw_cell(pos, color):
        r, c = pos
        rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)

      for m in mines:
        draw_cell(m, (220, 20, 60))
      draw_cell(goal, (34, 139, 34))
      draw_cell(agent, (30, 144, 255))

      pygame.display.flip()
      if self.clock:
        self.clock.tick(self.metadata.get("render_fps", 4))
      return None

    # If an unsupported mode requested
    raise NotImplementedError
  
  def reset(self, seed=None, options=None):
      super().reset(seed=seed)
      # Sample 4 unique positions from the grid
      flat_positions = random.sample(range(self.grid_size ** 2), 4)
      coords = [np.array([p // self.grid_size, p % self.grid_size], dtype=np.int32) for p in flat_positions]

      self.agent_pos, self.goal_pos, mine1, mine2 = coords
      self.mines = {tuple(mine1), tuple(mine2)}

      obs = {
          "agent": self.agent_pos,
          "target": self.goal_pos,
          "mine1": mine1,
          "mine2": mine2
      }
      return obs, {}
  
  def step(self, action):
    moves = {
        0: np.array([0, 1]),   # Right
        1: np.array([-1, 0]),  # Up
        2: np.array([0, -1]),  # Left
        3: np.array([1, 0])    # Down
    }
    intended = action
    perpendicular = {
        0: [1, 3],  # Right -> Up or Down
        1: [0, 2],  # Up -> Right or Left
        2: [1, 3],  # Left -> Up or Down
        3: [0, 2]   # Down -> Right or Left
    }
    actual_action = random.choices(
      [intended]+perpendicular[intended], weights=[0.7, 0.15, 0.15],k=1)[0]

    new_pos  = self.agent_pos + moves[actual_action]
    new_pos = np.clip(new_pos, 0, self.size - 1)  #stay within bounds
    self.agent_pos = new_pos
    
    reached_goal = np.array_equal(self.agent_pos, self.goal_pos)
    hit_mine = tuple(self.agent_pos) in self.mines
    
    if reached_goal:
      reward = 1.0
      done = True
    elif hit_mine:
      reward = -10.0
      done = True
    else:
      reward = -0.01
      done = False
    obs = {
        "agent": self.agent_pos,
        "target": self.goal_pos,
        "mine1": np.array(list(self.mines)[0]),
        "mine2": np.array(list(self.mines)[1])
    }
    return obs, reward, done, False, {}
