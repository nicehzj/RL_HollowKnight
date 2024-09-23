import gym
import numpy as np
import enum
import random
import time
import pyautogui # 自动化窗口操作

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.


class Actions(enum.Enum):
    @classmethod
    def random(cls):
        return random.choice(list(cls))


class Move(Actions):
    NO_OP = 0
    HOLD_LEFT = 1
    HOLD_RIGHT = 2


class Attack(Actions):
    NO_OP = 0
    ATTACK = 1


class Displacement(Actions):
    NO_OP = 0
    TIMED_SHORT_JUMP = 1
    TIMED_LONG_JUMP = 2
    DASH = 3
    SUPER_DASH = 4

    
class HollowKnightEnv(gym.Env):
    """
    Hollow Knight Environment with gym API
    """
    
    # 按键映射
    KEY_MAPS = {
        Move.HOLD_LEFT: 'a',
        Move.HOLD_RIGHT: 'd',
        Attack.ATTACK: 'j',
        Displacement.TIMED_SHORT_JUMP: 'space',
        Displacement.TIMED_LONG_JUMP: 'space',
        Displacement.DASH: 'k',
        Displacement.SUPER_DASH: 'l',
    }

    # 奖励映射
    REWARD_MAPS = {
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        Attack.ATTACK: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        Displacement.DASH: 0,
        Displacement.SUPER_DASH: 0,
    }

    HP_CKPT = np.array([], dtype=int) # HP
    ACTIONS = [Move, Attack, Displacement]

    def __init__(self, obs_shape=(160, 160), rgb=False, gap=0.165,
                 w1=.8, w2=.8, w3=-0.0001):
        """
        :param obs_shape: the shape of observation returned by step and reset
        :param w1: the weight of negative reward when being hit
                (for example, w1=1. means give -1 reward when being hit)
        :param w2: the weight of positive reward when hitting enemy
                (for example, w2=1. means give +1 reward when hitting enemy)
        :param w3: the weight of positive reward when not hitting nor being hit
                (for example, w3=-0.0001 means give -0.0001 reward when neither happens
        """
        self.monitor = self._find_window()
        self.holding = []
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self.prev_action = -1
        total_actions = np.prod([len(Act) for Act in self.ACTIONS])
        if rgb:
            obs_shape = (3,) + obs_shape
        else:
            obs_shape = (1,) + obs_shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                dtype=np.uint8, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(int(total_actions))
        self.rgb = rgb
        self.gap = gap
        self._prev_time = None

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self._hold_time = 0.2
        self._fail_hold_rew = -1e-4
        self._timer = None
        self._episode_time = None

    @staticmethod
    def _find_window():
        """
        find the window with title 'Hollow Knight'
        """
        window = pyautogui.getWindowsWithTitle('Hollow Knight')
        assert len(window) == 1, f'found {len(window)} windows called Hollow Knight {window}'
        window = window[0]
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()
        window.moveTo(0, 0)
        
        geo = None
        conf = 0.95
        while geo is None:
            geo = pyautogui.locateOnScreen('locator/geo.png', confidence=conf, grayscale=True, region=(0, 0, 250, 250))
            conf = max(0.92, conf*0.999)
            time.sleep(0.1)

        loc = {
            'left': geo.left - 36,
            'top': geo.top - 97,
            'width': 1020,
            'height': 692
        }
        return loc

    def step(self, action):
        # Implement this method
        pass

    def reset(self):
        # Implement this method
        pass

    def render(self):
        # Implement this method if needed
        pass


if __name__ == '__main__':
    env = HollowKnightEnv()
    loc = env._find_window()
    print(loc)
    pyautogui.moveTo(loc['left'], loc['top'])
    # pyautogui.moveTo(loc['left'] + loc['width'], loc['top'])
    pyautogui.moveTo