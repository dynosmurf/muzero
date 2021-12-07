import gym
from src.game import Turn

class CartpoleEnvWrapper():

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps = 5000
        self.env.reset()
        self.history = [Turn(None, 0, self.env.state, None, False)]

    def reset(self):
        self.env.reset()

    def get_history(self):
        return self.history

    def get_reward_history(self):
        return [h.action for h in self.history]

    def get_action_history(self):
        return [h.action for h in self.history]

    def get_state(self, idx):
        return self.history[idx].state

    def get_possible_moves(self):
        # can always move left and right
        return [0, 1]

    def is_done(self):
        return len(self.history) > 0 and self.history[-1].done

    def current_player(self):
        return 0

    def to_play(self, player, offset):
        # Single player game
        return 0

    def step(self, action):
        result = self.env.step(action) 
        history_entry = Turn(action=action, value=0, state=result[0], reward=result[1], done=result[2])
        self.history.append(history_entry)
        return history_entry 

    def get_metrics(self):
        metrics = {
            "value": len(self),
            "length": len(self)
            }
        return metrics

    def __len__(self):
        return len(self.history)
