from src.networks.network import NetworkOutput
import numpy as np

class MockNetwork():

    def __init__(self, outputs=None):
        self.iter = 0
        if outputs == None:
            self.outputs = [NetworkOutput(value=5, reward=1, policy=np.array([0.1, 0.2, 0.3, 0.4]), hidden_state=[[0,self.iter,1]])]
        else:
            self.outputs = outputs

    def initial_inference(self, observation, no_grad=True):

        out = self.outputs[self.iter % len(self.outputs)]
        self.iter += 1
        return out


    def recurrent_inference(self, hidden_state, action, no_grad=True):
        out =  self.outputs[self.iter % len(self.outputs)] 
        self.iter += 1
        return out


class MockEnvWrapper():

    def __init__(self, rng):
        self.rng = rng
        self.history = [(None, 0, self.rng.choice([0,1], size=(4,4)), None, False)]
        pass

    def reset(self):
        pass

    def get_history(self):
        return self.history

    def get_reward_history(self):
        return [h[3] for h in self.history]

    def get_action_history(self):
        return [h[0] for h in self.history]

    def get_state(self, idx):
        # assert abs(idx) < len(self.history)
        return self.history[idx][2]

    def get_possible_moves(self):
        return self.rng.choice([0, 1, 2, 3], size=2)

    def is_done(self):
        return len(self.history) > 10

    def current_player(self):
        return self.to_play(0, len(self.history))

    def to_play(self, curr_player, offset):
        return 1 if len(self.history) + curr_player + offset % 2 == 0 else 0

    def step(self, action):
        player_turn = self.to_play()
        # action taken, player who took action, resulting state, reward, is game over
        result = (action, player_turn, self.rng.choice([0,1], size=(4,4)), self.rng.choice([0,1]), len(self.history) + 1 > 10)
        self.history.append(result)
        return result

    def __len__(self):
        return len(self.history)


