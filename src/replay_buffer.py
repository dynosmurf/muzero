import numpy as np
from dataclasses import dataclass
import msgpack
import msgpack_numpy as m
from walrus import Database 
import tensorflow as tf

from src.game import GameLog
from src.data_classes import Batch

# this is required for msgpack_numpy to work
m.patch()


def random_action(action_space_size):
    return np.random.choice(list(range(action_space_size)))


def normalize(l):
    t = np.sum(l, axis=-1)
    return l / t


class ReplayBuffer(object):

    def __init__(self, config, db):
        self.db = db
        self.config = config
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = self.db.List('replays')
        self.stats = self.db.List('stats')
        self.db['step'] = 0


    def save_game(self, game, game_metrics, is_test):
        if len(self.buffer) > self.window_size:
            self.buffer.popleft()
            self.stats.popleft()
        self.buffer.append(msgpack.packb(game.serialize()))
        self.stats.append(msgpack.packb(game_metrics))


    def set_step(self, step):
        self.db['step'] = step


    def log_game_metrics(self, writer, metrics, is_test):
        step = int(self.db['step'])
        with writer.as_default():
            for key, value in metrics.items():
                if is_test:
                    tf.summary.scalar('test-'+key, value, step=step)
                else:
                    tf.summary.scalar(key, value, step=step)
                writer.flush()


    def avg_stat(self, key, length=None):
        if len(self) == 0:
            return 0
        elif length is None:
            return np.mean([int(msgpack.unpackb(s)[key]) for s in self.stats])
        else:
            stats = [int(msgpack.unpackb(s)[key]) for s in self.stats]
            return np.mean(np.array(stats)[len(stats) - length:])


    def last_stat(self, key):
        return int(msgpack.unpackb(self.stats[-1])[key])


    def sample_batch(self, num_unroll_steps, td_steps):
        game_logs = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in game_logs]

        observations = np.zeros((self.batch_size,) + self.config.input_shape, dtype="float32")
        actions = np.zeros((self.batch_size, num_unroll_steps), dtype="int32")

        # we want on extra value for each of these as we will have one
        # value from the initial_inference and then `unroll_steps` from recurrent
        unroll_shape = (self.batch_size, num_unroll_steps + 1)
        # note rewards in one shorter here
        rewards = np.zeros(unroll_shape, dtype="float32")
        values = np.zeros(unroll_shape, dtype="float32")
        policy_probs = np.zeros(unroll_shape + (self.config.action_space_size,), dtype="float32")
    
        # generate batch
        for i, (g, ig) in enumerate(game_pos):
            observations[i] = g.history[ig].state.reshape(self.config.input_shape)

            # we want actions taken after state[ig] to after state[ig + unroll_steps]
            actions[i] = self.get_actions(g, ig + 1)

            # we want values, policy_probs at state[ig]
            targets = self.get_targets(g, ig)
    
            values[i] = targets[0]
            rewards[i] = targets[1]
            rewards[i][0] = 0 # we want the length to match but the first reward to contribute 0 loss
            policy_probs[i] = targets[2]

        return Batch(
                observations=observations,
                actions=actions,
                rewards=rewards,
                values=values, 
                policy_probs=policy_probs)


    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        idx = np.random.choice(range(len(self.buffer)))
        return GameLog(msgpack.unpackb(self.buffer[int(idx)]))


    def sample_position(self, game):
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(np.arange(len(game)))


    def get_actions(self, game, idx):
        unroll_steps = self.config.num_unroll_steps
        action_space_size = self.config.action_space_size

        actions = np.empty(unroll_steps, dtype="int32")

        for i in range(unroll_steps):
            ig = idx + i
            actions[i] = game.history[ig].action if ig < len(game) else random_action(action_space_size)

        return actions


    def get_targets(self, game, idx):

        action_space_size = self.config.action_space_size
        unroll_steps = self.config.num_unroll_steps
        td_steps = self.config.td_steps
        discount = self.config.discount
    
        values = []
        reward_out = []
        policies = []

        rewards = [h.reward for h in game.history] 

        for i in range(idx, idx + unroll_steps + 1):

            bootstrap_index = i + td_steps

            if bootstrap_index < len(game.root_values):
                value = game.root_values[bootstrap_index] * discount**td_steps
            else:
                value = 0

            for j, reward in enumerate(rewards[i:bootstrap_index]):
                value += reward * discount**i

            if i > 0 and i <= len(rewards):
                last_reward = rewards[i - 1]
            else:
                last_reward = 0

            if i < len(game.root_values):
                values.append(value)
                reward_out.append(last_reward)
                policies.append(normalize(game.child_visits[i]))
            else:
                # past the end of the game we append "absorbing states"
                # note we specify a uniform policy
                values.append(0)
                reward_out.append(last_reward)
                policies.append([1/action_space_size for i in range(action_space_size)])

        return values, reward_out, policies


    def __len__(self):
        return len(self.buffer)



