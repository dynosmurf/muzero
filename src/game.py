import numpy as np
from dataclasses import dataclass
import logging

from src.prof import p, fl
from src.mcts import MonteCarloTreeSearch
from src.data_classes import Turn

def play_game(config, network, env, step, test=False):

    env.reset()
    log = GameLog()

    if test:
        temp = 0
    else:
        temp = config.visit_softmax_temperature_fn(step, config.training_steps)

    while not env.is_done() and len(env) < config.max_moves:
        
        mcts = MonteCarloTreeSearch(
                config,
                env, 
                network
                )

        mcts.execute(config.num_simulations)

        action = mcts.select_action(temp)

        result = env.step(action)

        log.update(result, mcts.get_root_visits(), mcts.get_root_value()) 
        
    if test:
        stats = env.get_metrics();
        logging.info(f"[TEST RESULT] %%%%#### steps={stats['length']} score={stats['value']} ####%%%%")      

    return log


class GameLog():

    def __init__(self, o=None):
        if o:
            self.history = [Turn(*t) for t in o[0]]
            self.child_visits = o[1]
            self.root_values = o[2]
        else:
            self.history = [] 
            self.child_visits = [] 
            self.root_values = []

    def update(self, turn, child_visits, root_value):
        self.history.append(turn)
        self.child_visits.append(child_visits)
        self.root_values.append(root_value)

    def __len__(self):
        return len(self.history)

    def serialize(self):
        return ([
                (t.action, t.reward, t.state, t.value, t.done) 
                for t in self.history
            ], 
            self.child_visits, self.root_values
            )






