import numpy as np
from src.mcts import MonteCarloTreeSearch
from src.prof import p, fl
import cProfile
import pstats
from dataclasses import dataclass

def play_game(config, network, env, step, test=False):

    env.reset()
    
    log = GameLog()
    e = 6
    if test:
        temp = 0
    else:
        temp = config.visit_softmax_temperature_fn(step, config.training_steps)

    print("[GAME] TEMP: ", test, temp)
    while not env.is_done() and len(env) < config.max_moves:
        
        h = p.start('mcts')
        #with cProfile.Profile() as pr:
        mcts = MonteCarloTreeSearch(
                config,
                env, 
                network
                )

        mcts.execute(config.num_simulations)

        #stats = pstats.Stats(pr)
        #stats.sort_stats(pstats.SortKey.TIME)
        #stats.dump_stats(filename=f'mcts_prof{len(env)}.prof')

        p.stop(h)
        p.dump_log('./prof1.prof')
        #fl.log(network.get_model_summary())
        #fl.dump_log('./debug.out')

        # TODO: config.get temp

        action = mcts.select_action(temp)

        result = env.step(action)
        log.update(result, mcts.get_root_visits(), mcts.get_root_value()) 


        
        fl.log(str(mcts) + "\n")
        fl.dump_log('./mmz-mcts-debug.out')

    if test:
        print(f"[TEST RESULT] %%%%#### steps={len(env.history)} ####%%%%")      

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
        return ([(t.action, t.reward, t.state, t.value, t.done) for t in self.history], self.child_visits, self.root_values)


@dataclass
class Turn:
    action: int
    reward: float
    state: np.ndarray
    value: float
    done: bool




