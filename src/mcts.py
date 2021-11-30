from numba import types    # import the types
from numba.experimental import jitclass

import numpy as np
import math
from src.prof import p, fl

from src.util import softmax_sample
from src.networks.utils import unscale_target

class MCTSNode():

    def __init__(self, prior):

        self.prior = prior

        # default leaf values
        self.visit_count = 0 
        self.to_play = -1
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.is_leaf = True

    def is_expanded(self):
        return self.is_leaf == False

    def expand(self, to_play, actions, network_output):

        self.to_play = to_play
        self.hidden_state = network_output.hidden_state
        self.reward = network_output.reward
        self.is_leaf = False

        # set child priors scaled to be valid probability distribution
        for i, action in enumerate(actions):
            self.children[action] = MCTSNode(0.5) #network_output.policy[i])

    def __str__(self):
        return "<prior={}, visit_count={}, value_sum={}, value={}, reward={}, children={}>".format(
                self.prior, self.visit_count, self.value_sum, self.value(), self.reward, len(self.children))

    def __repr__(self):
        return str(self)

    def value(self):
        return 0.0 if self.visit_count == 0 else float(self.value_sum / self.visit_count)

def update_min_max(min_max_tuple, value):
    return (min(min_max_tuple[0], value), max(min_max_tuple[1], value))

def normalize(min_max_tuple, value):
    if min_max_tuple[1] > min_max_tuple[0]:
        return (value - min_max_tuple[0]) / (min_max_tuple[1] - min_max_tuple[0])
    else:
        return value

"""
Provided with the current network, and the actions taken 
at time t = 1...T, perform a search over possible future trajectories.
"""

class MonteCarloTreeSearch():

    def __init__(self, config, env, network):

        self.config = config
        self.env = env
        self.network = network
        self.root = MCTSNode(0) 
        self.to_play = env.current_player() 
        self.value_bounds = (float("inf"), -1 * float("inf"))

        initial_state = env.get_state(-1).reshape(self.config.input_shape)
        # h = p.start('mcts-initial_inference')
        network_output = network.initial_inference(initial_state)
        # p.stop(h)

        legal_actions = env.get_possible_moves()

        self.root.expand(self.to_play, legal_actions, network_output)
        self.add_exploration_noise(self.root)

        # fl.log(f"[MCTS] [init]\n\n    {str(self.root)}\n")

    def get_root_value(self):
        return self.root.value_sum

    def get_root_visits(self):
        visits = np.zeros(self.config.action_space_size)
        for action, child in self.root.children.items():
            visits[action] = child.visit_count
        return visits

    def execute(self, num_simulations):
        
        # print(f"\n\n[SIM] [-----------------------------------]")
        for _ in range(num_simulations):
            # print(f"\n[STEP] [{_}]")
            #fl.log(f"[MCTS] [step] [{_}]\n")
            node = self.root

            #fl.log(f"    {self.root.children[0]}\n")
            #fl.log(f"    {self.root.children[1]}\n")
            #fl.log(f"    {self.ucb_score(self.root, self.root.children[0])}\n")
            #fl.log(f"    {self.ucb_score(self.root, self.root.children[1])}\n")

            search_path = [node]

            while not node.is_leaf:
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]

            # h2 = p.start('mcts-recurrent_inference')
            # Requires network execution may need to break out of numba context here
            network_output = self.network.recurrent_inference(parent.hidden_state, action)
            # p.stop(h2)

            next_player = self.env.to_play(self.to_play, len(search_path))
            all_actions = list(range(self.config.action_space_size))

            # print(f"\n[step]: policy_logits:{network_output.policy_logits}, reward:{network_output.reward}, value:{network_output.value}")

            node.expand(next_player, all_actions, network_output)

            # fl.log(f"\n    [expand]: policy_logits:{np.array(network_output.policy_logits)}, reward:{np.array(network_output.reward)}, value:{value}")

            # fl.log(f"\n    [before backpropagate] ----------\n")
            # fl.log(str(self))

            self.backpropagate(search_path, network_output.value, next_player)

            # fl.log(f"\n    [after backpropagate] ----------\n")
            # fl.log(str(self))

        # fl.dump_log("./mcts-full-debug.out")


    def select_action(self, temp=1):
        """ 
        Select a next action based on the visit count distribution at the root of the tree
        after a search has been performed
        """
        node = self.root
        actions = np.array(list(node.children.keys()))
        visit_counts = np.array([node.children[action].visit_count for action in actions])

        _, action = softmax_sample(visit_counts, actions, temp)
        return action

    def select_child(self, node):
        """
        Selects a child to explore based on the ucb (upper confidence bound) score
        of the node and it's children
        """
        
        # TODO: clean this up
        actions = list(node.children.keys())
        scores = [self.ucb_score(node, node.children[a]) for a in actions]
        max_score = np.max(scores)
        scores = [(scores[a], a) for a in actions if scores[a] == max_score]

        if len(scores) == 0:
            raise Exception("ZERO FOUND")

        idx = np.random.choice(list(range(len(scores))))
        max_action = scores[idx][1]

        return max_action, node.children[max_action]


    def ucb_score(self, parent, child):
        """
        Upper confidence bound score used to determine next action to select
        See MuZero-Appendix B Search Paragraph 3
        """
        config = self.config

        # P(s,a) + ...
        # Note c_init = 1.25 and c_base = 19652 in paper
        pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
            config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        # Q(s, a)
        if child.visit_count > 0:
            value_score = normalize(self.value_bounds, child.reward + config.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score


    def backpropagate(self, search_path, value, to_play):
        """
        After adding a new node to the search path we traverse back 
        up the tree along the search path and update the value, 
        visit_count, and value_min_max for the tree

        See MuZero-Apendix B "Backup"
        """

        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

            self.value_bounds = update_min_max(self.value_bounds, node.reward + self.config.discount * node.value())

            value = node.reward + self.config.discount * value


    def add_exploration_noise(self, node):
        """
        Adds some noise to the policy distribution of the root node 
        to encorage exploration. Adds noise while ensuring the prior is
        greater than or equal to zero and less than or equal to 1
        """

        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


    def __str__(self):
        """
        Get a human readable string representation of the search tree
        """

        def draw_node(padding, action, node):

            actions = list(node.children.keys())
            return (
                " "*padding 
                + str(action) + ":" 
                + str(node) 
                + "".join([
                    "\n" + draw_node(padding+1, action, child) 
                    for action, child in node.children.items()
                    ]))

        return str(self.value_bounds) + "\n" + draw_node(0, "x", self.root) + "\n"



