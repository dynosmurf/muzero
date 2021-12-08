import unittest
import random
import numpy as np

from src.config import *
from src.data_classes import NetworkOutput
from src.mcts import *

from test.mocks import *

seed = 2021
random.seed(seed)
np.random.seed(seed)

def log_notes(enable, *args):
    if enable:
        print(*args)

class TestMCTS(unittest.TestCase):

    def get_test_inst(self, network_output=None):
        o = False 
        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)
        config = self.get_test_config()
        network = MockNetwork(network_output)

        mcts = MonteCarloTreeSearch(config, env, network) 
        return mcts, rng

    def get_test_config(self):
        return Config(
                (1,4,4),
                (1,4,4),
                action_space_size=2,
                max_moves=10,
                discount=0.997,
                dirichlet_alpha=0.25,
                num_simulations=10,
                batch_size=10,
                td_steps=10,
                num_actors=1,
                lr_init=0.05,
                lr_decay_steps=350e3,
                visit_softmax_temperature_fn=lambda i: 1 
                )

    #
    # MCTSNode.expand 
    #------------------------------ 

    def test_node(self):

        n = MCTSNode(1)


    def test_expand_root(self):
        o = False 
        network_output = [NetworkOutput(value=10, hidden_state=np.array([[1.1]]), reward=1, policy=np.array([0.3, 0.7]))]
        inst, rng = self.get_test_inst(network_output)

        log_notes(o, inst.root.children[0])
        log_notes(o, inst.root.children[1])
        child1, child2 = (inst.root.children[0], inst.root.children[1])
        
        self.assertEqual(child1.visit_count, 0)
        self.assertEqual(child1.value(), 0)
        self.assertEqual(child1.reward, 0)

        self.assertEqual(child2.visit_count, 0)
        self.assertEqual(child2.value(), 0)
        self.assertEqual(child2.reward, 0)


    def test_expand_level_1(self):
        o = False
        network_output = [
                NetworkOutput(value=10, hidden_state=np.array([[1.1]]), reward=1, policy=np.float32([0.3, 0.7])),
                NetworkOutput(value=11, hidden_state=np.array([[1.1]]), reward=2, policy=np.float32([0.8, 0.2]))
                ]
        inst, rng = self.get_test_inst(network_output)

        next_player = 0
        parent = inst.root.children[0]

        self.assertEqual(len(parent.children), 0)

        parent.expand(next_player, [0, 1], network_output[1])

        self.assertEqual(len(parent.children), 2)

        child1, child2 = (parent.children[0], parent.children[1])

        self.assertEqual(parent.reward, 2)

        self.assertEqual(child1.prior, 0.5)
        self.assertEqual(child1.visit_count, 0)
        self.assertEqual(child1.value(), 0)
        self.assertEqual(child1.reward, 0)

        self.assertEqual(child2.prior, 0.5)
        self.assertEqual(child2.visit_count, 0)
        self.assertEqual(child2.value(), 0)
        self.assertEqual(child2.reward, 0)

    #
    # MCTS.select_child
    #------------------------------ 

    def test_select_child_root(self):

        # Should randomly select one of the child nodes from the
        # root
        o = False 
        inst, rng = self.get_test_inst()


        for i in range(10):
            count_a = np.random.randint(0,51)
            count_b = 50 - count_a
            
            inst.root.children[0].visit_count = count_a 
            inst.root.children[1].visit_count = count_b 
            samples = [inst.select_action(1) for i in range(10000)]
            max_count = max(count_a, count_b)
            
            err = np.mean(samples) - count_b / 50

            log_notes(o, f"count_a={count_a}, count_b={count_b}")
            log_notes(o, f"{np.mean(samples)} vs {count_b / 50}")
            log_notes(o, err)
            self.assertLess(err, 0.02)

    #
    # MCTS.select_action
    #------------------------------ 

    def test_select_action_temp_1(self):
        """
        When temp = 1 should select actions using a probability
        distribution equal to the visit_count of each child over
        total visits.
        """
        o = False 
        inst, rng = self.get_test_inst()


        for i in range(10):
            count_a = np.random.randint(0,51)
            count_b = 50 - count_a
            
            inst.root.children[0].visit_count = count_a 
            inst.root.children[1].visit_count = count_b 
            samples = [inst.select_action(1) for i in range(10000)]
            max_count = max(count_a, count_b)
            
            err = np.mean(samples) - count_b / 50

            log_notes(o, f"count_a={count_a}, count_b={count_b}")
            log_notes(o, f"{np.mean(samples)} vs {count_b / 50}")
            log_notes(o, err)
            self.assertLess(err, 0.01)

    def test_select_action_temp_0(self):
        """
        When temp = 0 should greedily select actions only
        from the child with the most visits.
        """
        o = False 
        inst, rng = self.get_test_inst()
        temp = 0

        for i in range(10):
            count_a = np.random.randint(0, 51)
            count_b = 50 - count_a
            
            inst.root.children[0].visit_count = count_a 
            inst.root.children[1].visit_count = count_b 
            samples = [inst.select_action(temp) for i in range(10000)]
            max_count = max(count_a, count_b)

            max_action = np.argmax([count_a, count_b])
            
            err = np.mean(samples) - max_action 
            
            log_notes(o, f"count_a={count_a}, count_b={count_b}")
            log_notes(o, f"{np.mean(samples)} vs {max_action}")
            log_notes(o, err)
            self.assertLess(err, 0.05)


    #
    # MCTS.add_exploration_noise
    #------------------------------ 

    def test_add_exploration_noise(self):
        o = False 
        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)
        env.get_possible_moves = lambda : [0,1]
        config = self.get_test_config()
        network_output = [NetworkOutput(value=10, hidden_state=[[1.1]], reward=1, policy=np.float32([0.3, 0.7]))]
        network = MockNetwork(network_output)

        mcts = MonteCarloTreeSearch(config, env, network) 

        # confirm that the priors of the root node children differ 
        # from the policy produced by the network at the root

        possible_moves = env.get_possible_moves()
        log_notes(o, "Possible moves: ", possible_moves)

        policy = network.initial_inference(env.get_state(-1)).policy
        log_notes(o, "Network output policy: ", policy)

        root_child_priors = [child.prior for action, child in mcts.root.children.items()]
        log_notes(o, "After noise: ", root_child_priors)

        assert np.sum(root_child_priors) - 1 < 10**-8
        assert not np.allclose(root_child_priors, list(policy)) 


    #
    # MCTS.ucb_score
    #------------------------------ 

    def test_ucb_initial(self):
        """
        With no visits the ucb_score should be zero for each child
        """
        inst, rng = self.get_test_inst()

        root_scores = [inst.ucb_score(inst.root, child) for child in inst.root.children.values()]

        self.assertEqual(root_scores, [0, 0])

    def test_ucb_even_root_branches(self):
        o = False 
        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)
        config = self.get_test_config()
        n_out_1 = NetworkOutput(value=10, hidden_state=np.float32([[1.1]]), reward=1, policy=np.float32([0.5, 0.5]))
        network = MockNetwork([n_out_1])

        inst = MonteCarloTreeSearch(config, env, network) 

        root_scores = [inst.ucb_score(inst.root, child) for child in inst.root.children.values()]

        log_notes(o, str(root_scores))
        
        self.assertEqual(root_scores, [0, 0])

    def test_ucb_uneven_root_branches(self):
        o = False
        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)
        config = self.get_test_config()
        n_out_1 = NetworkOutput(value=10, hidden_state=np.float32([[1.1]]), reward=1, policy=np.float32([0.6, 0.4]))
        network = MockNetwork([n_out_1])

        inst = MonteCarloTreeSearch(config, env, network) 

        root_scores = [inst.ucb_score(inst.root, child) for child in inst.root.children.values()]

        log_notes(o, str(root_scores))
        log_notes(o, [child.prior for child in inst.root.children.values()])
       
        # If there are no visits the root score should be 0 even if 
        self.assertEqual(root_scores, [0, 0])

    def test_ucb_score_initial(self):
        o = False 
        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)
        config = self.get_test_config()
        n_out_1 = NetworkOutput(value=0, hidden_state=np.array([[1.1]]), reward=1, policy=np.float32([0, 0.2, 0.3, 0.5]))
        network = MockNetwork([n_out_1])

        mcts = MonteCarloTreeSearch(config, env, network) 

        parent = MCTSNode(0)
        parent.expand(0, [1,2], n_out_1)

        log_notes(o, "Parent children: ", [(action, child.prior) for action, child in parent.children.items()])

        # initial scores should be 0 as parent and children have visit_count = 0
        initial_scores = [mcts.ucb_score(parent, parent.children[action]) for action in parent.children.keys()]
        log_notes(o, "Initial child scores: ", initial_scores)
        assert np.allclose(initial_scores, [0, 0])
  
    #
    # MCTS.execute
    #------------------------------ 

    def test_execute_multiple(self):
        o = False 
        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)
        config = self.get_test_config()
        config.action_space_size = 4
        n_out_1 = NetworkOutput(value=4, hidden_state=[[1.1]], reward=1, policy=np.float32([0, 0.2, 0.3, 0.5]))
        n_out_2 = NetworkOutput(value=3, hidden_state=[[1.1]], reward=0, policy=np.float32([0, 0.2, 0.3, 0.5]))
        network = MockNetwork([n_out_1, n_out_2])

        mcts = MonteCarloTreeSearch(config, env, network) 
        log_notes(o, "Initial tree state: \n{}".format(str(mcts)))

        # execute a single simulation
        log_notes(o, "--------- Step 1 ---------")
        mcts.execute(1)
        log_notes(o, "Tree after step: \n{}".format(str(mcts)))
    
        root = mcts.root
        selected_child = root.children[0]

        level_1_scores = [mcts.ucb_score(root, root.children[action]) for action in root.children.keys()]
        level_2_scores = [mcts.ucb_score(selected_child, selected_child.children[action]) for action in selected_child.children.keys()]
        log_notes(o, "Root scores after step: \n{}".format(level_1_scores))
        log_notes(o, "Level 1 scores after step: \n{}".format(level_2_scores))

        log_notes(o, "--------- Step 2 ---------")
        mcts.execute(1)
        log_notes(o, "Tree after step: \n{}".format(str(mcts)))
    
        root = mcts.root
        selected_child = root.children[1]

        level_0_scores = [mcts.ucb_score(root, root.children[action]) for action in root.children.keys()]
        level_1_scores = [mcts.ucb_score(selected_child, selected_child.children[action]) for action in selected_child.children.keys()]
        selected_child = selected_child.children[3]
        level_2_scores = [mcts.ucb_score(selected_child, selected_child.children[action]) for action in selected_child.children.keys()]
        log_notes(o, "Root scores after step: \n{}".format(level_0_scores))
        log_notes(o, "Level 1 scores after step: \n{}".format(level_1_scores))
        log_notes(o, "Level 2 scores after step: \n{}".format(level_2_scores))

        log_notes(o, "--------- Step 3 ---------")
        mcts.execute(1)
        log_notes(o, "Tree after step: \n{}".format(str(mcts)))
    
        root = mcts.root
        level_0_scores = [mcts.ucb_score(root, root.children[action]) for action in root.children.keys()]
        selected_child = root.children[1]
        level_1_scores = [mcts.ucb_score(selected_child, selected_child.children[action]) for action in selected_child.children.keys()]

        selected_child_1 = selected_child.children[2]
        level_2_scores_1 = [mcts.ucb_score(selected_child_1, selected_child_1.children[action]) for action in selected_child_1.children.keys()]

        selected_child_2 = selected_child.children[3]
        level_2_scores_2 = [mcts.ucb_score(selected_child_2, selected_child_2.children[action]) for action in selected_child_2.children.keys()]
        
        log_notes(o, "Root scores after step: \n{}".format(level_0_scores))
        log_notes(o, "Level 1 scores after step: \n{}".format(level_1_scores))
        log_notes(o, "Level 2 scores after step: \n{}\n{}".format(level_2_scores_1, level_2_scores_2))

        log_notes(o, "-------- Other Info ---------")
        log_notes(o, "Value Bounds: {}".format(mcts.value_bounds))


    def test_execute_multiple_2(self):

        config = self.get_test_config()

        action_history = [1,2,3,0,1]

        outputs = [
            NetworkOutput(value=0, hidden_state=[[1.1]], reward=0, policy={0: 0.8342013, 1: 0.16579875}),
            NetworkOutput(value=0.6121330261230469, hidden_state=[[1.1]], reward=0.28712379932403564, policy=np.array([0, 0.2, 0.3, 0.5])),
            NetworkOutput(value=0.5946230888366699, hidden_state=[[1.1]], reward=0.1633901596069336, policy=np.array([0, 0.2, 0.3, 0.5])),
            NetworkOutput(value=0.5646584033966064, hidden_state=[[1.1]], reward=0.14456570148468018, policy=np.array([0, 0.2, 0.3, 0.5])),
            NetworkOutput(value=0.6538723707199097, hidden_state=[[1.1]], reward=0.32829177379608154, policy=np.array([0, 0.2, 0.3, 0.5]))
            ]
        network = MockNetwork(outputs) 

        rng = np.random.default_rng(2021)
        env = MockEnvWrapper(rng)


        mcts = MonteCarloTreeSearch(
                config, env, network
                )

        # override the noise to set the priors explicity
        mcts.root.children[0].prior = 0.6285985525662817
        mcts.root.children[1].prior = 0.37140149213720186

        # override select_child so that the first selection is deterministic
        _select_child = mcts.select_child
        def mock_select_child(node):
            scores = [mcts.ucb_score(node, node.children[a]) for a in node.children.keys()]
            if np.sum(scores) == 0:
                return 1, node.children[1]
            else:
                return _select_child(node)

        mcts.select_child  = mock_select_child

        mcts.execute(4)

        self.assertEqual(mcts.root.value_sum, 3.9112387827422617)
        self.assertEqual(mcts.root.children[0].value_sum, 0.6538723707199097)
        self.assertEqual(mcts.root.children[1].value_sum, 2.075892534971237)

        self.assertEqual(mcts.root.children[0].visit_count, 1)
        self.assertEqual(mcts.root.children[1].visit_count, 3)


if __name__ == '__main__':
    unittest.main()
