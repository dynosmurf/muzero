from src.replay_buffer import ReplayBuffer
from src.network_storage import NetworkStorage
from src.game import play_game
import time
import numpy as np
from multiprocessing import Manager, Pool, set_start_method
from pprint import pprint
from src.prof import p, fl
from src.networks.utils import LRSched
import traceback
import cProfile
import pstats
from redis import Redis
from pottery import RedisList, RedisDict
from walrus import *

def run_selfplay(config, replay_buffer, network_storage, test=False):
    import tensorflow as tf
    network = config.network_factory()

    # wait for a weight set to be made available
    while len(network_storage) == 0:
        time.sleep(0.5)
    
    step = 0
    while True:
        try:
            h = p.start('set-weights')
            weights = network_storage.latest_network()
            network.set_weights([np.array(w) for w in weights])
            p.stop(h)

            env = config.env_factory()
            game_log = play_game(config, network, env, step, test=test)

            replay_buffer.save_game(game_log)
            step += 1
        except Exception as e:
            print("[ERROR] selfplay issue")
            print(e)
            traceback.print_exc()
            print(sys.exc_info()[2])



def train_network(config, network_storage, replay_buffer):
    import tensorflow as tf

    network = config.network_factory() 

    lr_schedule = LRSched(config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    network.compile(optimizer)

    network_storage.save_network(0, network)

    # wait for the first replay to be available
    while len(replay_buffer) == 0:
        time.sleep(0.5)

    for i in range(config.training_steps):

        try:
            h = p.start('train-step')
            print(">>> [TRAIN][{}] replay_buffer_size={} avg_len={} last_len={}".format(
                i, len(replay_buffer.buffer), replay_buffer.avg_len(), replay_buffer.last_len()))

            if i % 250 == 0:
                network_storage.save_network(i, network)
            
            h2 = p.start('sample-batch')
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)

            p.stop(h2)
            #try:
            network.train_step(batch, config.weight_decay)

            p.stop(h)
            p.dump_log('./prof1.prof')
            #except Exception as e:
            #    print("[ERROR] train step issue")
            #    pprint(batch)
            #    print(e)
        except Exception as e:
            print("[ERROR] train issue")
            print(e)
            traceback.print_exc()
            print(sys.exc_info()[2])

    network_storage.save_network(config.training_steps, network)

def get_storage(config):
    db = Database(host='muzero-redis', port=6379, db=8)
    weights = db.Hash('weights')
    replays = db.List('replays') 
    stats = db.List('stats')

    replay_buffer = ReplayBuffer(config, replays, stats)
    network_storage = NetworkStorage(config, weights)

    return replay_buffer, network_storage



def muzero(config):

    global wrapped_run_selfplay


    def wrapped_run_selfplay(i, test=False):
        print("[INFO] Starting worker {}".format(i))

        replay_buffer, network_storage = get_storage(config)

        run_selfplay(config, replay_buffer, network_storage, test)

    pool = Pool(processes=3)

    # start test worker
    pool.apply_async(wrapped_run_selfplay, [0, True])

    # start self_play workers
    for i in range(2):
        pool.apply_async(wrapped_run_selfplay, [i+1, False])



    replay_buffer, network_storage = get_storage(config)

    train_network(config, network_storage, replay_buffer)

    """
    with cProfile.Profile() as pr:
        train_network(config, network_storage, replay_buffer, 1)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='./prof_vis1.prof')
    """

    pool.close()

