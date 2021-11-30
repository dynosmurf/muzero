import time
import numpy as np
from multiprocessing import Pool
from redis import Redis
from walrus import *
import logging

from src.replay_buffer import ReplayBuffer
from src.network_storage import NetworkStorage
from src.game import play_game
from src.prof import p, fl
from src.networks.utils import LRSched

logging.basicConfig(level=logging.INFO)


def run_selfplay(config, replay_buffer, network_storage, test=False):
    import tensorflow as tf
    network = config.network_factory()

    # wait for a weight set to be made available
    while len(network_storage) == 0:
        time.sleep(0.5)
    
    step = 0
    while True:
        try:
            weights = network_storage.latest_network()
            network.set_weights([np.array(w) for w in weights])

            # TODO: Use reset here instead of new env
            env = config.env_factory()

            game_log = play_game(config, network, env, step, test=test)

            replay_buffer.save_game(game_log)
            step += 1

        except Exception as e:
            logging.error(f"Selfplay error:\n{e}", exc_info=True)


def log_step(step, metrics, replay_buffer):
    buffer_info = f"replay_buffer_size={len(replay_buffer)} avg_len={replay_buffer.avg_len(10)} last_len={replay_buffer.last_len()}"
    train_info = f"loss={metrics['loss']} value_loss={metrics['value_loss']}"

    logging.info(f"[{step}] {buffer_info} :: {train_info}")


def train_network(config, network_storage, replay_buffer):
    import tensorflow as tf

    network = config.network_factory() 

    # TODO: encapsulate
    lr_schedule = LRSched(config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    network.compile(optimizer)

    # save initial network
    network_storage.save_network(0, network)

    # wait for the first replay to be available
    while len(replay_buffer) == 0:
        time.sleep(0.5)

    for step in range(config.training_steps):

        try:

            if step % config.checkpoint_interval == 0:
                network_storage.save_network(step, network)
            
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            
            metrics = network.train_step(batch, config.weight_decay)

            log_step(step, metrics, replay_buffer)

        except Exception as e:
            logging.error(f"Training error:\n{e}", exc_info=True)

    network_storage.save_network(config.training_steps, network)


def get_storage(config):
    # TODO: Move into replay and network_storage classes
    db = Database(host='muzero-redis', port=6379, db=10)
    weights = db.Hash('weights')
    replays = db.List('replays') 
    stats = db.List('stats')

    replay_buffer = ReplayBuffer(config, replays, stats)
    network_storage = NetworkStorage(config, weights)

    return replay_buffer, network_storage


def muzero(config):

    global wrapped_run_selfplay

    def wrapped_run_selfplay(i, test=False):
        logging.info(f"Starting worker {i}")

        replay_buffer, network_storage = get_storage(config)

        run_selfplay(config, replay_buffer, network_storage, test)

    pool = Pool(processes=config.num_actors + 1)

    # start test worker
    pool.apply_async(wrapped_run_selfplay, [0, True])

    # start self_play workers
    for i in range(config.num_actors):
        pool.apply_async(wrapped_run_selfplay, [i+1, False])



    replay_buffer, network_storage = get_storage(config)

    train_network(config, network_storage, replay_buffer)

    pool.close()

