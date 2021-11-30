import msgpack
import msgpack_numpy as m
m.patch()

class NetworkStorage():

    def __init__(self, config, shared_dict):
        self._networks = shared_dict 
        self.config = config

    def latest_network(self):
        if len(self._networks) > 0:
            keys = [int(key) for key in self._networks.keys()]
            max_key = max(keys) 
            weights = msgpack.unpackb(self._networks[max_key])
            print(f"[LOAD] [{max_key}] w={weights[0][1][1]}")
            return weights 
        else:
            return None 

    def __len__(self):
        return len(self._networks)

    def save_network(self, step, network):
        print(f"[SAVE] [{step}] w={network.get_weights()[0][1][1]}")
        self._networks[step] = msgpack.packb(network.get_weights())
