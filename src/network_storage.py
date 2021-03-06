import msgpack
import msgpack_numpy as m
from walrus import Database 

# needed for msgpack_numpy to work correctly
m.patch()


class NetworkStorage():

    def __init__(self, config, db):
        self.db = db 
        self._networks = self.db.Hash('weights') 
        self.config = config


    def latest_network(self):
        if len(self._networks) > 0:
            keys = [int(key) for key in self._networks.keys()]
            max_key = max(keys) 
            weights = msgpack.unpackb(self._networks[max_key])
            return weights 
        else:
            return None 


    def __len__(self):
        return len(self._networks)


    def save_network(self, step, network):
        self._networks[step] = msgpack.packb(network.get_weights())
