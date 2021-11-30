import cProfile, pstats, io
from pstats import SortKey
import numpy as np
import time


def cpu_profile(method):
    def timed(*args, **kw):
        pr = cProfile.Profile()
        pr.enable()
        result = method(*args, **kw)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumtime').print_stats()
        print(s.getvalue())

        return result    
    return timed


def profile(method, t='cpu'):
    def wrapped(*args, **kw):
        if t == None:
            return method(*args, **kw)
        profiler = cpu_profile
        result = profiler(method)(*args, **kw)
        return result
    return wrapped


class ProfTimer():

    def __init__(self, context=None):
        now = time.perf_counter()
        self.context = context
        self.data = {}

    def start(self, _id):
        now = time.perf_counter()
        return (_id, now)

    def stop(self, handle):
        now = time.perf_counter()
        _id, start = handle

        self.data[_id] = self.data[_id] if _id in self.data else []
        self.data[_id].append(now - start)

    def dump_log(self, path):

        with open(path, 'a') as f:

            f.write(str(self))

    def __str__(self):
        out = ""
        for _id, times in self.data.items():
            context = self.context or ''
            out += f'{context}:{_id}:{len(times)}, {np.mean(times)}\n'

        return out


class FileLogger():

    def __init__(self):
        self._log = []
        pass

    def log(self, message):
        self._log.append(message)

    def dump_log(self, path):
        with open(path, 'a') as f:
            f.write('\n'.join(self._log))
        self._log = []


p = ProfTimer()
fl = FileLogger()

