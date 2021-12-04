import numpy as np

def group(y):
    y = y[y != 0]
    return np.split(y, np.where(np.diff(y) != 0)[0]+1) 

def stack(y):
    y = np.pad(y, (0,len(y) % 2), 'constant', constant_values=(0))
    y = y.reshape((2,len(y)//2))
    return y

def merge(s):
    if s.shape[1] == 0:
        return np.array([], dtype="uint8")
    v = np.apply_along_axis(np.sum, 0, s)
    return v

def score(s):
    if s.shape[1] == 0:
        return np.array([], dtype="uint8")
    s = s[:, np.apply_along_axis(np.count_nonzero, 0, s) == 2]
    if s.shape[1] == 0:
        return np.array([], dtype="uint8")
    s = np.apply_along_axis(np.sum, 0, s)
    return s

def merge_row(row):
    return np.concatenate([ merge(stack(g)) for g in group(row)])

def score_row(row):
    return np.concatenate([ score(stack(g)) for g in group(row)])

class Headless2048():

    GEN_OPTS = [2, 4]
    GEN_PROBS = [0.9, 0.1]

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, size, seed=None, track_history=False, initial_state=None):

        self.size = size
        self.rng = np.random.default_rng(seed)
        self.track_history = track_history

        self.reset();

        if not initial_state is None:
            self.state = initial_state
        else:
            self.state = np.zeros(shape=(size,size), dtype="uint8")
            # start with two tiles
            self._gen_tile()
            self._gen_tile()
           
        self.total_score = 0.0
        self.game_over = False

        self.history = [Turn(None, 0, np.copy(self.state), None, False)] 

    def reset(self):
        self.state = np.zeros(shape=(size,size), dtype="uint8")
        # start with two tiles
        self._gen_tile()
        self._gen_tile()
           
        self.total_score = 0.0
        self.game_over = False

        self.history = [Turn(None, 0, np.copy(self.state), None, False)] 

    def get_history(self):
        return self.history

    def get_reward_history(self):
        return [h.action for h in self.history]

    def get_action_history(self):
        return [h.action for h in self.history]

    def get_state(self, idx):
        return self.history[idx].state

    def get_possible_moves(self):
        moves = np.zeros(4)

        for r in range(4):
            rstate = np.rot90(self.state, k=r)

            # check if any columns are changed for this move direction
            for col in range(self.size):
                values = rstate[:, col]
                new_values = np.zeros(self.size, dtype="uint8")
                m = merge_row(values)
                new_values[:len(m)] = m 

                if not np.array_equal(new_values, values):
                    moves[r] = 1
                    continue

        return moves

    def is_done(self):
        return len(self.history) > 0 and self.history[-1].done

    def current_player(self):
        return 0

    def to_play(self, player, offset):
        # Single player game
        return 0

    def step(self, move):

        if self.game_over:
            return -1 

        pmoves = self.possible_moves()

        if pmoves[move] == 0:
            self.last_move = (move, 0, 0, np.copy(self.state))
            if self.history:
                self.history.append(self.last_move)
            return -1

        rstate = np.rot90(self.state, k=move)

        # we will work with a view of the state so we can always perform the same operations
        # creating a view is a constant time operation so should be fast
        
        score = 0.0
        for col in range(self.size):

            #first we merge consecutive
            new_values = merge_row(rstate[:, col])
            scores = score_row(rstate[:, col])
            score += np.sum(scores)

            rstate[:len(new_values), col] = new_values
            rstate[len(new_values):, col] = 0

        self.total_score += score

        self._gen_tile()

        self.last_move = (move, 1, score, np.copy(self.state))

        pmoves = self.possible_moves()

        if np.count_nonzero(pmoves) == 0:
            self._end_game()

        if self.history:
            history_entry = Turn(
                    action=move,
                    value=0,
                    state=np.copy(self.state),
                    reward=score,
                    done=self.game_over)
            
            self.history.append(history_entry)

        return score 

    def _end_game(self):
        self.game_over = True

    def _gen_tile(self):
        state = self.state
        open_tiles = np.argwhere(self.state == 0)
        if len(open_tiles) == 0:
            raise Exception('Cannot gen tile, no open spaces.')
        idx = self.rng.choice(open_tiles, 1)[0]
        v = self.rng.choice(self.GEN_OPTS, 1, p=self.GEN_PROBS)[0]
        state[idx[0]][idx[1]] = v

    def __str__(self):
        return str(self.state)

    def __len__(self):
        return len(self.history)

