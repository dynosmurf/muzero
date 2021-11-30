from numba import njit, prange
import numpy as np

@njit(cache=True)
def softmax_sample(counts, actions, temp):
    if temp == 0:
        i = np.argmax(counts)
        return counts[i], actions[i]
     
    p = 1/temp
    powers = counts**p
    total_sum = np.sum(powers)
    probabilities = powers / total_sum
    i = np.argmax(np.random.multinomial(1, probabilities, 1))
    return probabilities[i], actions[i]   





