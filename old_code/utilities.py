import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from itertools import combinations, chain, tee
from pprint import pprint

def permutations(size, exclude_min, exclude_max):
    minimum = int(exclude_min)
    maximum = int(2**size)-int(exclude_max)

    return np.array([
        list(f'{x:0{size}b}') 
        for x in range(minimum, maximum)
    ]).astype(int)


def expand_signal(signal):
    assert signal[0]<=2, "Signal's first value invalid!"
    signal_tail = signal[1:]
    if signal[0]==0:
        return np.array([[0,0,*signal_tail]])
    elif signal[0]==1:
        return np.array([[1,0,*signal_tail],
                         [0,1,*signal_tail]])
    else:
        return np.array([[1,1,*signal_tail]])


def simplify_lang(language):
    return np.row_stack([expand_signal(sig) for sig in language])

    
def create_languages(max_num_signals, add_silence=True):
    """
    creates all the languages with length at most 'max length'
    only excluding the system without any signal
    """
    
    
    # for each language, the indices of the signals in that language
    # (a chained version of the combination generators from 1 to max_num_signals)
    # TODO: this is not elegant, but I use the generator twice
    permut1, permut2 = tee(chain(*[
        combinations(np.arange(len(SIGNALS)), up_to)
        for up_to in range(1, max_num_signals+1)
    ]))

    language_complexities = np.array([
        np.sum(COMPLEXITIES[np.array(indices)])
        for indices in permut1
    ])

    # synthetic because it's still using the 0/1/2 notation
    # for the first element
    languages = [
        simplify_lang(SIGNALS[np.array(indices)])
        # signals[np.array(indices)]
        for indices in permut2
    ]

    if add_silence:
        # also add the silence signal to every language
        languages = np.array([
            np.row_stack(([[1, 1, 1, 1]], language))
            for language in languages
        ])

    return languages, language_complexities


def log_softmax(inputarray, alpha, axis=-1):
    array = np.array(inputarray) * alpha
    e_x = np.where(
        array == -np.inf,
        0, np.exp(array - np.max(array, axis=axis, keepdims=True))
    )
    return np.log(e_x / np.sum(e_x, axis=axis, keepdims=True))


def lognormalize(array, axis=1):
    a = logsumexp(array, axis=axis, keepdims=True)
    return array - a


def softmax(array, alpha, axis=0):
    unnorm = np.exp(alpha*array)
    return normalize(unnorm, axis=axis)


def normalize(array, axis):
    return array / np.sum(array, axis=axis, keepdims=True)


def random_choice_prob_index(a, axis=1):
    """
    copied this from a stackoverflow question
    'vectorizing np.random.choice for given 2d array of probabilities
    along axis'
    """
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def calculate_logprob_production(alpha, observations, languages):
    """
    Parameters
    ----------
    observations: array
        Array with shape (# observations, 4)
    Returns
    -------
    list
        A list with # languages len, each element 
        is an array with shape (#observations, # signal
    """

    logprobs_production = []
    for language in languages:

        # list with shape (# languages, # observations, 
        # # signals in that language, 4)
        # which says for each combination of signal and observation
        # if the signal is incompatible with the observation (-1)
        # compatible with the observation (0)
        # or compatible but not observed (1)
        # print("1: ", language[None].shape)
        # print("2: ", observations[:,None].shape)
        diff = language[None] - observations[:,None]

        # those signal/obs combos that are incompatible
        # get distance inf
        # shape (# observations, # signals)

        difference_temp = np.where(
            np.any(diff==-1, axis=2),
            np.inf,
            np.sum(diff, axis=2)
        )

        # shape (# observations, # signals)
        logprobs_production.append(
            log_softmax(difference_temp, alpha=-0.5, axis=1)
        )
    return logprobs_production


def speaker(language, num_observations, alpha=-3):
    """
    Parameters
    ----------
    language: array
        An array with shape (# signals, 4) modelling a single language
    num_observations: int
        And int specifying how many observations the speaker has to make
    alpha: float
        Strength of tendency for precise knowledge
    """
    
    number_covered_areas = np.sum(SIGNALS_EXPANDED, axis=1)

    observations_logprobs = log_softmax(
        number_covered_areas, alpha, axis=0
    )

    observations_indices = np.random.choice(
        len(SIGNALS_EXPANDED),
        size=num_observations, 
        p=np.exp(observations_logprobs)
    )

    observations = SIGNALS_EXPANDED[observations_indices]

    logprobabilities_production = logprob_production(
        alpha=alpha,
        observations=observations,
        languages=language[None]
    )

    # choose one signal for each row of probabilities_production
    choices_indices = random_choice_prob_index(
        np.exp(logprobabilities_production), axis=1
    )

    return observations, logprobabilities_production, choices_indices


def learner(observations, logprobabilities_production, 
        choices_indices, logprior):
    """
    Parameters
    ----------
    observations: array
        1d array with the observations
    logprobabilities_production: list of arrays
        list containing for each language 
        a 2d array with shape (# observations, # signals)
        containing the probability that the speaker
        would produce each signal given each observation in 'observations'
    choices_indices: array
        1d array with the indices of the signals that the speaker sent
    logprior: array
        
    """
    # calculate the logprobability of all the choices for
    # the given observations
    loglikelihood = [
        np.sum(logprob_array[observations, choices_indices])    
        for logprob_array in logprobabilities_production
    ]



# creates all possible boolean strings of length 4, 
# excluding the degenerate ones
# the 4 elements are: p-q, q-p, p&q, not(p or q)
# signals = permutations(4, True, True)

# the first element indicates the number of 1s
# in the first two elements (if 1, the message should
# split into (0, 1, ..) and (1, 0, ..)
SIGNALS, COMPLEXITIES = zip(
    ([0, 0, 1], 5), 
    ([0, 1, 0], 7),
    ([0, 1, 1], 21),
    ([1, 0, 0], 7),
    ([1, 0, 1], 3),
    ([1, 1, 0], 3),
    ([1, 1, 1], 11),
    #  these four are
    # 'reflections' (from the pow of p and q)
    # of the previous four
    # ([1, 0, 0, 0], 7),
    # ([1, 0, 0, 1], 3),
    # ([1, 0, 1, 0], 3),
    # ([1, 0, 1, 1], 11),
    ([2, 0, 0], 17), 
    ([2, 0, 1], 11), 
    ([2, 1, 0], 5)
)

SIGNALS = np.array(SIGNALS)
SIGNALS_EXPANDED = np.array(simplify_lang(SIGNALS))
COMPLEXITIES = np.array(COMPLEXITIES)

languages, language_complexities = create_languages(5) 
# speaker(languages[10], 4)
observations_possible = permutations(
        4, exclude_min=True, exclude_max=False)
print(observations_possible.shape)
probs = [
    np.exp(x) for x in calculate_logprob_production(
        3,
        observations=observations_possible,
        languages=languages
)]
pprint(probs)
