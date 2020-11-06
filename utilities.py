from functools import reduce, lru_cache
from types import FunctionType
from itertools import product
import numpy as np


def bit_not(n, numbits=4):
    return (1<<numbits)-1-n


def combine(f_1, f_2):
    """
    Substitute the first argument in f_1 for f_2
    """
    return f_1.replace('_',f_2,1)


def compose(m_1, m_2):
    """
    This function puts m_2 in place of the first argument of m_1
    m_1 is a function 
    m_2 can be either a function or a bit strin
    """
    # print('m_1: ', m_1, ' m_2: ', m_2)
    if type(m_2) == FunctionType:
        n_args = find_n_args(m_2)
        if n_args == 1:
            breakpoint()
            return lambda j: m_1(m_2(j))
        elif n_args == 2:
            return lambda j: lambda k: m_1(m_2(j)(k))
    else:
        return m_1(m_2)


def find_n_args(f):
    """
    Finds the number of arguments of a curried function
    """
    count = 0
    while True:
        try:
            f = f(1)
            count += 1
        except TypeError:
            return count


def find_shortest_formulas(m_dict):
    """
    Since formulas are added to saturated in order of complexity,
    whenever I encounter a meaning that's already in 
    saturated, I don't need to add it to saturated
    Because the new meaning is going to be at least
    as complex as the one that's already in saturated
    Otherwise, add to saturated
    NOTE: this relies on the fact that the shortest
    formula for a meaning is a combination of shortest formulas 
    NOTE: lexicon should contain first p and q, and then
    the other terms in order of complexity
    """
    saturated, saturated_m, unsaturated, unsaturated_m, keys = [], [], [], [], []
    for k,v in m_dict.items():
        if type(v)==FunctionType:
            n_args = find_n_args(v)
            unsaturated_m.append(v)
            unsaturated.append(k + '(' + ','.join(['_']*n_args) + ')')
        else:
            saturated.append(k)
            saturated_m.append(v)
        keys.append(k)

    # store the formulas and meanings for 
    # each of the unique expressions
    # store unique formulas the you already know the meaning of here
    unique_formulas = [*saturated]
    # store the relative meanings here 
    # NOTE: in the same order
    unique_meanings = [*saturated_m]
    # store the lexicon (which contains the expressions
    # with parentheses like 'not(_)')
    lexicon = [*saturated, *unsaturated]
    lexicon_m = [*saturated_m, *unsaturated_m]
    # start with just the initial unsaturated in it
    # unsaturated terminal nodes
    terminal_nodes = [*unsaturated]
    terminal_m = [*unsaturated_m]
    while len(unique_meanings) < 16:
        new_nodes = []
        new_m = []
        # loop through the terminal nodes
        for current_node, current_m in zip(terminal_nodes,terminal_m):
                # loop through the lexicon
                # and try to add each of the saturated
                for n, m in zip(lexicon, lexicon_m):
                    new_node = combine(current_node, n)
                    new_meaning = compose(current_m, m)
                    # when encountering a saturated node, 
                    # check if it is in unique already
                    if '_' not in new_node:  
                        # new_node_meaning = evaluate_formula(new_node,m_dict)
                        if new_meaning not in unique_meanings:
                            unique_formulas.append(new_node)
                            unique_meanings.append(new_meaning)
                    else:
                        new_nodes.append(new_node)
                        new_m.append(new_meaning)
        terminal_nodes = new_nodes
        terminal_m = new_m
    return [f'{x:04b}' for x in unique_meanings], unique_formulas


if __name__=='__main__':

    m_dict = {
        'p': 0b1100,
        'q': 0b1010,
        # 'or': lambda x,y: x|y,
        # 'and': lambda x,y: x&y,
        # 'not': lambda x: bit_not(x),
        'nand': lambda x: lambda y: bit_not(x&y)
    }
    # print(bin(evaluate_formula('or(not(p),q)', m_dict)))
    print(np.column_stack(find_shortest_formulas(m_dict)))
