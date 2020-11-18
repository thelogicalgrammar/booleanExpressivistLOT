from re import split
from functools import reduce, lru_cache
from itertools import combinations
from pprint import pprint
from types import FunctionType
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


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
        # n_args = find_n_args(m_2)
        # if n_args == 1:
        #     return lambda j: m_1(m_2(j))
        # elif n_args == 2:
        #     return lambda j: lambda k: m_1(m_2(j)(k))
        n_args_m1 = find_n_args(m_1)
        n_args_m2 = find_n_args(m_2)
        # htere I'm being very explicit but there's probably
        # a much more elegant way
        if n_args_m1 == 1 and n_args_m2 == 2:
            return lambda x: lambda y: m_1(m_2(x)(y))
        elif n_args_m1 == 2 and n_args_m2 == 2:
            return lambda x: lambda y: lambda z: m1(m2(y)(z))(x)
        else:
            # n_args_2 == 1  is the case of e.g. negation but we
            # decided to not have unary operators in the lexicon
            raise InputError('Invalid number of arguments')
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
                # print("proposed: ", current_node, n)
                new_meaning = compose(current_m, m)
                # when encountering a saturated node, 
                # check if it is in unique already
                # NOTE: not equivalent to nested ifs
                if '_' not in new_node:
                    if new_meaning not in unique_meanings:  
                        unique_formulas.append(new_node)
                        unique_meanings.append(new_meaning)
                else:
                    new_nodes.append(new_node)
                    new_m.append(new_meaning)
        # new_nodes, new_m = reduce_symmetric_operators(new_nodes, new_m)
        terminal_nodes = new_nodes
        terminal_m = new_m
    return [f'{x:04b}' for x in unique_meanings], unique_formulas


def calculate_for_all_functionally_complete():
    m_dict = {
        'p': 0b1100,
        'q': 0b1010,
        # 'F': 0b0000,
        # 'T': 0b1111,
        # 'not': lambda x: bit_not(x),
        'or': lambda x: lambda y: x|y,
        'and': lambda x: lambda y: x&y,
        'nand': lambda x: lambda y: bit_not(x&y),
        'nor': lambda x: lambda y: bit_not(x|y),
        '->': lambda x: lambda y: bit_not(x)|y,
        '<-': lambda x: lambda y: bit_not(y)|x,
        'n->': lambda x: lambda y: bit_not(bit_not(x)|y),
        'n<-': lambda x: lambda y: bit_not(bit_not(y)|x),
        '<->': lambda x: lambda y: (x&y)|(bit_not(x)&bit_not(y)),
        'n<->': lambda x: lambda y: bit_not((x&y)|(bit_not(x)&bit_not(y)))
    }
    
    f_complete = [
        # one element
        ('nor',),
        ('nand',),
        # two elements
        # ('or', 'not'),
        # ('and', 'not'),
        # ('->', 'not'),
        # ('<-', 'not'),
        # ('->', 'F'),
        # ('<-', 'F'),
        ('->', 'n<->'),
        ('<-', 'n<->'),
        ('->', 'n->'),
        ('->', 'n<-'),
        ('<-', 'n->'),
        ('<-', 'n<-'),
        # ('n->', 'not'),
        # ('n<-', 'not'),
        # ('n->', 'T'),
        # ('n<-', 'T'),
        ('n->', '<->'),
        ('n<-', '<->'),
        # three elements
        # ('or', '<->', 'F'),
        ('or', '<->', 'n<->'),
        # ('or', 'n<->', 'T'),
        # ('and', '<->', 'F'),
        ('and', '<->', 'n<->'),
        # ('and', 'n<->', 'T')
    ]
    # calculate all combinations of signals
    # of up to 5 elements
    # which are supersets of at least one of the sets in f_complete
    f_all = []
    binary_names = [
        'or', 'and', 'nand', 'nor', '->', '<-', 'n->', 'n<-', '<->', 'n<->']
    for i in range(1,6):
        for signals_combination in combinations(binary_names, r=i):
            # print([sig in sig_f_complete for sig_f_complete in f_complete])
            if any([
                all([sig in signals_combination for sig in sig_complete_tuple])
                for sig_complete_tuple in f_complete]):
                f_all.append(signals_combination)

    pprint(f_all)
    dictionary_complete = {}
    for complete_set in f_all:
        print(complete_set)
        m_dict_restricted = {k:m_dict[k] for k in complete_set}
        m_dict_restricted.update({
            'p': m_dict['p'],
            'q': m_dict['q']
        })
        shortest = find_shortest_formulas(m_dict_restricted)
        print(np.column_stack(shortest),'\n')
        dictionary_complete[complete_set]=np.column_stack(shortest)
    return dictionary_complete


def reduce_symmetric_operators(nodes, meanings):
    """
    Remove the leafs that are redundant according to the
    symmetric operators.
    Parameters
    ----------
    nodes: list of strings
        List of strings, each string being one compositionally obtained
        language.
    meanings: list 
        List of meaning objects, corresponding to the strings in nodes.
    """
    symmetric_dict = {
        # 'F': 0b0000,
        # 'T': 0b1111,
        # 'not': lambda x: bit_not(x),
        'p': False,
        'q': False,
        '_': False,
        'or': True,
        'and': True,
        'nand': True,
        'nor': True,
        '->': False,
        '<-': False,
        'n->': False,
        'n<-': False,
        '<->': True,
        'n<->': True
    }

    essential_nodes = []
    essential_meanings = []
    for node, meaning in zip(nodes, meanings):
        # check that the node 
        divided_node = split('\(|\)|,', node)
        symmetric_operators = [
            symmetric_dict[a] 
            for a in divided_node
            if a != ''
        ]
        # the test is that there should not be
        # two symmetric operators in a row
        symmetric_in_a_row = any([
            symmetric_operators[i] and symmetric_operators[i+1]
            for i in range(0,len(symmetric_operators)-1)
        ])
        if not symmetric_in_a_row:
            essential_nodes.append(node)
            essential_meanings.append(meaning)
    return essential_nodes, essential_meanings


def calculate_expected_complexity(dict_complete):
    """
    Calculates expected complexity for each lang,
    where complexity is calculated as the number of operations 
    in the formula, found as the number of '('.
    """
    expected_complexity_dict = {}
    for tuple_primitives, shortest_defs in dict_complete.items():
        expected_complexity_dict[tuple_primitives] = np.sum([
            s.count('(') for s in shortest_defs[:,1]
        ])/len(shortest_defs)
    return expected_complexity_dict


def calculate_language_complexity(lang_dict):
    complexities = {
        'and': 1,
        'or': 1,
        'nor': 1,
        'nand': 2,
        'n<->': 3,
        '<->': 3,
        '->': 2,
        '<-': 2,
        'n->': 1,
        'n<-': 1
    }

    complexity = {
        key: sum([complexities[k] for k in key])
        for key,_ in lang_dict.items()
    }

    return complexity


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

    # dict_complete = calculate_for_all_functionally_complete()
    # dict_complexities_3 = calculate_expected_complexity(dict_complete)
    # pprint(list(sorted(dict_complexities_3.items(), key=lambda x:x[1])))
    # dict_complexities_1 = calculate_language_complexity(dict_complete)
    
    # for name in dict_complete.keys():
    #     complex_1 = dict_complexities_1[name]
    #     complex_3 = dict_complexities_3[name]
    #     print(complex_1, complex_3, ','.join(name))
    #     plt.text(
    #         complex_1,
    #         complex_3,
    #         s=','.join(name)
    #     )
    # plt.xlim(0, 8)
    # plt.ylim(0, 3)
    # plt.show()

