from re import split
import os.path
import pickle
from functools import reduce, lru_cache
from itertools import combinations
from pprint import pprint
from types import FunctionType
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from language_parser import get_indentation_levels, evaluate_formula, np_to_string
import argparse


def bit_not(n, numbits=4):
    return (1<<numbits)-1-n


def combine(f_1, f_2, first_top_unsaturated):
    """
    Substitute the first argument on top level in f_1 for f_2
    """
    # Old implementation which adds simply
    # to the first occurrence of _
    # return f_1.replace('_',f_2,1)
    return np.concatenate((
        # everything before the _
        f_1[:first_top_unsaturated],
        # the argument itself
        f_2,
        # everything after the _
        f_1[first_top_unsaturated+1:]
    ))


def find_n_args(f):
    """
    Finds the number of arguments of a function
    """
    return f.__code__.co_argcount


def check_symmetry(new_node, new_nodes, pivot):
    """
    check that new_node is not symmetric to any of new_nodes
    First flips the formula around the pivot, 
    Then checks that the flipped version is not identical to any of
    the nodes in new_nodes

    Parameters
    ----------
    new_node: string
        The node that is proposed to add to the list of new nodes
    new_nodes: list of strings
        The nodes that have already been added in that generation
    levels_top_unsat: boolean array 
        For each char in new_node, whether the char is the same level
    """
    assert new_node[pivot]==',', 'Pivot isnt a comma'
    # the first argument returned is simply new_node
    f, levels = get_indentation_levels(new_node, False)
    # get the last open bracket before the pivot
    # at the same indentation level as the pivot

    before_p = np.nonzero(
        (f[:pivot]=='(') & (levels[:pivot]==levels[pivot])
    )[0][-1]
    # get first closed bracket after the pivot
    # at the same indentation level of the pivot
    after_p = pivot + np.nonzero(
        (f[pivot:]==')') & (levels[pivot:]==levels[pivot])
    )[0][0]

    flipped_node = np.concatenate((
        # include the opening bracket
        f[:before_p+1],
        # from the pivot to before the closing bracket (second arg)
        f[pivot+1:after_p],
        # comma corresponding to the pivot
        np.array([',']),
        # from the after the opening bracket to pivot
        f[before_p+1:pivot],
        # everything after and including the closing bracket
        f[after_p:]
    ))
    return np_to_string(flipped_node) in new_nodes


def find_shortest_formulas(m_dict):
    """
    Since formulas are added to saturated in order of complexity,
    whenever I encounter a meaning that's already in 
    saturated, I don't need to add it to saturated
    Because the new meaning is going to be at least
    as complex as the one that's alread in saturated
    Otherwise, add to saturated
    NOTE: this relies on the fact that the shortest
    formula for a meaning is a combination of shortest formulas 
    NOTE: lexicon should contain first p and q, and then
    the other terms in order of complexity
    """
    saturated, saturated_m, unsaturated, unsaturated_m = [], [], [], []
    for k,v in m_dict.items():
        if type(v)==FunctionType:
            n_args = find_n_args(v)
            unsaturated_m.append(v)
            unsaturated.append(k + '(' + ','.join(['_']*n_args) + ')')
        else:
            saturated.append(k)
            saturated_m.append(v)

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
    lexicon = [np.array(list(a)) for a in lexicon]
    # start with just the initial unsaturated in it
    # unsaturated terminal nodes
    terminal_nodes = [*unsaturated]
    while len(unique_meanings) < 16:
        new_nodes = []
        # loop through the terminal nodes
        for current_node in terminal_nodes:
            # loop through the lexicon
            # and try to add each of the saturated
            for n in lexicon:
                # replace the first occurrence of _ on the top
                # nesting level of current_node with lexicon entry n
                f, levels = get_indentation_levels(
                    current_node, False)
                first_top_unsaturated = np.argmin(
                    np.where(f=='_',levels,np.inf))
                new_node = np_to_string(combine(f, n, first_top_unsaturated))

                # store the levels of the substitution to use it later
                # to find the pivot
                levels_top_unsat = (
                    (levels==levels[first_top_unsaturated])&(f=='_'))

                # when encountering a saturated node, 
                # check if it is in unique already
                # NOTE: not equivalent to nested ifs
                if '_' not in new_node:
                    new_meaning = evaluate_formula(new_node, m_dict)
                    if new_meaning not in unique_meanings:  
                        unique_formulas.append(new_node)
                        unique_meanings.append(new_meaning)
                else:
                    # assert levels_top_unsat.sum() < 3, "Strange!"
                    # NOTE: if the character before the '_' is an open
                    # bracket, add directly without checking for symmetry
                    # (python's 'or' short-circuits, so if the first
                    # condition is true the second isn't evaluated)
                    # The condition isn't saved if it's the second arg
                    # AND it's symmetric to a previous expression
                    if (
                            new_node[first_top_unsaturated-1]=='(' or 
                            # TODO: only if the operation is symmetrc
                            not check_symmetry(
                                new_node,
                                new_nodes,
                                first_top_unsaturated-1
                            )):
                        new_nodes.append(new_node)
                # breakpoint()
        # new_nodes, new_m = reduce_symmetric_operators(new_nodes, new_m)
        terminal_nodes = new_nodes
        print(len(terminal_nodes),'')
    return [f'{x:04b}' for x in unique_meanings], unique_formulas


def calculate_functionally_complete():
    f_complete = [
        # one element
        ('nor',),
        ('nand',),
        # two elements
        ('or', 'not'),
        ('and', 'not'),
        ('c', 'not'),
        ('ic', 'not'),
        # ('c', 'F'),
        # ('ic', 'F'),
        ('c', 'XOR'),
        ('ic', 'XOR'),
        ('c', 'nc'),
        ('c', 'nic'),
        ('ic', 'nc'),
        ('ic', 'nic'),
        ('nc', 'not'),
        ('nic', 'not'),
        # ('nc', 'T'),
        # ('nic', 'T'),
        ('nc', 'bc'),
        ('nic', 'bc'),
        # three elements
        # ('or', 'bc', 'F'),
        ('or', 'bc', 'XOR'),
        # ('or', 'XOR', 'T'),
        # ('and', 'bc', 'F'),
        ('and', 'bc', 'XOR'),
        # ('and', 'XOR', 'T')
    ]
    # calculate all combinations of signals
    # of up to 5 elements
    # which are supersets of at least one of the sets in f_complete
    f_all = []
    # list of the binary operators
    names = ['not', 'or', 'and', 'nand', 'nor', 
            'c', 'ic', 'nc', 'nic', 'bc', 'XOR']
    for i in range(1,6):
        for signals_combination in combinations(names, r=i):
            # print([sig in sig_f_complete for sig_f_complete in f_complete])
            if any([
                all([sig in signals_combination for sig in sig_complete_tuple])
                for sig_complete_tuple in f_complete]):
                f_all.append(signals_combination)

    return f_all


def calculate_for_all_functionally_complete(m_dict):
    f_all = calculate_functionally_complete()

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


def calculate_expected_complexity(dict_complete):
    """
    Calculates expected complexity for each lang,
    where complexity is calculated as the number of operations 
    in the formula, found as the number of '('.

    Parameters
    ----------
    dict_complete: dict
        Dictionary where each key is a tuple (or ordered dict) of primitives
        and each value is 
    """
    expected_complexity_dict = {}
    for tuple_primitives, shortest_defs in dict_complete.items():
        expected_complexity_dict[tuple_primitives] = np.sum([
            s.count('(') for s in shortest_defs[:,1]
        ])/len(shortest_defs)
    return expected_complexity_dict


def calculate_language_complexity(lang_dict):

    negation_cost = 1
    complexities = {
        'not': 1 + negation_cost,
        'and': 1,
        'or': 1,
        'nor': 1 + 2*negation_cost,
        'nand': 2 + 2*negation_cost,
        'XOR': 3 + negation_cost,
        'bc': 3,
        'c': 2 + negation_cost,
        'ic': 2 + negation_cost,
        'nc': 1 + negation_cost,
        'nic': 1 + negation_cost
    }
    # complexities = {
    #     'not': 1,
    #     'and': 1,
    #     'or': 1,
    #     'nor': 1,
    #     'nand': 2,
    #     'XOR': 3,
    #     'bc': 3,
    #     'c': 2,
    #     'ic': 2,
    #     'nc': 1,
    #     'nic': 1
    # }

    complexity = {
        key: sum([complexities[k] for k in key])
        for key,_ in lang_dict.items()
    }

    return complexity


def save_languages(langs_formatted, path='functionally_complete'):
    with open(f'{path}.txt', 'w', newline='\n') as openfile:
        openfile.write(langs_formatted)


def update_languages_file(
        file_path='functionally_complete',
        folder_path='minimal_formulas'):
    """
    Parameters
    ----------
    file_path: str
        Path to the file containing the list of functionally complete languages
        which should be updated
    folder_path: str
        Path to the folder containing the pickle files with the minimal formulas
    """
    with open(f'{file_path}.txt', 'r') as openfile:
        currently_stored = [s.strip() for s in openfile]
    remaining_langs = [
        lang for lang in currently_stored
        if not os.path.isfile(f'{folder_path}/{lang}.pickle')
    ]
    print(remaining_langs)
    langs_formatted = '\n'.join(remaining_langs)
    save_languages(langs_formatted, file_path)


if __name__=='__main__':

    total_dict = {
        'p': 0b1100,
        'q': 0b1010,
        'F': 0b0000,
        'T': 0b1111,
        'not': lambda x: bit_not(x),
        'or': lambda x, y: x|y,
        'and': lambda x, y: x&y,
        'nand': lambda x, y: bit_not(x&y),
        'nor': lambda x, y: bit_not(x|y),
        'c': lambda x, y: bit_not(x)|y,
        'ic': lambda x, y: bit_not(y)|x,
        'nc': lambda x, y: bit_not(bit_not(x)|y),
        'nic': lambda x, y: bit_not(bit_not(y)|x),
        'bc': lambda x, y: (x&y)|(bit_not(x)&bit_not(y)),
        'XOR': lambda x, y: bit_not((x&y)|(bit_not(x)&bit_not(y)))
    }

    parser = argparse.ArgumentParser(description='Input for simulation')

    parser.add_argument(
        '--primitives',
        type=str,
        help='name of primitives separated by a _'
    )

    parser.add_argument(
        '--action',
        type=str,
        help=(
            'What to do. Either "single_minimal" or "all_minimal"'
            'or "save_functionally_complete"'
            'or "update_languages_file"'
            )
    )

    args = parser.parse_args()

    if args.action == 'all_minimal':
        dict_complete = calculate_for_all_functionally_complete(total_dict)
        dict_complexities_3 = calculate_expected_complexity(dict_complete)
        pprint(list(sorted(dict_complexities_3.items(), key=lambda x:x[1])))
        dict_complexities_1 = calculate_language_complexity(dict_complete)

    elif args.action == 'single_minimal':
        file_path = f'./minimal_formulas/{args.primitives}.pickle'
        if os.path.isfile(file_path):
            print('Already calculated')
        else:
            primitives = args.primitives.split('_')

            m_dict_restricted = {k:total_dict[k] for k in primitives}
            m_dict_restricted.update({
                'p': total_dict['p'],
                'q': total_dict['q']
            })
            shortest = find_shortest_formulas(m_dict_restricted)
            with open(file_path, 'wb') as openfile:
                pickle.dump(shortest, openfile)
    
    elif args.action == 'save_functionally_complete':
        complete_langs = calculate_functionally_complete()
        langs_formatted = '\n'.join([
            '_'.join(a) for a in complete_langs
        ])
        save_languages(langs_formatted)

    elif args.action == 'update_languages_file':
        update_languages_file()
