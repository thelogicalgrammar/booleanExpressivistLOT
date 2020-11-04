from functools import reduce, lru_cache
from types import FunctionType
from itertools import product
import numpy as np


def bit_not(n, numbits=4):
    return (1<<numbits)-1-n


def get_indentation_levels(formula):
    formula_np = np.array(list(formula))
    position_closed = formula_np==')'
    formula_open = formula_np == '('
    formula_close = np.where(position_closed,-1,0)
    formula_level = (formula_close+formula_open).cumsum()
    brackets_position = formula_open | position_closed
    formula_without_brackets = formula_np[~brackets_position]
    formula_level_without_brackets = formula_level[~brackets_position]
    return formula_without_brackets, formula_level_without_brackets


def np_to_string(f):
    return f.astype('|S1').tostring().decode('utf-8')


def separate(f,levs):
    """
    Parameters
    ----------
    f: array
        np array of characters representing the formula
    levs: array
        Int array of the nesting level of f
        Where argument levels is shifted down so it starts from 0
    Returns
    -------
    list of lists
        Each list contains a tuple (argument f array, argument levels)
    """
    assert len(f)==len(levs), 'formula and scope have different lens'
    # one comma for every argument in main scope.
    # possibly 0, e.g. for negation
    mask_commas_in_main = (f==',')&(levs==0)
    indices_commas_in_main = np.argwhere(mask_commas_in_main).flatten()
    # f without commas in main scope
    f_without_commas = f[~mask_commas_in_main]
    f_separated = np.split(f_without_commas, indices_commas_in_main)
    levs_without_commas = levs[~mask_commas_in_main]
    levs_separated = np.split(levs_without_commas, indices_commas_in_main)
    return zip(f_separated, levs_separated)


# TODO: wrap this in a decorator that simplifies e.g. exhcnages of p and q
# @lru_cache(maxsize=None)
def eval_formula_recursive(f_wo_brackets, level_wo_brackets, m_dict):
    # if it's a basic formula, return its meaning
    if np.all(level_wo_brackets==0):
        return m_dict[np_to_string(f_wo_brackets)]

    # if it's not a basic formula, run this function 
    # on its components and then combine the meanings
    operator = np_to_string(f_wo_brackets[level_wo_brackets==0])
    operator_m = m_dict[operator]

    # split the arguments
    inner_indices = level_wo_brackets > 0
    arguments_tuples = separate(
        f_wo_brackets[inner_indices],
        level_wo_brackets[inner_indices]-1) 

    # run the function on the arguments
    args_meanings = [
        eval_formula_recursive(f,lev,m_dict)
        for f, lev in arguments_tuples]

    return operator_m(*args_meanings)


def evaluate_formula(formula,m_dict):
    """
    The assumption is that the formulas have shape
        OP(formula,formula)
    NOTE: no spaces
    If a formula contains brackets, it should be analysed further
    Otherwise it's a basic expression
    """
    # find the nesting level of every character in the formula
    f_wo_brackets, level_wo_brackets = get_indentation_levels(formula)
    return eval_formula_recursive(
        f_wo_brackets, level_wo_brackets, m_dict) 


def combine(f_1, f_2):
    """
    Substitute the first argument in f_1 for f_2
    """
    return f_1.replace('_',f_2,1)

def compose(m_1, m_2):
    """
    This function puts m_2 in place of the first argument of m_1
    m_1 is a function 
    """
    if type(m_2) == FunctionType:
        n_args = m_2.__code__.co_argcount
        if n_args == 1:
            pass
        elif n_args == 2:
            pass
        

def find_shortest_formulas(saturated, unsaturated, m_dict):
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
    # store the formulas and meanings for 
    # each of the unique expressions
    unique_formulas = [*saturated]
    unique_meanings = [m_dict[f] for f in saturated]
    lexicon = [*saturated, *unsaturated]
    # start with just the initial unsaturated in it
    # unsaturated terminal nodes
    terminal_nodes = [*unsaturated]
    while len(unique_meanings) < 16:
        new_nodes = []
        # loop through the terminal nodes
        for current_node in terminal_nodes:
            # loop through the lexicon
            # and try to add each of the saturated
            for m in lexicon:
                new_node = combine(current_node, m)
                # when encountering a saturated node, 
                # check if it is in unique already
                if '_' in new_node:  
                    new_node_meaning = evaluate_formula(new_node,m_dict)
                    if new_node_meaning not in unique_meanings:
                        unique_formulas.append(new_node)
                        unique_meanings.append(new_node_meaning)
                else:
                    new_nodes.append(new_node)
        terminal_nodes = new_nodes
    return [f'{x:04b}' for x in unique_meanings], unique_formulas


def from_dict_to_unsaturated(m_dict):
    saturated, unsaturated = [], []
    for k,v in m_dict.items():
        if type(v)==FunctionType:
            unsaturated.append(
                k + '(' + ','.join(['_']*v.__code__.co_argcount) + ')')
        else:
            saturated.append(k)
    return saturated, unsaturated


if __name__=='__main__':

    m_dict = {
        'p': 0b1100,
        'q': 0b1010,
        # 'or': lambda x,y: x|y,
        # 'and': lambda x,y: x&y,
        # 'not': lambda x: bit_not(x),
        'nand': lambda x,y:bit_not(x&y)
    }
    saturated, unsaturated = from_dict_to_unsaturated(m_dict)
    # saturated = ['p','q']
    # unsaturated = [
    #     'or(_,_)',
    #     'and(_,_)',
    #     'not(_)'
    # ]

    # print(bin(evaluate_formula('or(not(p),q)', m_dict)))
    print(np.column_stack(
        find_shortest_formulas(saturated, unsaturated, m_dict)
    ))
