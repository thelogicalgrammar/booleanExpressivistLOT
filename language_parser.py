import numpy as np


def get_indentation_levels(formula, exclude_brackets):
    if type(formula)==str:
        formula_np = np.array(list(formula)) 
    position_closed = formula_np==')'
    formula_open = formula_np=='('
    formula_close = np.zeros(len(position_closed))
    formula_close[1:] = (np.where(position_closed,-1,0))[:-1]
    formula_level = (formula_close+formula_open).cumsum()
    if exclude_brackets:
        brackets_position = formula_open | position_closed
        formula_without_brackets = formula_np[~brackets_position]
        formula_level_without_brackets = formula_level[~brackets_position]
        return formula_without_brackets, formula_level_without_brackets
    else:
        return formula_np, formula_level


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
    f_wo_brackets, level_wo_brackets = get_indentation_levels(formula,True)
    return eval_formula_recursive(
        f_wo_brackets,
        level_wo_brackets,
        m_dict
    ) 

