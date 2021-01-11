from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pickle import load
from utilities import calculate_expected_complexity, calculate_language_complexity


def plot_languages(dict_complexities_1, dict_complexities_3):

    attested_languages = (
        frozenset(['nor', 'and', 'or', 'not']),
        frozenset(['and', 'or', 'not']),
        frozenset(['and', 'not']),
        frozenset(['or', 'not']),
    )

    fig, ax = plt.subplots()
    for name in dict_complexities_1.keys():

        if not any([i in ['nc', 'nic', 'bc', 'XOR', 'c', 'ic'] for i in name]) and 'not' in name:
        # if 'not' in name:

            complex_1 = dict_complexities_1[name]
            complex_3 = dict_complexities_3[name]

            if name in attested_languages:
                color = 'red'
                zorder = 10
                # ax.text(
                #     complex_1, complex_3,
                #     s=','.join(name),
                #     fontsize='xx-small'
                # )
            else:
                color='black'
                zorder = 1

#             ax.scatter(
#                 complex_1,
#                 complex_3,
#                 color=color,
#                 zorder=zorder
#             )
            ax.text(
                complex_1,
                complex_3,
                s=','.join(name),
                fontsize='xx-small',
                rotation=90,
                color=color
            )

    ax.set_xlim(0,3)
    ax.set_ylim(0,12)

    plt.show()


def get_minimal_languages(folder):
    languages = {}
    for path in glob(folder+'/*.pickle'):
        with open(path, 'rb') as openfile:
            booleans, formulas = load(openfile)
            # dictionary = {b:f for b,f in zip(booleans,formulas)}
            dictionary = np.column_stack((booleans,formulas))
            name = frozenset(
                os.path.splitext(os.path.basename(path))[0]
                .split('_')
            )
            languages[name] = dictionary
    return languages


languages = get_minimal_languages('minimal_formulas')

dict_complexities_1 = calculate_expected_complexity(languages)
dict_complexities_3 = calculate_language_complexity(languages)
plot_languages(dict_complexities_1, dict_complexities_3)
