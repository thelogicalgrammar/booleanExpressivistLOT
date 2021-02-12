from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pickle import load
from utilities import calculate_expected_complexity, calculate_language_complexity


def plot_languages(dict_usage_complexities, dict_cognitive_complexity):
    """
    Plot the languages stored in the dictionaries
    """
    attested_languages = (
        frozenset(['nor', 'and', 'or', 'not']),
        frozenset(['and', 'or', 'not']),
        frozenset(['and', 'not']),
        frozenset(['or', 'not']),
    )

    fig, ax = plt.subplots(figsize=(8.27,4))
    for name in dict_usage_complexities.keys():

        # if not any([i in ['nc', 'nic', 'bc', 'XOR', 'c', 'ic'] for i in name]) and 'not' in name:
        if 'not' in name:
        # if True:

            usage_complexity = dict_usage_complexities[name]
            cognitive_complexity = dict_cognitive_complexity[name]

            if name in attested_languages:
                color = 'red'
                zorder = 10
                if name == frozenset(['or', 'not']):
                    yshift = 0.4
                else:
                    yshift = 0
                ax.text(
                    usage_complexity + 0.02,
                    cognitive_complexity + 0.3 + yshift,
                    s=','.join(name),
                    fontsize='x-small'
                )
            else:
                color='black'
                zorder = 1

#             ax.scatter(
                  # usage_complexity, cognitive_complexity,
#                 color=color,
#                 zorder=zorder
#             )
            # ax.text(
                  # usage_complexity, cognitive_complexity,
            #     s=','.join(name),
            #     fontsize='xx-small',
            #     rotation=90,
            #     color=color
            # )
            ax.scatter(usage_complexity,cognitive_complexity,color=color)

    ax.set_xlabel('Usage complexity')
    ax.set_ylabel('Conceptual complexity')
    # ax.set_xlim(0,3)
    ax.set_xlim(1.05,2.8)

    # plt.show()
    plt.savefig('figure.png', dpi=300, transparent=True)


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

dict_usage_complexity = calculate_expected_complexity(languages)
dict_cognitive_complexity = calculate_language_complexity(languages)

# print(calculate_expected_complexity(languages)[frozenset({'not', 'and'})])
# pprint(
#     sorted(
#         languages[frozenset({'not', 'or', 'and', 'nor'})],
#         key=lambda x: x[0]
#     )
# )
print(len([a for a in languages.keys() if 'not' in a]))
# plot_languages(dict_usage_complexity, dict_cognitive_complexity)
