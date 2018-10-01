import dateutil.parser

from matplotlib import pyplot as plt
import seaborn as sns


def graph_oob_segs(data, path):
    x = sorted(list(data.keys()))
    y = [data[k] for k in x]
    plot = sns.barplot(y, x, palette='Set3', orient='h')
    plot.tick_params(labelsize=5)
    plt.savefig(path, dpi=300)
    plt.close('all')


def graph_oobs_over_gens(data, path):
    data['OBE'] = data['OOB']
    plot = sns.lmplot(x='Generation', y='OBE', data=data,
                      palette='muted', scatter_kws={'s': 0.1})
    plt.savefig(path, dpi=300)
    plt.close('all')


def graph_final_oobs(data, path):
    plot = sns.barplot(x='Aggression', y='OOB',
                       hue='Boundary', data=data, palette='muted')
    plt.savefig(path, dpi=300)
    plt.close('all')
