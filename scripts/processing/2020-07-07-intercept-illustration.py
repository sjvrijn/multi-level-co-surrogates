# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pyprojroot import here

import processing as proc

print(f'Running script: {__file__}')


plot_dir = here('plots/2020-07-07-intercept-illustration/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)


def calc_intercept(b, h0, l0, gradient, costratio):
    l = (b - h0 + gradient*l0) / (gradient + costratio)
    h = b - costratio*l
    return h, l


def do_plot():
    nhigh, nlow = 30, 75

    ratio = .4
    b1, b2 = 60, 80
    B1 = np.linspace(b1, 0, int(b1/ratio) + 1, endpoint=True)
    B2 = np.linspace(b2, 0, int(b2/ratio) + 1, endpoint=True)
    gradient = 0.75
    G = np.arange(200)*gradient + (nhigh - gradient*nlow)
    intercept_high, intercept_low = calc_intercept(b2, nhigh, nlow, gradient, ratio)

    X = np.arange(nhigh+1).reshape((-1, 1))*gradient + np.arange(nlow+1).reshape((1, -1))
    X = np.ma.masked_equal(np.triu(X), 0)

    plt.figure(figsize=(5.2, 3.9))
    plt.grid(alpha=.6, ls=':')
    plt.imshow(X, origin='lower')
    plt.plot(np.arange(len(B2)), B2, label='extended budget $b_0 + b$')
    plt.plot(G, label='calculated gradient', linestyle='--')
    plt.plot(intercept_low, intercept_high, 'bx', label='intercept')
    plt.ylabel('$n_h$')
    plt.xlabel('$n_l$')
    plt.xlim([0, 130])
    plt.ylim([0, 55])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=2)
    for ext in proc.extensions:
        plt.savefig(plot_dir / f'budget-extension-intercept.{ext}',
                    dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    do_plot()
