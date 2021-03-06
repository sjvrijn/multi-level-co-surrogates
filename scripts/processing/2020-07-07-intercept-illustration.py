# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pyprojroot import here


plot_dir = here('plots/', warn=False)
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
    gradient = 3
    G = np.arange(200)*gradient + (nhigh - gradient*nlow)
    intercept_high, intercept_low = calc_intercept(b2, nhigh, nlow, gradient, ratio)

    X = np.arange(nhigh+1).reshape((-1, 1))*gradient + np.arange(nlow+1).reshape((1, -1))
    X = np.ma.masked_equal(np.triu(X), 0)

    plt.grid(alpha=.6, ls=':')
    plt.imshow(X, origin='lower')
    plt.plot(np.arange(len(B1)), B1, label='initial budget (60)')
    plt.plot(np.arange(len(B2)), B2, label='extended budget (80)')
    plt.plot(G, label='calculated gradient')
    plt.plot(intercept_low, intercept_high, 'bx', label='intercept')
    plt.ylabel('$n_h$')
    plt.xlabel('$n_l$')
    plt.xlim([0, 130])
    plt.ylim([0, 55])
    plt.xticks(np.arange(6)*25)
    plt.legend(loc=2)
    plt.savefig(plot_dir / 'budget-extension-intercept.png', bbox_inches='tight')


if __name__ == '__main__':
    do_plot()
