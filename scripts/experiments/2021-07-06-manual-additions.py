#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2021-07-06-manual-additions.py: running multi-fidelity bayesian optimizers,
but having manually pre-selected the fidelity order
"""

import argparse
from itertools import cycle
from warnings import simplefilter

import mf2
import numpy as np
import pandas as pd
from collections import namedtuple
from operator import itemgetter
from pyprojroot import here
from scipy.optimize import minimize
from time import time

import multiLevelCoSurrogates as mlcs

from experiments import create_subsampling_error_grid, plot_archive
from multiLevelCoSurrogates import scale_to_function

save_dir = here('files/2021-07-06-manual-additions/', warn=False)
save_dir.mkdir(parents=True, exist_ok=True)
plot_dir = here('plots/2021-07-06-manual-additions/', warn=False)
plot_dir.mkdir(parents=True, exist_ok=True)


def proto_EG_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, fidelity_order, mfm_opts, num_reps=50):
    np.random.seed(20160501)
    N_RAND_SAMPLES = 100
    fidelity_iter = iter(cycle(fidelity_order))

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    Entry = namedtuple('Entry', 'budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction')
    entries = []

    #make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    #subtract mf-DoE from budget
    budget = init_budget - (doe_n_high + doe_n_low*cost_ratio)

    #create archive
    archive = mlcs.CandidateArchive.from_bi_fid_doe(high_x, low_x, high_y, low_y)

    proto_eg = mlcs.ProtoEG(archive, num_reps=num_reps, mfm_opts=mfm_opts)
    proto_eg.subsample_errorgrid()

    mfm = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=archive, **mfm_opts)

    start = time()
    iter_since_high_eval = 0
    while budget > 0:

        tau = mlcs.calculate_tau(proto_eg.error_grid['mses'], cost_ratio)
        # compare \tau with current count t to select fidelity, must be >= 1
        fidelity = next(fidelity_iter)

        # predict best place to evaluate:
        if fidelity == 'high':
            #best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
            candidates = select_high_fid_only_candidates(archive)
            candidate_predictions = [
                (cand, mfm.models['high'].predict(cand.reshape(1, -1)))
                for cand in candidates
            ]

            x = min(candidate_predictions, key=itemgetter(1))[0].ravel()
            iter_since_high_eval = 0
            budget -= 1
        else:  # elif fidelity == 'low':
            # simple optimization for low-fid
            x = minimize(
                lambda x: mfm.models['high'].predict(x.reshape(1, -1)),
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(-1, ),
                bounds=func.bounds.T,
            ).x

            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(N_RAND_SAMPLES, func.ndim))
                fitnesses = mfm.models['high'].predict(random_candidates)
                x = random_candidates[np.argmin(fitnesses)]
                # print(f'... replaced by {x}')

            iter_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        y = func[fidelity](x.reshape(1, -1))[0]
        archive.addcandidate(candidate=x.flatten(), fitness=y, fidelity=fidelity)

        # update model & error grid
        mfm.retrain()
        proto_eg.update_errorgrid_with_sample(x, fidelity=fidelity)
        plot_title = f'{func.ndim}D {func.name} with {budget:.1f} budget left'
        proto_eg.plot_errorgrid(
            title=plot_title,
            as_log=True,
            save_as=plot_dir / f'protoeg-EG-opt-{func.name}-{budget/cost_ratio:.0f}',
            save_exts=('png',),
            xlim=(3, init_budget),
            ylim=(2, (init_budget // 2)),
        )
        try:
            plot_archive(
                archive,
                func,
                title=plot_title,
                save_as=plot_dir / f'protoeg-archive-opt-{func.name}-{budget/cost_ratio:.0f}',
                save_exts=('png',),
            )
        except NotImplementedError:
            pass

        # logging
        entries.append(Entry(budget, iter_since_high_eval, tau, fidelity, time()-start, archive.count('high'), archive.count('low'), proto_eg.reuse_fraction))

    return mfm, pd.DataFrame.from_records(entries, columns=Entry._fields), archive


def simple_multifid_bo(func, budget, cost_ratio, doe_n_high, doe_n_low, fidelity_order, mfm_opts, num_reps=50):
    np.random.seed(20160501)
    fidelity_iter = iter(cycle(fidelity_order))

    if doe_n_high + cost_ratio*doe_n_low >= budget:
        raise ValueError('Budget should not be exhausted after DoE')

    Entry = namedtuple('Entry', 'budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction')
    entries = []

    #make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    #subtract mf-DoE from budget
    budget -= (doe_n_high + doe_n_low*cost_ratio)

    #create archive
    archive = mlcs.CandidateArchive.from_bi_fid_doe(high_x, low_x, high_y, low_y)

    #make mf-model using archive
    mfbo = mlcs.MultiFidelityBO(func, archive, **mfm_opts)

    start = time()
    iter_since_high_eval = 0
    while budget > 0:
        #select next fidelity to evaluate:
        #sample error grid
        EG = create_subsampling_error_grid(archive, num_reps=num_reps, func=func, **mfm_opts)
        tau = mlcs.calculate_tau(EG, cost_ratio)

        #compare \tau with current count t to select fidelity, must be >= 1
        fidelity = next(fidelity_iter)

        #predict best place to evaluate:
        if fidelity == 'high':
            #best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
            candidates = select_high_fid_only_candidates(archive)
            candidate_predictions = [
                (cand, mfbo.models['high'].predict(cand.reshape(1, -1)))
                for cand in candidates
            ]

            x = min(candidate_predictions, key=itemgetter(1))[0].ravel()
            iter_since_high_eval = 0
            budget -= 1
        else:  # elif fidelity == 'low':
            #simple optimization for low-fid
            x = minimize(
                lambda x: mfbo.models['high'].predict(x.reshape(1, -1)),
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(-1, ),
                bounds=func.bounds.T,
            ).x

            N_RAND_SAMPLES = 100
            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(N_RAND_SAMPLES, func.ndim))
                fitnesses = mfbo.models['high'].predict(random_candidates)
                x = random_candidates[np.argmin(fitnesses)]
                # print(f'... replaced by {x}')

            iter_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        y = func[fidelity](x.reshape(1, -1))[0]
        archive.addcandidate(x, y, fidelity=fidelity)
        entries.append(Entry(budget, iter_since_high_eval, tau, fidelity, time()-start, archive.count('high'), archive.count('low'), 0))

        #update model
        mfbo.retrain()

    return mfbo, pd.DataFrame.from_records(entries, columns=Entry._fields), archive



def select_high_fid_only_candidates(archive):
    all_low = {
        tuple(candidate)
        for candidate in archive.getcandidates(fidelity='low').candidates
    }
    all_high = {
        tuple(candidate)
        for candidate in archive.getcandidates(fidelity='high').candidates
    }
    selected_candidates = all_low - all_high
    return [np.array(cand).reshape(1, -1) for cand in selected_candidates]


def main(idx=None):
    import sklearn
    simplefilter("ignore", category=FutureWarning)
    simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)
    simplefilter("ignore", category=mlcs.TauSmallerThanOneWarning)
    simplefilter("ignore", category=mlcs.LowHighFidSamplesWarning)
    num_iters = 1
    np.random.seed(20160501)

    mfm_opts = dict(
        kernel='Matern',
        scaling='off',
    )

    functions = [
        mf2.branin,
        mf2.currin,
        mf2.himmelblau,
        mf2.six_hump_camelback,
        mf2.park91a,
        mf2.hartmann6,
        mf2.borehole,
        mf2.bohachevsky,
        mf2.booth,
        mf2.park91b,
    ]

    if idx is not None:
        functions = [functions[idx]]

    for func in functions:
        print(func.name)
        for budget in [20]:  # 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]:

            kwargs = dict(
                init_budget=budget,
                cost_ratio=0.2,
                doe_n_high=5,
                doe_n_low=10,
                num_reps=5,
                fidelity_order=['low', 'low', 'high'],
                mfm_opts=mfm_opts,
            )
            for idx in range(num_iters):
                #do_run(func, 'fixed', fixed_ratio_multifid_bo, kwargs)
                #do_run(func, f'naive-b{budget}-i{idx}', simple_multifid_bo, kwargs)
                do_run(func, f'proto-eg-b{budget}-i{idx}', proto_EG_multifid_bo, kwargs)


def do_run(func, name, run_func, kwargs):
    print(f'    {name}...')
    _, df, archive = run_func(
        func=func,
        **kwargs
    )
    df.to_csv(save_dir / f'{func.name}-tracking-{name}.csv')
    # with open(save_dir / f'{func.name}-archive-{name}.pkl', 'wb') as f:
    #     dump(str(archive.data), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, nargs='?')
    args = parser.parse_args()

    main(args.idx)
