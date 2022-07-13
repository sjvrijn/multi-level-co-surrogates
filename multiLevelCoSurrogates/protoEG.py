from collections import defaultdict
from typing import Tuple, Union
from textwrap import fill

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import multiLevelCoSurrogates as mlcs


class ProtoEG:

    def __init__(
            self,
            archive: mlcs.CandidateArchive,
            num_reps: int=50,
            interval: int=2,
            mfm_opts=None,
    ):
        """Container for everything needed to create (advanced) Error Grids"""

        self.archive = archive
        self.num_reps = num_reps
        self.interval = interval
        self.mfm_opts = mfm_opts if mfm_opts is not None else dict()

        self.models = {}
        self.test_sets = defaultdict(list)  # test_sets[(n_high, n_low)] = [test_1, ..., test_nreps]
        self.error_grid = None  # xr.Dataset

        self.num_models_trained = 0
        self.num_models_reused = 0


    def subsample_errorgrid(self):
        """Create an error grid by subsampling from the known archive"""
        instance_spec = mlcs.InstanceSpec.from_archive(
            self.archive, num_reps=self.num_reps, step=self.interval
        )

        error_records = []
        for h, l, rep in instance_spec.instances:
            mlcs.set_seed_by_instance(h, l, rep)
            train, test = self.archive.split(h, l)

            idx_hash = train.indices
            if idx_hash in self.models:
                model = self.models[idx_hash]
                self.num_models_reused += 1
            else:
                model = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=train,
                                                **self.mfm_opts)
                self.models[idx_hash] = model
                self.num_models_trained += 1

            test_x, test_y = test.getcandidates(fidelity='high')
            mse = mean_squared_error(test_y, model.top_level_model.predict(test_x))
            error_records.append([h, l, rep, 'high_hier', mse])

        self.num_models_trained += len(error_records)
        columns = ['n_high', 'n_low', 'rep', 'model', 'mses']

        tmp_df = pd.DataFrame.from_records(error_records, columns=columns, index=columns[:4])
        self.error_grid = xr.Dataset.from_dataframe(tmp_df)


    def update_errorgrid_with_sample(self, X, fidelity: str):
        """Add a new sample of given fidelity and update Error Grid accordingly"""

        instance_spec = mlcs.InstanceSpec.from_archive(
            self.archive, num_reps=self.num_reps, step=self.interval
        )

        full_doe = self.archive.as_doe()
        X = X.reshape(1, -1)

        for h, l in instance_spec.pixels:
            if (h,l) in self.models:
                indices_to_resample = self._get_resample_indices(fidelity, h, l)
            else:
                self.models[(h,l)] = [None] * self.num_reps
                self.test_sets[(h,l)] = [None] * self.num_reps
                indices_to_resample = range(self.num_reps)
                self._extend_error_grid(fidelity=fidelity, coord=(h, l))

            for idx in indices_to_resample:
                mlcs.set_seed_by_instance(h, l, idx)
                train_doe, test_doe = mlcs.split_with_include(full_doe, h, l, must_include=X, fidelity=fidelity)
                test_high = test_doe.high

                self.test_sets[(h,l)][idx] = test_high

                # Create an archive from the MF-function and MF-DoE data
                train_archive = self._create_train_archive(train_doe)

                # create and store model
                model = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=train_archive, **self.mfm_opts)
                self.models[(h,l)][idx] = model

                # calculate and store error of model at this `idx`
                test_y = self.archive.getfitnesses(test_high, fidelity='high')
                mse = mean_squared_error(test_y, model.top_level_model.predict(test_high))
                self.error_grid['mses'].loc[h, l, idx, 'high_hier'] = mse

            self.num_models_trained += len(indices_to_resample)
            self.num_models_reused += self.num_reps - len(indices_to_resample)

            if fidelity == 'high':
                indices_to_update_errors = set(range(self.num_reps)) - set(indices_to_resample)
                for idx in indices_to_update_errors:
                    self._update_errors_of_existing_model(X, h, l, idx)


    @property
    def reuse_fraction(self):
        total = self.num_models_trained + self.num_models_reused
        return self.num_models_reused / total


    def _get_resample_indices(self, fidelity: str, h: int, l: int):
        fraction = 1 - self.calculate_reuse_fraction(h, l, fidelity)
        num_models_to_resample = int(fraction * self.num_reps)
        indices_to_resample = np.random.choice(self.num_reps, size=num_models_to_resample,
                                               replace=False)
        return indices_to_resample


    def _create_train_archive(self, train):
        train_low_y = self.archive.getfitnesses(train.low, fidelity='low')
        train_high_y = self.archive.getfitnesses(train.high, fidelity='high')
        return mlcs.CandidateArchive.from_bi_fid_doe(train.high, train.low, train_high_y, train_low_y)


    def _extend_error_grid(self, fidelity: str, coord: Tuple[int, int]):
        """Extend the error grid with new coordinate values"""
        if fidelity not in ['high', 'low']:
            raise ValueError(f"invalid fidelity '{fidelity}', should be 'high' or 'low'")

        n_high, n_low = coord
        if fidelity == 'high':
            dim = 'n_high'
            coord = n_high
        else:  # if fidelity == 'low'
            dim = 'n_low'
            coord = n_low

        tmp_ds = xr.Dataset(coords=self.error_grid.coords).assign_coords({dim: [coord]})
        self.error_grid = xr.merge([self.error_grid, tmp_ds])


    def _update_errors_of_existing_model(self, X, h: int, l: int, idx: int):
        """Add X to test set for models[(h,l)][idx] and recalculate MSE"""
        # add (X, y) to test-set for that model
        test_high = self.test_sets[(h, l)][idx]
        test_high = np.concatenate([test_high, X])
        self.test_sets[(h, l)][idx] = test_high

        # recalculate error with new test-set
        test_y = self.archive.getfitnesses(test_high, fidelity='high')
        model = self.models[(h, l)][idx]
        mse = mean_squared_error(test_y, model.top_level_model.predict(test_high))
        self.error_grid['mses'].loc[h, l, idx, 'high_hier'] = mse


    def calculate_reuse_fraction(self, num_high: int, num_low: int, fidelity: str,
                                 *, max_high: int=None, max_low: int=None) -> float:
        r"""Calculate the fraction of models that can be reused

        Given `max_high` H, `max_low` L, `num_high` h and `num_low` l, the number of
        unique possible subsamples is given by binom(H, h) * binom(L-h, l-h), i.e.:

        /H\/L-h\
        \h/\l-h/

        In the iterative case when samples are added to H or L one at a time,
        it can be expected that some of the subsamples would not use the new
        samples, and therefore that a fraction of previous subsamples can be
        reused when calculating the Error Grid for the next iteration.

        Assuming subsampling is done uniformly at random, the fraction of
        subsamples in the 'next' iteration that only use samples from the
        previous iteration is equal to the ratio between the numbers of total
        possible subsamples for those given sizes, i.e.:

        /H\/L-h\
        \h/\l-h/
        ___________

        /H+1\/L-h\
        \ h /\l-h/

        if H := H+1, or


        /H\/L-h\
        \h/\l-h/
        ___________

        /H\/L+1-h\
        \h/\ l-h /

        if L := L+1

        :param num_high: number of high-fidelity samples in the subsample
        :param num_low: number of low-fidelity samples in the subsample
        :param fidelity: fidelity in which the latest sample has been added
        :param max_high: total number of high-fidelity samples, taken from self.archive if not given
        :param max_low: total number of low-fidelity samples, taken from self.archive if not given
        :returns: fraction [0, 1] of samples that can be reused
        """

        max_high = self.archive.count('high') if not max_high else max_high
        max_low = self.archive.count('low') if not max_low else max_low

        if fidelity == 'high':
            fraction = (max_high-num_high+1) / (max_high+1)
        elif fidelity == 'low':
            fraction = (max_low-num_low+1) / (max_low-num_high+1)
        else:
            raise ValueError(f'Invalid fidelity `{fidelity}` given, expected `high` or `low`.')

        if not (0 <= fraction <= 1):
            raise ValueError('Invalid fraction calculated, please check inputs')
        return fraction


    def plot_errorgrid(self, title, vmin=None, vmax=None,
                       contours=0, as_log=False, save_as=None, save_exts=('pdf', 'png'),
                       include_colorbar=True, label_y=True, title_width=None,
                       xlim=None, ylim=None):
        """Plot a heatmap of the median MSE for each possible combination of high
        and low-fidelity samples. For comparison, the MSE for the high-only and
        low-only models are displayed as a bar to the left and bottom respectively.

        :param title: title to use at top of the image
        :param vmin: minimum value for color scale normalization
        :param vmax: maximum value for color scale normalization
        :param contours: number of contour lines to draw. Default: 0
        :param as_log: display the log10 of the data or not (default False)
        :param save_as: desired filename for saving the image. Not saved if `None`
        :param include_colorbar: whether or not to include a colorbar. Default: True
        :param label_y: whether or not to include axis label and ticks for y-axis. Default: True
        """
        if not save_as:
            return  # no need to make the plot if not showing or saving it

        data = self.error_grid['mses']
        LABEL_N_HIGH = "$n_h$"
        LABEL_N_LOW = "$n_l$"

        fig, ax = plt.subplots(figsize=(7.5, 4))

        ax.set_aspect(1.)
        data = data.median(dim='rep')
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        if as_log:
            data = np.log10(data)
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

        imshow_style = {'cmap': 'viridis_r', 'norm': norm, 'origin': 'lower'}

        plot_title = f'{"log10 " if as_log else ""}Median MSE for $z_h$ - {title}'
        if title_width:
            plot_title = fill(plot_title, width=title_width)
        plt.title(plot_title)

        da_hh = data.sel(model='high_hier')

        extent = get_extent(data)
        img = ax.imshow(da_hh, extent=extent, **imshow_style)
        if contours:
            ax.contour(da_hh, levels=contours, antialiased=False,
                       extent=extent, colors='black', alpha=.2, linewidths=1)

        divider = make_axes_locatable(ax)

        if label_y:
            ax.set_ylabel(LABEL_N_HIGH)
        else:
            ax.yaxis.set_tick_params(left=False, labelleft=False, which='both')
        ax.set_xlabel(LABEL_N_LOW)

        if xlim:
            ax.set_xlim((xlim[0]-.5, max(xlim[1], max(data.n_low))+.5))
        if ylim:
            ax.set_ylim((ylim[0]-.5, max(ylim[1], max(data.n_high))+.5))

        if include_colorbar:
            cax = divider.append_axes("right", size=0.2, pad=0.05)
            fig.colorbar(img, cax=cax)

        plt.tight_layout()
        if save_as:
            for ext in save_exts:
                plt.savefig(f'{save_as}.{ext}')
        plt.close('all')


def get_extent(data: xr.DataArray):
    """Calculate an 'extent' for an Error Grid such that axis ticks are
    centered in the 'pixels'
    """
    return [
        np.min(data.n_low) - 0.5,
        np.max(data.n_low) + 0.5,
        np.min(data.n_high) - 0.5,
        np.max(data.n_high) + 0.5,
    ]
