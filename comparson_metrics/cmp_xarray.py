import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pint_xarray
import xarray
import xarray as xr

from veccomp.core import Metric
from . import metrics
from .core import Comparison

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

pint_xarray.__version__
__version__ = version("veccomp")

logger = logging.getLogger(__package__)


@xr.register_dataarray_accessor("cmp")
class CmpArrayAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._observation = None
        self._prediction = None
        self._cmp_method = None
        self.interpolation_direction = None

    def compute_metric(self, other, metric, *args, **kwargs):
        if isinstance(metric, str):
            if metric in metrics.metric_dict:
                m = metrics.metric_dict[metric]()
        else:
            m = metric()

        cmp = Comparison(self._obj, other)

        if cmp.o.units != cmp.p.units:
            p = cmp.p.pint.quantify().pint.to(cmp.o.units)
        else:
            p = cmp.p

        if m.unit == metrics.Unit.REAL:
            units = cmp.o.units
        else:
            units = 'dimensionless'
        attrs = {'units': units}
        attrs.update(m.get_attr_dict())
        attrs['standard_name'] = 'comparison_value'

        cmp_val = m.compute(cmp.o.values, p.values, *args, **kwargs)

        # build xr.DataArray and if not a float, add coordinates
        if isinstance(cmp_val, float):
            cmp_val = xarray.DataArray(data=cmp_val,
                                       attrs=attrs)
        else:
            cmp_val = xarray.DataArray(data=cmp_val,
                                       coords=cmp.o.coords,
                                       attrs=attrs)

        return xr.Dataset(data_vars={'observation': cmp.o, 'prediction': cmp.p, m.name: cmp_val},
                          attrs={'interpolation_direction': cmp.interpolation_direction})


@xr.register_dataset_accessor("cmp")
class CmpDatasetAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._observation = None
        self._prediction = None
        self._cmp_method = None
        self.interpolation_direction = None

    def compute_metric(self, other: xarray.Dataset,
                       method: str or Metric, components: List[str], *args, **kwargs):
        if isinstance(method, str):
            if method in metrics.metric_dict:
                m = metrics.metric_dict[method]()
        else:
            m = method()

        logger.debug(f'Components used for metric: {components}')
        if len(components) == 0:
            raise ValueError('No components passed!')

        if len(components) < 2:
            raise ValueError('Number of components must be greater than 1, otherwise this is '
                             'a scalar comparison rather than a vector comparison.')

        for c in components:
            if c not in self._obj or c not in other:
                raise ValueError(f'Data Array "{c}" must exist in both datasets.')

        cmp_comp = [Comparison(self._obj[c], other[c]) for c in components]
        arr1 = np.stack([cmp.o[:] for cmp in cmp_comp], axis=-1)
        arr2 = np.stack([cmp.p[:] for cmp in cmp_comp], axis=-1)

        cmparr = m.compute(observation=arr1, prediction=arr2, *args, **kwargs)

        # build a new xr.Dataset with u_observation, v_observation, u_prediction, v_prediction, ...
        cmpval = xarray.DataArray(name=m.name, data=cmparr,
                                  dims=cmp_comp[0].dims, coords=cmp_comp[0].coords,
                                  attrs=dict(units='', long_name=m.name))
        cmpval.attrs.update(m.get_attr_dict())
        cmpval.attrs['standard_name'] = 'comparison_metric'

        observation_xarrays = {f'{_cmp_comp.o.name}_observation': _cmp_comp.o for _cmp_comp in cmp_comp}
        prediction_xarrays = {f'{_cmp_comp.p.name}_prediction': _cmp_comp.p for _cmp_comp in cmp_comp}

        ds_out = xarray.Dataset({**observation_xarrays, **prediction_xarrays, **{m.name: cmpval}})
        ds_out.attrs['interpolation_direction'] = cmp_comp[0].interpolation_direction
        self.interpolation_direction = cmp_comp[0].interpolation_direction
        return ds_out

    def _get_plotting_datasets(self):
        observation = self._obj.observation
        prediction = self._obj.prediction
        cmp_val = None
        for data_var in self._obj.data_vars:
            sn = self._obj.data_vars[data_var].attrs.get('standard_name')
            if sn == 'comparison_value':
                cmp_val = self._obj.data_vars[data_var]
                continue
        return observation, prediction, cmp_val

    def plot(self, figsize, **kwargs):
        observation, prediction, cmp_val = self._get_plotting_datasets()
        if cmp_val is not None:
            if observation.ndim == 1:
                fig, axs = plt.subplots(1, 3, figsize=figsize)
                observation.plot(ax=axs[0], **kwargs)
                prediction.plot(ax=axs[1], **kwargs)
                cmp_val.plot(ax=axs[2], **kwargs)
                plt.tight_layout()
            elif observation.ndim == 2:
                fig, axs = plt.subplots(1, 3, figsize=figsize)
                observation.plot(ax=axs[0], **kwargs)
                prediction.plot(ax=axs[1], **kwargs)
                cmp_val.plot(ax=axs[2], **kwargs)
                plt.tight_layout()

    def contourf(self, figsize, **kwargs):
        observation, prediction, cmp_val = self._get_plotting_datasets()
        if cmp_val is not None:
            if observation.ndim == 2:
                fig, axs = plt.subplots(1, 3, figsize=figsize)
                observation.plot.contourf(ax=axs[0], **kwargs)
                prediction.plot.contourf(ax=axs[1], **kwargs)
                cmp_val.plot.contourf(ax=axs[2], **kwargs)
                plt.tight_layout()
