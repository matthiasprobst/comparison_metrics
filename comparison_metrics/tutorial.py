import pathlib

import numpy as np
import xarray as xr

av_names = [
    # '2d_vec_field',
    '1d_signals',  # two arbitrary 1d sinusoidal data arrays
    'vortex1',  # ila vortex pair with piv evaluation setting 1
    'vortex2',  # ila vortex pair with piv evaluation setting 2 to generate different vectors than with setting 1
    'rasm'  # xarray tutorial dataset from rasm (temperature data)
]

_parent = pathlib.Path(__file__).parent


def load_dataset(name):
    if name not in av_names:
        raise Exception(f'Tutorial dataset with name "{av_names} not available')

    if name == '1d_signals':
        _x1 = np.linspace(0, 3, 11)
        xshift = 0.2
        _x2 = np.linspace(0, 3, 8) + xshift  # shift x data of other dataset

        x1 = xr.DataArray(dims='x', data=_x1,
                          attrs={'units': 'mm', 'long_name': 'x'})
        x2 = xr.DataArray(dims='x', data=_x2,
                          attrs={'units': 'mm', 'long_name': 'x'})

        s1 = xr.DataArray(dims='x', data=np.sin(_x1),
                          attrs={'units': 'au', 'long_name': 'signal 1'},
                          coords={'x': x1})
        s2 = xr.DataArray(dims='x', data=1.1 * np.sin(_x2 - xshift),
                          attrs={'units': 'au', 'long_name': 'signal 1'},
                          coords={'x': x2})
        s2[4] = np.nan
        return s1, s2
        # return xr.Dataset({'s1': s1, 's2': s2})

    elif name == 'vortex1':
        return xr.load_dataset(_parent.joinpath('../data/vortex1.nc'))

    elif name == 'vortex2':
        return xr.load_dataset(_parent.joinpath('../data/vortex2.nc'))

    elif name == 'rasm':
        from xarray import tutorial
        return tutorial.load_dataset('rasm')
