import logging

import numpy as np
import xarray
from pint_xarray import unit_registry as ureg

logger = logging.getLogger(__package__)
ureg.default_format = 'C~'


class Comparison:
    interpolation_direction: str = None

    def __init__(self, observation: xarray.Dataset, prediction: xarray.Dataset):
        ndim = observation.ndim

        if ndim > 2:
            raise ValueError(f'Observation dataset must be <3D but is {observation.ndim}')
        if ndim != prediction.ndim:
            raise ValueError(f'Observation and prediction dataset have unequal dimension: {ndim}D '
                             f'vs {prediction.ndim}')

        observation_coords = [observation.coords[cname] for cname in observation.coords if
                              observation.coords[cname].ndim != 0]
        prediction_coords = [prediction.coords[cname] for cname in prediction.coords if
                             prediction.coords[cname].ndim != 0]

        coord_min = [max(observation_coords[i].min(), prediction_coords[i].min()) for i in range(ndim)]
        coord_max = [min(observation_coords[i].max(), prediction_coords[i].max()) for i in range(ndim)]

        o_sizes = [coord[np.logical_and(coord > coord_min[i], coord < coord_max[i])].size for (i, coord) in
                   enumerate(observation_coords)]
        p_sizes = [coord[np.logical_and(coord > coord_min[i], coord < coord_max[i])].size for (i, coord) in
                   enumerate(prediction_coords)]
        observation_size = np.prod(o_sizes)
        prediction_size = np.prod(p_sizes)

        if observation_size > prediction_size:
            # interpolate observation onto prediction
            self.interpolation_direction = 'observation onto prediction'
            logger.info(f'interpolating {self.interpolation_direction}')

            self.o = observation.interp({coord.name: coord for coord in prediction_coords})
            self.p = prediction
            self.dims = prediction.dims
            self.coords = prediction.coords
        else:
            self.interpolation_direction = 'prediction onto observation'
            logger.info(f'interpolating {self.interpolation_direction}')

            self.p = prediction.interp({coord.name: coord for coord in observation_coords})
            self.o = observation
            self.dims = observation.dims
            self.coords = observation.coords
