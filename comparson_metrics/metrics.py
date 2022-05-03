import logging
from enum import Enum
from pathlib import Path
from typing import List

import bibtexparser
import numpy as np
import xarray as xr
from pint_xarray import unit_registry as ureg

logger = logging.getLogger(__package__)
ureg.default_format = 'C~'

bib_tex_filename = Path(__file__).parent.joinpath('metrics.bib')


class Unit(Enum):
    DIMENSIONLESS = 0
    REAL = 1


class Metric:
    """Class that documents a comparison metric. Only with compute() the comparison metric is computed"""
    name: str = None
    long_name: str = None
    lim: List = None
    best: float = None
    worst: float or List = None
    unit: str = None  # "dimensionless" or "real"
    description: str = 'N.A.'  # optional description about the method
    bibtex: str = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out = f'Method Name: "{self.long_name}"'
        out += f'\nbibtex: {self.bibtex}'
        if self.lim[0] == np.inf:
            lim_low = u'∞'
        elif self.lim[0] == -np.inf:
            lim_low = u'-∞'
        else:
            lim_low = str(self.lim[0])

        if self.lim[1] == np.inf:
            lim_high = u'∞'
        elif self.lim[1] == -np.inf:
            lim_high = u'-∞'
        else:
            lim_high = str(self.lim[1])

        if self.best == np.inf:
            best = u'∞'
        elif self.best == -np.inf:
            best = u'-∞'
        else:
            best = str(self.best)

        if self.worst == np.inf:
            worst = u'∞'
        elif self.worst == -np.inf:
            worst = u'-∞'
        else:
            worst = str(self.worst)

        out += f'\nLimits: [{lim_low}, {lim_high}]'
        out += f' (worst match: {worst}; best match: {best})'
        return out

    @property
    def is_fully_defined(self):
        if self.name is None:
            logger.error(f'No "name" is given to metric {self.__class__.name}')
            return False
        if self.long_name is None:
            logger.error(f'No "long_name" is given to metric {self.__class__.name}')
            return False
        if self.bibtex is None:
            logger.error(f'No "bibtex" is given to metric {self.__class__.name}')
            return False
        if self.lim is None:
            logger.error(f'No "lim" is given to metric {self.__class__.name}')
            return False
        if not isinstance(self.lim, (list, tuple)):
            logger.error(f'Attribute "lim" is not a list!')
            return False
        if len(self.lim) != 2:
            logger.error(f'Attribute "lim" must have two entries not {len(self.lim)}')
            return False
        if self.lim[0] >= self.lim[1]:
            logger.error(f'Attribute lim has wrong entries')
            return False
        if self.best is None:
            logger.error(f'No "best" is given to metric {self.__class__.name}')
            return False
        if self.worst is None:
            logger.error(f'No "worst" is given to metric {self.__class__.name}')
            return False
        if self.unit is None:
            logger.error(f'No "unit" is given to metric {self.__class__.name}')
            return False
        return True

    def get_bibtex(self) -> dict:
        """Returns the reference as BibTeX-string."""

        with open(bib_tex_filename, 'r') as bibtex_file:
            bibtex_str = bibtex_file.read()

        bib_database = bibtexparser.loads(bibtex_str)
        bibstrs = [bib_database.get_entry_dict()[bstr] for bstr in self.bibtex if bstr]
        return bibstrs

    def add_bibtex_to_bibfile(self, bibfile: str, mode: str = 'a'):
        bibstrs = self.get_bibtex()
        if not bibstrs:
            raise ValueError('No bibtex written to file because metric provides no bibtex')

        db_selection = bibtexparser.bibdatabase.BibDatabase()
        db_selection.entries = bibstrs
        writer = bibtexparser.bwriter.BibTexWriter()
        with open(bibfile, mode) as bibfile:
            bibfile.write(writer.write(db_selection))

    def get_attr_dict(self):
        return {'lim': self.lim,
                'best': self.best,
                'worst': self.worst,
                'unit': self.unit,
                'long_name': self.long_name,
                'bibtex': self.bibtex}

    def compute(self, observation: np.ndarray, prediction: np.ndarray):
        # overwrite by subclass
        pass


class AE(Metric):
    """The absolute error, simple difference between observation and prediction"""
    name = 'AE'
    long_name = 'Absolute Error'
    lim = [0., (-np.inf, np.inf)]
    best = 0.
    worst = np.inf
    unit = Unit.REAL
    description: str = 'The absolute error between.'  # optional description about the method
    bibtex = []

    def compute(self, observation: np.ndarray, prediction: np.ndarray, **kwargs):
        return observation - prediction


def RE(Metric):
    """Relative Error. The observation is taken as reference"""
    name = 'RE'
    long_name = 'Relative Error'
    lim = [0., p.nan]
    best = 0.
    worst = np.nan
    unit = Unit.REAL
    description: str = 'The relative error.'  # optional description about the method
    bibtex = []

    def compute(self, observation: np.ndarray, prediction: np.ndarray, **kwargs):
        return (prediction - observation) / observation


def ARE(Metric):
    """Absolute relative Error. The observation is taken as reference"""
    name = 'ARE'
    long_name = 'Absolute relative Error'
    lim = [0., p.nan]
    best = 0.
    worst = np.nan
    unit = Unit.REAL
    description: str = 'The absolute value of the relative error.'  # optional description about the method
    bibtex = []

    def compute(self, observation: np.ndarray, prediction: np.ndarray, **kwargs):
        return np.abs((prediction - observation) / observation)


class RI(Metric):
    """Relevance Index (cosine of angle between vectors)"""
    name = 'RI'
    long_name = 'relevance index'
    lim = [-1, 1]
    best = 1.
    worst = -1.
    unit = Unit.DIMENSIONLESS
    bibtex = ['liu2011development', 'willman2020quantitative', 'kuo2014large', 'van2018large',
              'shen2021temporal']
    description = 'Similar to LSI and ASI. Computes the cosine between vectors.'

    def compute(self, observation: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        observation_abs = np.linalg.norm(observation, axis=-1)
        prediction_abs = np.linalg.norm(prediction, axis=-1)

        p = np.sum(observation * prediction, axis=-1)

        normalization = observation_abs * prediction_abs
        if isinstance(normalization, np.ndarray):
            normalization[normalization == 0] = np.nan

        return p / normalization


class ASI(Metric):
    """Angular Similarity Index (equal to RI, cosine of angle between vectors)"""
    name = 'ASI'
    long_name = 'angular similarity index'
    lim = [-1, 1]
    best = 1.
    worst = -1.
    units = Unit.DIMENSIONLESS
    bibtex = ['raschi2012cfd', 'tang2013numerical']
    description: str = 'Similar to LSI and RI. Computes the cosine between vectors.'

    def compute(self, observation: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        ri = RI()
        return ri.compute(observation, prediction)


class LSI(Metric):
    """Local Structure Index (LSI) (equal to RI, ASI, thus cosine of vector fields)"""
    name = 'LSI'
    long_name = 'local structure index'
    lim = [-1, 1]
    best = 1.
    worst = -1
    units = Unit.DIMENSIONLESS
    bibtext = ['zhao2019multi', ]
    description: str = 'Similar to RI and ASI. Computes the cosine between vectors.'

    def compute(self, observation, prediction):
        ri = RI()
        self.result = ri.compute(observation, prediction)  # ASI=RI
        return self.result


class MAE(Metric):
    """Mean Absolute Error.
    Note that in wustenhagen2021cfd, the normalization is a reference velocity"""
    name = 'MAE'
    long_name = 'mean absolute error'
    lim = [-np.inf, np.inf]
    best = 0  #
    worst = (-np.inf, np.inf)
    units = Unit.REAL
    description = 'measure for prediction mean bias'
    bibtex = ['zhao2019multi', 'wustenhagen2021cfd']

    def compute(self, observation: np.ndarray, prediction: np.ndarray) -> float:
        tmp = np.abs(observation - prediction)
        if isinstance(tmp, np.ndarray):
            tmp[np.isinf(tmp)] = np.nan
            result = np.nanmean(tmp)
        elif isinstance(tmp, xr.DataArray):
            tmp = tmp.fillna(np.nan)
            result = tmp.mean().pint.dequantify(format=ureg.default_format)
            result.attrs.update(self.get_attr_dict())
        self.result = result
        return self.result


# register all metrics here:
metric_dict = {m.name: m for m in (AE, RI, MAE)}
