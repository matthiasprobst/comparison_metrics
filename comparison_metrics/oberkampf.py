import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

nx = 101
L = 2 * np.pi
x = xr.DataArray(dims='x',
                 data=np.linspace(0, L, nx),
                 attrs={'units': 'm',
                        'long_name': 'x'})

nt = 1000
dt = 1
time = xr.DataArray(dims='time',
                    data=np.arange(0, nt * dt, dt),
                    attrs={'units': 's',
                           'long_name': 'x'})

coefficient_of_varation = 0.1  # std/mu
print(f'coefficient of variation: {coefficient_of_varation}')
random_val = np.empty((nx, nt))
for ix in range(nx):
    mu = 10 * np.sin(x[ix])
    random_val[ix, :] = np.random.normal(mu,
                                         abs(coefficient_of_varation * mu),
                                         nt)

exp_data = xr.DataArray(dims=('x', 'time'), data=random_val,
                        coords={'x': x, 'time': time},
                        attrs={'long_name': 'experimental data',
                               'units': 'au'})

exp_data[:, :].plot.contourf()
plt.show()
exp_data[:, 10].plot()
plt.show()

#
# N = 100
# yexp = np.random.normal(100, 2, N)
# yexp_mean=np.mean(yexp)
# yexp_std=np.std(yexp)
#
# dz=0.01
# z = np.arange(-100, 100, dz)
# print(np.sum(yexp_std/np.sqrt(N)*np.abs(z/yexp_mean)*stats.t.pdf(z, N-1))*dz)
