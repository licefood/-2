import numpy as np
import requests
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
import os


class RCS:

    def __init__(self, r):
        self.r = r

    def calculate_rcs(self, f):
        w = 3e8 / f
        k = 2 * np.pi / w
        kr = k * self.r

        def h_func(n, x):
            return spherical_jn(n, x) + 1j * spherical_yn(n, x)

        def a_func(n, kr):
            return spherical_jn(n, kr) / h_func(n, kr)

        def b_func(n, kr):
            b = kr * spherical_jn(n - 1, kr) - n * spherical_jn(n, kr)
            d = kr * h_func(n - 1, kr) - n * h_func(n, kr)
            return b / d

        result = 0
        for n in range(1, 30):
            term = ((-1) ** n) * (n + 0.5) * (b_func(n, kr) - a_func(n, kr))
            result += term

        return (w ** 2 / np.pi) * (np.abs(result) ** 2)


class vivod:

    def save(filename, f_values, results):
        with open(filename, 'w') as file:
            file.write('     f            rcs\n')
            for x, y in zip(f_values, results):
                file.write(f'{x:.4f}    {y:.4f}\n')


url = "https://jenyay.net/uploads/Student/Modelling/task_rcs_02.txt"
response = requests.get(url)
with open('file.txt', 'wb') as file:
    file.write(response.content)

with open('file.txt') as f:
    M = f.read().split()

D = float(M[38])
fmin = float(M[39])
fmax = float(M[40])


f = np.linspace(fmin,fmax,500)
rcs_calculator = RCS(D / 2)
r = []

for F in f:
    rcs = rcs_calculator.calculate_rcs(F)
    r.append(rcs)

if not os.path.exists('results'):
    os.makedirs('results')

vivod.save('results/results.txt', f, r)

plt.figure(figsize=(16, 9))
plt.plot(f, r, label='f(x)', color='g')
plt.title('График')
plt.xlabel('F')
plt.ylabel('rcs')
plt.grid(True)
plt.show()
