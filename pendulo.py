'''
Integra la ecuacion del pendulo usando Runge-Kutta de orden 2.
'''

import numpy as np
import matplotlib.pyplot as plt


g = 9.8 # m/s/s
R = 1 # m

phi_0 = np.pi / 16
omega_0 = 0

# Plotea la solucion para pequenas oscilaciones

T = 2 * np.pi / np.sqrt(g / R)
t_to_plot = np.linspace(0, 4 * T, 300)
phi_pequenas_osc = phi_0 * np.cos(np.sqrt(g/R) * t_to_plot)

plt.clf()
plt.plot(t_to_plot, phi_pequenas_osc, label='Pequeñas oscilaciones')

plt.legend()
plt.show()