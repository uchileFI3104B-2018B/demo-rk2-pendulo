'''
Integra la ecuacion del pendulo usando Runge-Kutta de orden 2.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton


g = 9.8 # m/s/s
R = 1 # m

phi_0 = np.pi / 2
omega_0 = 0

# Plotea la solucion para pequenas oscilaciones

T = 2 * np.pi / np.sqrt(g / R)
t_to_plot = np.linspace(0, 4 * T, 300)
phi_pequenas_osc = phi_0 * np.cos(np.sqrt(g/R) * t_to_plot)

plt.clf()
plt.plot(t_to_plot, phi_pequenas_osc, label='Pequeñas oscilaciones')

# Implementa RK2

def pendulo(t, y):
    """
    Ecuacion del pendulo.
    Inputs:
    =======
    t : [float] tiempo en el cual se evalua la funcion.
    y : [np.ndarray] corresponden a phi, omega.

    Output:
    =======
    output: [lista de 2 elementos float] corresponden a [omega, -g/R sin(phi)].
    """
    output = np.array([y[1], -g / R * np.sin(y[0])])
    return output

def calcula_k1(func, paso, t, y):
    """
    Inputs:
    =======
    func: [funcion, outputs np.ndarray] la funcion a integrar.
    paso: [float] tamaño del paso temporal
    t : [float] tiempo
    y : [np.ndarray]
    """
    output = paso * func(t, y)
    return output

def calcula_k2(func, paso, t, y):
    """
    Uds rellenan aca.
    """
    k1 = calcula_k1(func, paso, t, y)
    output = paso * func(t + paso/2, y + k1 / 2)
    return output

def paso_rk2(func, paso, t, y):
    k2 = calcula_k2(func, paso, t, y)
    y_new = y + k2
    return y_new


y_rk2 = np.zeros((len(t_to_plot), 2))
y_rk2[0] = phi_0, omega_0

h = t_to_plot[1] - t_to_plot[0]

for i in range(1, len(t_to_plot)):
    y_rk2[i] = paso_rk2(pendulo, h, t_to_plot[i-1], y_rk2[i-1])

plt.plot(t_to_plot, y_rk2[:, 0], label='Runge-Kutta 2')

plt.legend()
plt.show()

# plt.savefig('comparacion-pendulo.png')

phi_interpolant = interp1d(t_to_plot, y_rk2[:,0])
t_packed = np.linspace(0.5, 2, 1000)

primer_cero = newton(phi_interpolant, 0.5)
segundo_cero = newton(phi_interpolant, 3)

plt.figure(2)
plt.clf()

plt.plot(t_to_plot, y_rk2[:, 0], 'o')
plt.plot(t_packed, phi_interpolant(t_packed), '.')
plt.axvline(primer_cero)
plt.axvline(segundo_cero)
plt.axhline(0)

plt.show()

print("El periodo del pendulo es {:.2f} seg".format((segundo_cero - 
                                                     primer_cero)))