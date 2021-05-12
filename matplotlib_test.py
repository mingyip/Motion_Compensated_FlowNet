import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint
from scipy.misc import derivative

def system(vect, t):
    x, y = vect
    return [x - y - x * (x**2 + 5 * y**2), x + y - y * (x**2 + y**2)]

vect0 = [(-2 + 4*np.random.random(), -2 + 4*np.random.random()) for i in range(1)]
t = np.linspace(0, 100, 1000)

color=['red','green','blue','yellow', 'magenta']

plot = plt.figure()


for i, v in enumerate(vect0):
    sol = odeint(system, v, t)
    plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0]-sol[:-1, 0], sol[1:, 1]-sol[:-1, 1], scale_units='xy', angles='xy', scale=1, color=color[i])    

plt.show(plot)   