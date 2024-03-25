#!/usr/bin/env python
"""Calculate the values of surface height (h) and east-west velocity
(u) in a dish of water where a point disturbance of h initiates waves.
Use the simplified shallow water equations on a non-staggered grid.

This is an implementation of lab7 section 4.3.

Example usage from the notebook::

from numlabs.lab7 import rain
# Run 5 time steps on a 9 point grid
rain.rain(5,9)

Example usage from the shell::

  # Run 5 time steps on a 9 point grid
  $ rain.py 5 9

The graph window will close as soon as the animation finishes.  And
the default run for 5 time steps doesn't produce much of interest; try
at least 100 steps.

Example usage from the Python interpreter::

  $ python
  ...
  >>> import rain
  >>> # Run 200 time steps on a 9 point grid
  >>> rain.rain((200, 9))
"""
from __future__ import division
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar
import os,glob
from matplotlib import animation

class Quantity(object):
    """Generic quantity to define the data structures.
    """
    def __init__(self, n_grid, n_time):
        """Initialize an object with prev, now, and next arrays of
        n_grid points, and a store array of n_time time steps.
        """
        self.n_grid = n_grid
        # Storage for values at previous, current, and next time step
        self.prev = np.empty(n_grid)
        self.now = np.empty(n_grid)
        self.next = np.empty(n_grid)
        # Storage for results at each time step.  In a bigger model
        # the time step results would be written to disk and read back
        # later for post-processing (such as plotting).
        self.store = np.empty((n_grid, n_time))


    def store_timestep(self, time_step, attr='next'):
        """Copy the values for the specified time step to the storage
        array.

        The `attr` argument is the name of the attribute array (prev,
        now, or next) that we are going to store.  Assigning the value
        'next' to it in the function def statement makes that the
        default, chosen because that is the most common use (in the
        time step loop).
        """
        # The __getattribute__ method let us access the attribute
        # using its name in string form;
        # i.e. x.__getattribute__('foo') is the same as x.foo, but the
        # former lets us change the name of the attribute to operate
        # on at runtime.
        self.store[:, time_step] = self.__getattribute__(attr)


    def shift(self):
        """Copy the .now values to .prev, and the .next values to .new.

        This reduces the storage requirements of the model to 3 n_grid
        long arrays for each quantity, which becomes important as the
        domain size and model complexity increase.  It is possible to
        reduce the storage required to 2 arrays per quantity.
        """
        # Note the use of the copy() method from the copy module in
        # the standard library here to get a copy of the array, not a
        # copy of the reference to it.  This is an important and
        # subtle aspect of the Python data model.
        self.prev = copy.copy(self.now)
        self.now = copy.copy(self.next)

    def get_value(self):
        return self.store

    def get_prev(self):
        return np.array(self.prev)

    def get_now(self):
        return np.array(self.now)

    def get_next(self):
        return np.array(self.next)


def initial_conditions(u, h, ho, setting='gaussian'):
    """Set the initial condition values.
    """
    if setting == 'point':
        u.prev[:] = 0
        h.prev[:] = 0
        h.prev[len(h.prev) // 2] = ho

    elif setting == 'gaussian':
        # Sets a gaussian initial condition with sigma = height/4
        def gaussian(x, sigma):
            d = len(h.prev) // 2
            return ho * np.exp(-(x - d)**2/(2*sigma))
        u.prev[:] = 0
        hlen = len(h.prev)
        sigma = hlen/10
        h.prev[:] = gaussian(np.linspace(0, hlen, hlen), sigma=sigma)

    else:
        raise ValueError("Invalid setting argument")


def boundary_conditions(u_array, h_array, n_grid):
    """Set the boundary condition values.
    """
    # Dirichlet BCs
    u_array[0] = 0 
    u_array[n_grid - 2] = 0 
    # u = 0 boundary point needs to be in-between h_n-1 and h_n neumann condition
    # Neumann BCs
    h_array[0] = h_array[1]
    h_array[n_grid-1] = h_array[n_grid-2]


def first_time_step(u, h, g, H, dt, dx, ho, gu, gh, n_grid, setting='general'):
    """Calculate the first time step values from the analytical
    predictor-corrector derived from equations 4.18 and 4.19.
    """
    if setting == 'point':
        u.now[1:n_grid - 1] = 0
        factor = gu * ho 
        midpoint = n_grid // 2
        u.now[midpoint - 1] = -factor
        u.now[midpoint] = factor
        h.now[1:n_grid - 1] = 0
        h.now[midpoint] = ho - g * H * ho * dt ** 2 / (dx ** 2)

    elif setting == 'general':
        # Calculate predicted u, h
        u1 = np.zeros(n_grid)
        h1 = np.zeros(n_grid)
        # Spatial second derivatives for staggered grid same as the leap-frog equations
        u1[1:n_grid - 1] = u.prev[1:n_grid - 1] - gu * (h.prev[2:n_grid] - h.prev[1:n_grid - 1])
        h1[1:n_grid - 1] = h.prev[1:n_grid - 1] - gh * (u.prev[1:n_grid - 1] - u.prev[:n_grid - 2])
        h1[0] = h1[1]
        h1[-1] = h1[-2]
        # Average to find u, h at dt/2
        u05 = 0.5*(u.prev + u1)
        h05 = 0.5*(h.prev + h1)
        # Staggered entre-difference corrector scheme with half-step dx/2
        u.now[1:n_grid - 1] = u.prev[1:n_grid - 1] - gu * (h05[2:n_grid] - h05[1:n_grid - 1])
        h.now[1:n_grid - 1] = h.prev[1:n_grid - 1] - gh * (u05[1:n_grid - 1] - u05[:n_grid - 2])
        u.now[0] = 0
        u.now[-2] = 0 # Same as in boudary_conditions
        h.now[0] = h.now[1]
        h.now[-1] = h.now[-2]


def leap_frog(u, h, gu, gh, n_grid): 
    """Calculate the next time step values using the leap-frog scheme
    derived from equations 4.16 and 4.17.
    """
#    for pt in np.arange(1, n_grid - 1):
#        u.next[pt] = u.prev[pt] - 2 * gu * (h.now[pt + 1] - h.now[pt])
#        h.next[pt] = h.prev[pt] - 2 * gh * (u.now[pt] - u.now[pt - 1])
#     Alternate vectorized implementation:
    u.next[1:n_grid - 1] = (u.prev[1:n_grid - 1]
                            - 2*gu * (h.now[2:n_grid] - h.now[1:n_grid - 1]))
    h.next[1:n_grid - 1] = (h.prev[1:n_grid - 1]
                            - 2*gh * (u.now[1:n_grid - 1] - u.now[:n_grid - 2]))




def make_graph(u, h, dt, n_time):
    """Create graphs of the model results using matplotlib.

    You probably need to run the rain script from within ipython,
    in order to see the graphs.  And
    the default run for 5 time steps doesn't produce much of interest;
    try at least 100 steps.
    """

    # Create a figure with 2 sub-plots
    fig, (ax_u, ax_h) = plt.subplots(2,1, figsize=(10,10))

    # Set the figure title, and the axes labels.
    the_title = fig.text(0.25, 0.95, 'Results from t = %.3fs to %.3fs' % (0, dt*n_time))
    ax_u.set_ylabel('u [cm/s]')
    ax_h.set_ylabel('h [cm]')
    ax_h.set_xlabel('Grid Point')

    # We use color to differentiate lines at different times.  Set up the color map
    cmap = plt.get_cmap('viridis')
    cNorm  = colors.Normalize(vmin=0, vmax=1.*n_time)
    cNorm_inseconds = colors.Normalize(vmin=0, vmax=1.*n_time*dt)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # Only try to plot 20 lines, so choose an interval if more than that (i.e. plot
    # every interval lines
    interval = int(np.ceil(n_time/20))

    # Do the main plot
    for time in range(0, n_time, interval):
        colorVal = scalarMap.to_rgba(time)
        ax_u.plot(u.store[:, time], color=colorVal)
        ax_h.plot(h.store[:, time], color=colorVal)

    # Add the custom colorbar
    ax2 = fig.add_axes([0.95, 0.05, 0.05, 0.9])
    cb1 = colorbar.ColorbarBase(ax2, cmap=cmap, norm=cNorm_inseconds)
    cb1.set_label('Time (s)')
    return




def rain(args, setting='point'):
    """Run the model.

    args is a 2-tuple; (number-of-time-steps, number-of-grid-points)
    setting takes in one of ('point', 'gaussian') for setting up the initial conditions
    """
    n_time = int(args[0])
    n_grid = int(args[1])
#     Alternate implementation:
#     n_time, n_grid = map(int, args)

    # Constants and parameters of the model
    g = 980                     # acceleration due to gravity [cm/s^2]
    H = 1                       # water depth [cm]
    dt = 0.01                  # time step [s]
    dx = 1                      # grid spacing [cm]
    ho = 0.01                   # initial perturbation of surface [cm]
    gu = g * dt / dx            # first handy constant
    gh = H * dt / dx            # second handy constant
    # Create velocity and surface height objects
    u = Quantity(n_grid, n_time) 
    h = Quantity(n_grid, n_time)
    # Set up initial conditions and store them in the time step
    # results arrays
    initial_conditions(u, h, ho, setting=setting)
    u.store_timestep(0, 'prev')
    h.store_timestep(0, 'prev')
    # Calculate the first time step values from the
    # predictor-corrector, apply the boundary conditions, and store
    # the values in the time step results arrays
    first_time_step(u, h, g, H, dt, dx, ho, gu, gh, n_grid)
    boundary_conditions(u.now, h.now, n_grid)
    u.store_timestep(1, 'now')
    h.store_timestep(1, 'now')
    # Time step loop using leap-frog scheme
    for t in np.arange(2, n_time):
        # Advance the solution and apply the boundary conditions
        leap_frog(u, h, gu, gh, n_grid)
        boundary_conditions(u.next, h.next, n_grid)
        # Store the values in the time step results arrays, and shift
        # .now to .prev, and .next to .now in preparation for the next
        # time step
        u.store_timestep(t)
        h.store_timestep(t)
        u.shift()
        h.shift()

    # Plot the results as colored graphs

    make_graph(u, h, dt, n_time)
    return u, h, dt, n_time



if __name__ == '__main__':
    # sys.argv is the command-line arguments as a list. It includes
    # the script name as its 0th element. Check for the degenerate
    # cases of no additional arguments, or the 0th element containing
    # `sphinx-build`. The latter is a necessary hack to accommodate
    # the sphinx plot_directive extension that allows this module to
    # be run to include its graph in sphinx-generated docs.
    #
    #  the following command, executed in the plotfile directory makes a movie on ubuntu called
    #   outputmplt.avi
    #  which can be
    #  looped with mplayer -loop 0
    #
    #  mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o outputmplt.avi
    #
    if len(sys.argv) == 1 or 'sphinx-build' in sys.argv[0]:
        # Default to 50 time steps, and 9 grid points
        rain((50, 9))
        plt.show()
    elif len(sys.argv) == 3:
        # Run with the number of time steps and grid point the user gave
        rain(sys.argv[1:])
        plt.show()
    else:
        print ('Usage: rain n_time n_grid')
        print ('n_time = number of time steps; default = 5')
        print ('n_grid = number of grid points; default = 9')
