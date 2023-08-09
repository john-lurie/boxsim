"""Simulator and supporting functions"""
import matplotlib.pyplot as plt
import numpy as np

def outer_subtract_self(array):
    """Compute the outer subtraction between an array and itself."""
    rank = array.ndim
    
    if rank == 1:
        return array - array[:, np.newaxis]
    else:
        raise ValueError(f"Expected array of dimension 1, got: {rank}")


class Simulator:
    """Two-dimensional gas particle simulator."""
    def __init__(self, width=100.0, radius=1.0):

        self.width = width
        self.radius = radius

        self.pos_x = None
        self.pos_y = None
        self.vel_x = None
        self.vel_y = None
        
        self.fig = None
        self.ax = None
        self.plotter = None

    def make_particles(self, pos_vel=None):
        """
        Initialize the particles.
        
        If positions and velocities are not specified, they will be chosen
        from a random uniform distribution.
        """
        msg = f"Expected 'pos_vel' to have shape (4, N), got: {pos_vel.shape}"
        
        if pos_vel is not None:
            if pos_vel.ndim == 2 and pos_vel.shape[0] == 4:
                self.pos_x = pos_vel[0]
                self.pos_y = pos_vel[1]
                self.vel_x = pos_vel[2]
                self.vel_y = pos_vel[3]
            else:
                raise ValueError(msg)
        else:
            # FUTURE: Draw from random uniform distribution.
            pass

    def run(self):
        """Run the simulation."""
        self.fig, self.ax = plt.subplots(dpi=120)
        self.ax.set_xlim(0.0, self.width)
        self.ax.set_ylim(0.0, self.width)
        self.ax.set_aspect(1)
        self.fig.canvas.draw()
        
        window_ext = self.ax.get_window_extent().width
        ms =  (self.radius / self.width) * window_ext * (72.0 / self.fig.dpi)
        self.plotter, = self.ax.plot([], [], marker='o', ms=ms, mew=0,
                                     color='k')
        
        self.plotter.set_data(self.pos_x, self.pos_y)
        
        plt.show()
