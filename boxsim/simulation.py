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


def vector_magnitude(component_x, component_y):
    """Return the vector magnitude given x- and y-components."""
    return np.sqrt(component_x**2 + component_y**2)


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

    def reflect_walls(self):
        """Any particle outside of walls is reflected."""
        # Logical not (~) makes inequality simpler.
        outside_x = ~ (self.radius < self.pos_x < self.width - self.radius)
        outside_y = ~ (self.radius < self.pos_y < self.width - self.radius)
        self.vel_x[outside_x] *= -1
        self.vel_y[outside_y] *= -1

    def move_particles(self, timestep):
        """Multiply velocities by a given timestep to update positions."""
        self.pos_x += self.vel_x * timestep
        self.pos_y += self.vel_y * timestep

    def setup_plot(self, show=False):
        """Setup the plotting window."""
        self.fig, self.ax = plt.subplots(dpi=120)
        self.ax.set_xlim(0.0, self.width)
        self.ax.set_ylim(0.0, self.width)
        self.ax.set_aspect(1)

        # Set marker size to be in data units of current window.
        # If window is resized, marker will not be in data units.
        self.fig.canvas.draw()
        window = self.ax.get_window_extent().width
        ms =  (2 * self.radius / self.width) * window * (72 / self.fig.dpi)
        self.plotter, = self.ax.plot(self.pos_x, self.pos_y, marker='o', ms=ms,
                                     mew=0, color='k')
        if show:
            plt.show()

    def run(self, duration=10.0, animate=False):
        """Run the simulation."""
        if animate:
            plt.ion()

        time = 0.0
        while time <= duration:
            speed = vector_magnitude(self.vel_x, self.vel_y)

            if np.max(speed) == 0.0:
                print("No particles are moving. Simulation over!")
                break
            else:
                # Set timestep so that fastest particle moves 0.2 * radius.
                # This will result in a long simulation if speed is low.
                timestep = 0.2 * self.radius / np.max(speed)

                if animate:
                    self.plotter.set_data(self.pos_x, self.pos_y)
                    plt.pause(timestep)

                self.reflect_walls()
                self.move_particles(timestep)
                time += timestep
