"""Simulator and supporting functions"""
import numpy as np

def outer_subtract_self(array):
    """Compute the outer subtraction between an array and itself."""
    rank = array.ndim
    
    if rank == 1:
        return array - array[:, np.newaxis]
    else:
        raise ValueError(f"Expected array of dimension 1, got: {rank}")


def stack_pos_vel(positions_x, positions_y, velocities_x, velocities_y):
    """Stack position and velocity arrays."""
    return np.vstack((positions_x, positions_y, velocities_x, velocities_y))


class Simulator:
    """Two-dimensional gas particle simulator."""
    def __init__(self, side_length=10.0, relative_radius=0.01, n_particles=50,
                 seed=999):
        
        self.side_length = side_length
        self.particle_radius = side_length * relative_radius
        self.n_particles = n_particles
        # Random number generator
        self.rng = np.random.default_rng(seed)
        # Edges of the box including particle_radius
        # Box is a square, so inner and outer same for x and y.
        self.inner_edge = 0.0 + self.particle_radius
        self.outer_edge = side_length - self.particle_radius
        
        self.positions_x = None
        self.positions_y = None

    def make_particles(self, positions_velocities=None):
        """
        Initialize the particles.
        
        If positions and velocities are not specified, they will be
        chosen from a random uniform distribution.
        """
        if positions_velocities is not None:
            if positions_velocities.ndim == 2 and positions_velocities.shape[0] == 4:
                self.positions_x = positions_velocities[0]
                self.positions_y = positions_velocities[1]
                self.velocities_x = positions_velocities[2]
                self.velocities_y = positions_velocities[3]
            else:
                raise ValueError(f"Expected 'positions' to have shape "
                                 "(4, N), got: {positions.shape}")
        else:
            self.positions_x = self.rng.uniform(self.inner_edge, 
                                                self.outer_edge,
                                                self.n_particles)
            self.positions_y = self.rng.uniform(self.inner_edge,
                                                self.outer_edge, 
                                                self.n_particles)
