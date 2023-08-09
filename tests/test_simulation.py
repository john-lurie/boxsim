import unittest

import numpy as np

from boxsim import simulation

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.simulator = simulation.Simulator()

    def test_simulator_init(self):
        # Confirm default values.
        self.assertEqual(self.simulator.width, 100.0)
        self.assertEqual(self.simulator.radius, 1.0)
        pos_vel_tup = (self.simulator.pos_x, self.simulator.pos_y,
                       self.simulator.vel_x, self.simulator.vel_y)
        self.assertEqual(pos_vel_tup, (None, None, None, None))

    def test_make_particles(self):
        # Make a single particle.
        pos_vel = np.array([[1], [2], [3], [4]])
        self.assertEqual(pos_vel.shape, (4, 1))
        self.simulator.make_particles(pos_vel)
        self.assertEqual(self.simulator.pos_x, pos_vel[0])
        self.assertEqual(self.simulator.pos_y, pos_vel[1])
        self.assertEqual(self.simulator.vel_x, pos_vel[2])
        self.assertEqual(self.simulator.vel_y, pos_vel[3])

        # Raises ValueError because array shape should be (4, N), not (1, 4).
        one_row = np.array([[0, 0, 0, 0]])
        self.assertEqual(one_row.shape, (1, 4))
        with self.assertRaises(ValueError):        
            self.simulator.make_particles(one_row)

    def test_plotting(self):
        self.simulator = simulation.Simulator(radius=3)
        pos_vel = np.array([[50.0], [50.0], [2.0], [1.0]])
        self.simulator.make_particles(pos_vel)
        self.simulator.run()

    def test_outer_subract_self(self):
        # Try passing an array of shape (1, 3).
        two_dim = np.array([[0, 0, 0]])
        with self.assertRaises(ValueError):
            difference_matrix = simulation.outer_subtract_self(two_dim)
        
        array = np.array([1, 2, 3])
        difference_matrix = simulation.outer_subtract_self(array)
        expected_matrix = np.array([[ 0,  1,  2],
                                    [-1,  0,  1],
                                    [-2, -1,  0]]) 
        np.testing.assert_array_equal(difference_matrix, expected_matrix)
