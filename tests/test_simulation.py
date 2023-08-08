import unittest

import numpy as np

from boxsim import simulation

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.simulator = simulation.Simulator()

    def tearDown(self):
        pass

    def test_initialize_particles(self):
        # Positions are supplied as an array.
        positions_x = np.array([1, 2, 3])
        positions_y = np.array([4, 5, 6])
        velocities_x = np.array([-1, 0, 1])
        velocities_y = np.array([1, 0, -1])
        pos_vel_stack = simulation.stack_pos_vel(positions_x, positions_y,
                                                 velocities_x, velocities_y)
        self.assertEqual(pos_vel_stack.shape, (4, 3))
        self.simulator.make_particles(positions_velocities=pos_vel_stack)

        # Raises ValueError because array shape should be (2, N). 
        bad_positions = np.array([[1, 2, 3, 4, 5]])
        self.assertEqual(bad_positions.shape, (1, 5))
        with self.assertRaises(ValueError):        
            self.simulator.make_particles(positions_velocities=bad_positions)

        # If no positions are given, draw them randomly.
        self.simulator.make_particles()
        self.assertEqual(self.simulator.positions_x.shape, (50,))
        self.assertEqual(self.simulator.positions_y.shape, (50,))

    def test_outer_subract_self(self):
        # Try passing an array of shape (1, 3).
        two_dimensions = np.array([[0, 0, 0]])
        message = "Expected array of dimension 1, got: 2"
        with self.assertRaisesRegex(ValueError, message):
            difference_matrix = simulation.outer_subtract_self(two_dimensions)
        
        row_vector = np.array([1, 2, 3])
        difference_matrix = simulation.outer_subtract_self(row_vector)
        expected_matrix = np.array([[ 0,  1,  2],
                                    [-1,  0,  1],
                                    [-2, -1,  0]]) 
        np.testing.assert_array_equal(difference_matrix, expected_matrix)
