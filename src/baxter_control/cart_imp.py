import rospy
import numpy as np

class PID(object):
    def __init__(self, goal=0.0, trajectory=0.0,
                 null_damping=0.0, null_stiffness=0.0,
                 cart_stiffness={}, cart_damping={}):

        # Damping and stiffness parameters
        self._null_damping = null_damping
        self._null_stiffness = null_stiffness
        self._cart_stiffness = cart_stiffness
        self._cart_damping = cart_damping

        self._prev_err = 0.0

        self._cur_time = 0.0
        self._prev_time = 0.0

        self.initialize()

    def initialize(self):
        self._prev_err = 0.0
        self._cur_time = rospy.get_time()
        self._prev_time = self._cur_time

    def set_goal(self, goal):
        self._goal = goal

    def set_trajectory(self, trajectory):
        self._trajectory = trajectory

    def set_null_damping(self, null_damping):
        self._null_damping = null_damping

    def set_null_stiffness(self, null_stiffness):
        self._null_stiffness = null_stiffness

    def set_cart_stiffness(self, cart_stiffness):
        self._cart_stiffness = cart_stiffness

    def set_cart_damping(self, cart_damping):
        self._cart_damping = cart_damping

    def compute_output(self, error):
        self._cur_time = rospy.get_time()
        dt = self._cur_time - self._prev_time
        de = error - self._prev_err

        # Compute PID output here using self._goal and self._trajectory
        # ...

        # Optionally use self._null_damping, self._null_stiffness,
        # self._cart_stiffness, and self._cart_damping for more advanced control
        # ...

        self._prev_time = self._cur_time
        self._prev_err = error

        # Return the computed output
        return # ...
