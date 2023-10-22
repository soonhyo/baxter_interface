import rospy
import numpy as np
from pyquaternion import Quaternion

class CartIMP(object):
    def __init__(self, goal=0.0, trajectory=0.0,
                 null_damping=1.0, null_stiffness=0.0,
                 cart_stiffness=[200, 200, 200, 20, 20, 20],
                 cart_damping=[1, 1, 1, 1, 1, 1],
                 name=None,
                 kin=None,
                 limb=None):

        # Damping and stiffness parameters
        self._null_damping = null_damping
        self._null_stiffness = null_stiffness
        self._cart_stiffness = np.array(cart_stiffness, dtype=np.int)
        self._cart_damping = np.array(cart_damping, dtype=np.int)

        self._name = name
        self._kin = kin
        self._limb = limb
        self.jacobian = self._kin.jacobian()

        self._prev_err = 0.0

        self._cur_time = 0.0
        self._prev_time = 0.0


        self.joint_names = self._limb.joint_names()
        self.n_joints = len(self.joint_names)

    def initialize(self):
        self._prev_err = 0.0
        self._cur_time = rospy.get_time()
        self._prev_time = self._cur_time
        self._springs = {self._name + '_s0': 200,
                         self._name + '_s1': 220,
                         self._name + '_e0': 120,
                         self._name + '_e1': 50,
                         self._name + '_w0': 20,
                         self._name + '_w1': 20,
                         self._name + '_w2': 15}

        self._damping = {self._name + '_s0': 2,
                         self._name + '_s1': 2,
                         self._name + '_e0': 2,
                         self._name + '_e1': 2,
                         self._name + '_w0': 2,
                         self._name + '_w1': 2,
                         self._name + '_w2': 2}

        self.cartesian_stiffness = np.eye(6)
        for i, s in enumerate(self._cart_stiffness):
            self.cartesian_stiffness[i][i] = s

        self.cartesian_damping = np.eye(6)
        for i, d in enumerate(self._cart_damping):
            self.cartesian_damping[i][i] = d
        self.nullspace_stiffness = 50
        self.nullspace_damping = 1.0
        print("cartesian_stiffness ", self.cartesian_stiffness)
        print("cartesian_damping ", self.cartesian_damping)
        print("nullspace_stiffness", self.nullspace_stiffness)
        print("nullspace_damping", self.nullspace_damping)

    def set_null_damping(self, null_damping):
        self._null_damping = null_damping

    def set_null_stiffness(self, null_stiffness):
        self._null_stiffness = null_stiffness

    def set_cart_stiffness(self, cart_stiffness):
        self._cart_stiffness = cart_stiffness

    def set_cart_damping(self, cart_damping):
        self._cart_damping = cart_damping

    def set_cart_damping(self, jacobian):
        self.jacobian = jacobian

    def get_fk(self, q):
        ee_state = self._kin.forward_position_kinematics(q)
        position= np.asarray(ee_state[:3])
        orientation = np.asarray([ee_state[-1], ee_state[3], ee_state[4], ee_state[5]])
        return position, orientation

    def calculate_orientation_error(self, orientation_1_np, orientation_2_np):
        orientation_1_q = Quaternion(orientation_1_np)
        orientation_2_q = Quaternion(orientation_2_np)
        if np.dot(orientation_1_np, orientation_2_np) < 0.0:
            orientation_2_q = -1 * orientation_2_q
        error_quaternion = orientation_1_q * orientation_2_q.inverse
        error_quaternion_angle_axis = error_quaternion.angle * error_quaternion.axis
        return error_quaternion_angle_axis


    def update(self, q, dq):
        # self.q = self._limb.joint_angles()
        # self.dq = self._limb.joint_velocities()
        self.q = q
        self.dq = dq

        self.position, self.orientation = self.get_fk(self.q)

        # # end_pose = self._limb.endpoint_pose()
        # # self.position = [end_pose["position"].x, end_pose["position"].y, end_pose["position"].z]
        # # self.orientation = [end_pose["orientation"].w, end_pose["orientation"].x,end_pose["orientation"].y,end_pose["orientation"].z]

        self.jacobian = self._kin.jacobian(self.q)
        self.jacobian_transpose_pinv = np.linalg.pinv(self.jacobian.T)
        # self.cart_inertia = self._kin.cart_inertia(self.q)
        # self.coriolis = self._kin.coriolis_matrix(self.q, self.dq)

    def compute_output_(self, q_d):
        # current state
        cmd = dict()
        for joint in self.joint_names:
            # spring portion
            cmd[joint] = self._springs[joint] * (q_d[joint] -
                                                 self.q[joint])
            # # damping portion
            cmd[joint] -= self._damping[joint] * self.dq[joint]
        # print(cmd)
        return cmd

    def compute_output(self, q_d):
        # current state
        cmd = dict()

        position_d, orientation_d = self.get_fk(q_d)

        joint_name = q_d.keys()
        q_d = np.asarray(list(q_d.values()))
        q = np.asarray(list(self.q.values()))
        dq = np.asarray(list(self.dq.values()))


        # Compute error term
        error = np.zeros(6)
        error[:3] = self.position - position_d
        error[3:] = self.calculate_orientation_error(self.orientation, orientation_d)
        error = error[np.newaxis]
        # Kinematic pseudo-inverse

        # Initialize torque vectors
        tau_task = np.zeros(self.n_joints)
        tau_nullspace = np.zeros(self.n_joints)
        # tau_ext = np.zeros(self.n_joints)
        # Calculate task torque
        tau_task = self.jacobian.T.dot(-self.cartesian_stiffness.dot(error.T) - self.cartesian_damping.dot(self.jacobian.dot(dq.T).T)).T

        # Calculate nullspace torque
        nullspace_projector = np.eye(self.n_joints) - self.jacobian.T.dot(self.jacobian_transpose_pinv)
        tau_nullspace = nullspace_projector.dot(self.nullspace_stiffness * (q_d -q) - self.nullspace_damping * dq)

        # # Calculate external torque
        # tau_ext = self.jacobian.T.dot(self.cartesian_wrench)

        tau_d = np.asarray(tau_task + tau_nullspace).squeeze()
        # tau_d = np.asarray(tau_task).squeeze()

        #tau_d = tau_task
        # self.tau_c = self.saturate_torque_rate(tau_d, self.tau_c, self.delta_tau_max)
        for idx, joint in enumerate(self.joint_names):
            # spring portion
            cmd[joint] = np.clip(tau_d[idx], -1, 1)
        # rospy.loginfo(cmd)

        return cmd
