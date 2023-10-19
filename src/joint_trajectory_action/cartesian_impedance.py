#!/usr/bin/env python3

import numpy as np
from scipy.linalg import pinv


def calculate_orientation_error(orientation_d, orientation):
    if np.dot(orientation_d.coeffs(), orientation.coeffs()) < 0.0:
        orientation.coeffs()[:] = -orientation.coeffs()
    error_quaternion = orientation * orientation_d.inverse()
    error_quaternion_angle_axis = error_quaternion.to_rotation_vector()
    return error_quaternion_angle_axis


def filtered_update(target, current, filter):
    return (1.0 - filter) * current + filter * target

def filter_step(update_frequency, filter_percentage):
    kappa = -1 / np.log(1 - min(filter_percentage, 0.999999))
    return 1.0 / (kappa * update_frequency + 1.0)

def saturate_value(x, x_min, x_max):
    return max(min(x, x_max), x_min)


def saturate_torque_rate(tau_d_calculated, tau_d_saturated, delta_tau_max):
    for i in range(len(tau_d_calculated)):
        difference = tau_d_calculated[i] - tau_d_saturated[i]
        tau_d_saturated[i] += saturate_value(difference, -delta_tau_max, delta_tau_max)

def damping_rule(stiffness):
    return 2 * math.sqrt(stiffness)


class CartesianImpedanceController:
    def __init__(self):
        self.set_stiffness(np.array([200, 200, 200, 20, 20, 20, 0]))
        self.cartesian_stiffness = self.cartesian_stiffness_target
        self.cartesian_damping = self.cartesian_damping_target

    def init_desired_pose(self, position_d_target, orientation_d_target):
        self.set_reference_pose(position_d_target, orientation_d_target)
        self.position_d_ = position_d_target
        self.orientation_d_ = orientation_d_target

    def init_nullspace_config(self, q_d_nullspace_target):
        self.set_nullspace_config(q_d_nullspace_target)
        self.q_d_nullspace = q_d_nullspace_target

    def set_number_of_joints(self, n_joints):
        self.n_joints = n_joints
        self.q = np.zeros(self.n_joints)
        self.dq = np.zeros(self.n_joints)
        self.jacobian = np.zeros((6, self.n_joints))
        self.q_d_nullspace = np.zeros(self.n_joints)
        self.q_d_nullspace_target = self.q_d_nullspace
        self.tau_c = np.zeros(self.n_joints)

    def set_stiffness(self, stiffness, auto_damping=True):
        assert all(v >= 0 for v in stiffness), "Stiffness values need to be positive."
        self.cartesian_stiffness_target = np.diag(stiffness[:6])
        self.nullspace_stiffness_target = stiffness[6]
        if auto_damping:
            self.applyDamping()

    def set_stiffness_(self, t_x, t_y, t_z, r_x, r_y, r_z, n, auto_damping=True):
        stiffness_vector = np.array([t_x, t_y, t_z, r_x, r_y, r_z, n])
        self.set_stiff_ness(stiffness_vector, auto_damping)

    def set_stiffness_target(self, t_x, t_y, t_z, r_x, r_y, r_z, auto_damping=True):
        stiffness_vector = np.array([t_x, t_y, t_z, r_x, r_y, r_z, self.nullspace_stiffness_target])
        self.set_stiff_ness(stiffness_vector, auto_damping)

    def setDampingFactors(self, d_x, d_y, d_z, d_a, d_b, d_c, d_n):
        damping_new = np.array([d_x, d_y, d_z, d_a, d_b, d_c, d_n])
        for i in range(len(damping_new)):
            if damping_new[i] < 0:
                damping_new[i] = self.damping_factors[i]
        self.damping_factors = damping_new
        self.apply_damping()

    def apply_damping(self):
        for i in range(6):
            assert self.damping_factors[i] >= 0, "Damping values need to be positive."
            self.cartesian_damping_target[i, i] = self.damping_factors[i] * damping_rule(self.cartesian_stiffness_target[i, i])
        assert self.damping_factors[6] >= 0, "Damping values need to be positive."
        self.nullspace_damping_target = self.damping_factors[6] * damping_rule(self.nullspace_stiffness_target)

    def set_reference_pose(self, position_d_target, orientation_d_target):
        self.position_d_target = np.array(position_d_target)
        self.orientation_d_target = np.array(orientation_d_target.coeffs())  # Assumed to have a coeffs() method
        self.orientation_d_target /= np.linalg.norm(self.orientation_d_target)

    def set_nullspace_config(self, q_d_nullspace_target):
        assert len(q_d_nullspace_target) == self.n_joints, "Nullspace target needs to be the same size as n_joints_"
        self.q_d_nullspace_target = np.array(q_d_nullspace_target)

    def set_filtering(self, update_frequency, filter_params_nullspace_config, filter_params_stiffness,
                     filter_params_pose, filter_params_wrench):
        self.set_update_frequency(update_frequency)
        self.setFilterValue(filter_params_nullspace_config, self.filter_params_nullspace_config)
        self.setFilterValue(filter_params_stiffness, self.filter_params_stiffness)
        self.setFilterValue(filter_params_pose, self.filter_params_pose)
        self.setFilterValue(filter_params_wrench, self.filter_params_wrench)

    def set_max_torque_delta(self, d, update_frequency=None):
        assert d >= 0, "Allowed torque change must be positive"
        self.delta_tau_max = d
        if update_frequency is not None:
            self.set_update_frequency(update_frequency)

    def apply_wrench(self, cartesian_wrench_target):
        self.cartesian_wrench_target = np.array(cartesian_wrench_target)

    def calculate_commanded_torques(self, q, dq, position, orientation, jacobian):
        self.q = np.array(q)
        self.dq = np.array(dq)
        self.position = np.array(position)
        self.orientation = np.array(orientation.coeffs())  # Assumed to have a coeffs() method
        self.jacobian = np.array(jacobian)

        return self.calculate_commanded_torques()  # Assuming this is defined elsewhere

    def calculate_commanded_torques(self):
        # Perform a filtering step
        self.update_filtered_nullspace_config()
        self.update_filtered_stiffness()
        self.update_filtered_pose()
        self.update_filtered_wrench()

        # Compute error term
        error = np.zeros(6)
        error[:3] = self.position - self.position_d
        error[3:] = self.calculate_orientation_error(self.orientation_d, self.orientation)  # Assuming this method exists

        # Kinematic pseudo-inverse
        jacobian_transpose_pinv = np.linalg.pinv(self.jacobian.T)

        # Initialize torque vectors
        tau_task = np.zeros(self.n_joints)
        tau_nullspace = np.zeros(self.n_joints)
        tau_ext = np.zeros(self.n_joints)

        # Calculate task torque
        tau_task = self.jacobian.T.dot(-self.cartesian_stiffness.dot(error) - self.cartesian_damping.dot(self.jacobian.dot(self.dq)))

        # Calculate nullspace torque
        nullspace_projector = np.eye(self.n_joints) - self.jacobian.T.dot(jacobian_transpose_pinv)
        tau_nullspace = nullspace_projector.dot(self.nullspace_stiffness * (self.q_d_nullspace - self.q) - self.nullspace_damping * self.dq)

        # Calculate external torque
        tau_ext = self.jacobian.T.dot(self.cartesian_wrench)

        # Calculate the commanded torque
        tau_d = tau_task + tau_nullspace + tau_ext
        self.tau_c = self.saturate_torque_rate(tau_d, self.tau_c, self.delta_tau_max)  # Assuming this method exists

        return self.tau_c

    def get_state(self):
        q = np.copy(self.q_)
        dq = np.copy(self.dq_)
        position = np.copy(self.position_)
        orientation = self.orientation_.coeffs()  # Assuming coeffs() returns the quaternion coefficients
        position_d, orientation_d, cartesian_stiffness, nullspace_stiffness, q_d_nullspace, cartesian_damping = self.get_state_aux()

        return q, dq, position, orientation, position_d, orientation_d, cartesian_stiffness, nullspace_stiffness, q_d_nullspace, cartesian_damping

    def get_state_aux(self):
        position_d = np.copy(self.position_d_)
        orientation_d = self.orientation_d_.coeffs()  # Assuming coeffs() returns the quaternion coefficients
        cartesian_stiffness = np.copy(self.cartesian_stiffness_)
        nullspace_stiffness = self.nullspace_stiffness_
        q_d_nullspace = np.copy(self.q_d_nullspace_)
        cartesian_damping = np.copy(self.cartesian_damping_)

        return position_d, orientation_d, cartesian_stiffness, nullspace_stiffness, q_d_nullspace, cartesian_damping


    def get_last_commands(self):
        return np.copy(self.tau_c_)

    def get_applied_wrench(self):
        return np.copy(self.cartesian_wrench_)

    def get_pose_error(self):
        return np.copy(self.error_)

    def set_update_frequency(self, freq):
        assert freq >= 0, "Update frequency needs to be greater or equal to zero"
        self.update_frequency_ = max(freq, 0.0)

    def set_filter_value(self, val, saved_val):
        assert 0 < val <= 1.0, "Filter params need to be between 0 and 1."
        saved_val = saturate_value(val, 0.0000001, 1.0)

    def filter_step(self, update_frequency, filter_param):
        return 1.0 - math.exp(-update_frequency / filter_param)

    def filtered_update(self, target, current, step):
        return (1.0 - step) * current + step * target

    def update_filtered_nullspace_config(self):
        if self.filter_params_nullspace_config_ == 1.0:
            self.q_d_nullspace_ = np.copy(self.q_d_nullspace_target_)
        else:
            step = self.filter_step(self.update_frequency_, self.filter_params_nullspace_config_)
            self.q_d_nullspace_ = self.filtered_update(self.q_d_nullspace_target_, self.q_d_nullspace_, step)

    def update_filtered_stiffness(self):
        if self.filter_params_stiffness_ == 1.0:
            self.cartesian_stiffness_ = np.copy(self.cartesian_stiffness_target_)
            self.cartesian_damping_ = np.copy(self.cartesian_damping_target_)
            self.nullspace_stiffness_ = self.nullspace_stiffness_target_
        else:
            step = self.filter_step(self.update_frequency_, self.filter_params_stiffness_)
            self.cartesian_stiffness_ = self.filtered_update(self.cartesian_stiffness_target_, self.cartesian_stiffness_, step)
            self.cartesian_damping_ = self.filtered_update(self.cartesian_damping_target_, self.cartesian_damping_, step)
            self.nullspace_stiffness_ = self.filtered_update(self.nullspace_stiffness_target_, self.nullspace_stiffness_, step)

    def update_filtered_pose(self):
        if self.filter_params_pose_ == 1.0:
            self.position_d_ = np.copy(self.position_d_target_)
            # Assuming orientation_d_ and orientation_d_target_ are quaternion objects with a method coeffs()
            self.orientation_d_.coeffs = np.copy(self.orientation_d_target_.coeffs)
        else:
            step = self.filter_step(self.update_frequency_, self.filter_params_pose_)
            self.position_d_ = self.filtered_update(self.position_d_target_, self.position_d_, step)
            # Slerp operation for quaternions; assuming orientation_d_ has a slerp method
            self.orientation_d_ = self.orientation_d_.slerp(step, self.orientation_d_target_)

    def update_filtered_wrench(self):
        if self.filter_params_wrench_ == 1.0:
            self.cartesian_wrench_ = np.copy(self.cartesian_wrench_target_)
        else:
            step = self.filter_step(self.update_frequency_, self.filter_params_wrench_)
            self.cartesian_wrench_ = self.filtered_update(self.cartesian_wrench_target_, self.cartesian_wrench_, step)
