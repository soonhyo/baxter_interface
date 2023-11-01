#!/usr/bin/env python3
#input to target, and use d when calculating
import rospy
import tf2_ros
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from baxter_core_msgs.msg import JointCommand
from .cartesian_impedance import *

from baxter_pykdl import baxter_kinematics

import dynamic_reconfigure.client

import numpy as np

from pyquaternion import Quaternion

import copy

def quaternion_to_numpy(quaternion):
    return np.asarray([quaternion.w, quaternion.x, quaternion.y, quaternion.z])

def calculate_orientation_error(orientation_1_np, orientation_2_np):
    orientation_1_q = Quaternion(orientation_1_np)
    orientation_2_q = Quaternion(orientation_2_np)
    if np.dot(orientation_1_np, orientation_2_np) < 0.0:
        orientation_2_q = -1 * orientation_2_q
    error_quaternion = orientation_1_q * orientation_2_q.inverse
    error_quaternion_angle_axis = error_quaternion.angle * error_quaternion.axis
    return error_quaternion_angle_axis


def filter_step(update_frequency, filter_percentage):
    kappa = -1 / np.log(1 - min(filter_percentage, 0.999999))
    return 1.0 / (kappa * update_frequency + 1.0)

def saturate_value(x, x_min, x_max):
    print("x", x)
    print("x_min", x_min)
    print("x_max", x_max)
    return np.clip(x, x_min, x_max)


def saturate_torque_rate(tau_d_calculated, tau_d_saturated, delta_tau_max):
    print("tau_d_calculated", tau_d_calculated)
    print("tau_d_saturated", tau_d_saturated)
    for i in range(len(tau_d_calculated)):
        difference = tau_d_calculated[i] - tau_d_saturated[i]
        print("diff:",difference)
        tau_d_saturated[i] += saturate_value(difference, -delta_tau_max, delta_tau_max)
    return tau_d_saturated

def damping_rule(stiffness):
    return 2 * np.sqrt(stiffness)

def quaternion_multiply(quaternion1, quaternion0):
    return quaternion_to_numpy(Quaternion(quaternion1) * Quaternion(quaternion0))


class CartesianImpedanceController:
    def __init__(self, limb, kin ,update_frequency):
        self.limb = limb # baxter interface limb
        self.name = self.limb.name
        self.joint_names = self.limb.joint_names()
        self.update_frequency = update_frequency

        self.jacobian = None

        # Dynamic reconfigure parameters
        self.cartesian_stiffness = np.eye(6)
        self.cartesian_stiffness_target = np.eye(6)

        self.cartesian_damping = np.eye(6)
        self.cartesian_damping_target = np.eye(6)


        self.nullspace_stiffness_target = 50.0
        self.nullspace_damping_target = 1.0

        self.nullspace_stiffness = self.nullspace_stiffness_target
        self.nullspace_damping = self.nullspace_damping_target

        self.cartesian_wrench = np.zeros(6)
        self.cartesian_wrench_target = self.cartesian_wrench

        # robot state
        self.position_d = None # from trajectory point
        self.orientation_d = None # from trajectory point
        self.position_d_target = None # from filtering when retargeting
        self.orientation_d_target = None # from filtering when retargeting

        self.n_joints = len(self.joint_names)
        self.q = np.zeros(self.n_joints)# current positiions
        self.dq = np.zeros(self.n_joints)# current velocities
        self.jacobian = np.zeros((6, self.n_joints))
        self.q_d_nullspace = np.zeros(self.n_joints) # from trajectory point
        self.q_d_nullspace_target = self.q_d_nullspace # for filtering when retargeting

        self.tau_c = np.zeros(self.n_joints) # torque command

        self.cmd_time = rospy.Time.now()

        # Initialize min and max values for stiffness and damping
        self.trans_stf_min = 0.0
        self.trans_stf_max = 2000.0
        self.rot_stf_min = 0.0
        self.rot_stf_max = 200.0
        self.ns_stf_min = 0.0
        self.ns_stf_max = 50.0

        self.trans_dmp_min = 0.0
        self.trans_dmp_max = 1.0
        self.rot_dmp_min = 0.0
        self.rot_dmp_max = 1.0
        self.ns_dmp_min = 0.0
        self.ns_dmp_max = 1.0

        self.tau_c_min = -1.0
        self.tau_c_max = 1.0

        # self.set_stiffness([200, 200, 200, 20, 20, 20, 0], False)
        # self.set_damping_factors([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # baxter dynamics model
        self._kin = kin
        self.root_frame = self._kin.get_kdl_chain()[0]

        # ros tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.loginfo("Finished initialization.")

    # init current q and pose
    def init_desired_pose(self, position_d_target, orientation_d_target):
        self.set_reference_pose(position_d_target, orientation_d_target)
        self.position_d = position_d_target
        self.orientation_d = orientation_d_target

    def init_q_d_nullspace(self, q_d_nullspace_target):
        self.set_q_d_nullspace_target(q_d_nullspace_target)
        self.q_d_nullspace = q_d_nullspace_target

    def init_dynamic_reconfigure(self):
        self.dynamic_server_wrench_param = dynamic_reconfigure.client.Client("/CartesianImpedanceJointTrajectoryActionServer", timeout=30, config_callback=self.dynamic_cb)
        return True

    def dynamic_cb(self, config):
        if config['stiffness']:
            tx = saturate_value(config['translation_x_s'], self.trans_stf_min, self.trans_stf_max)
            ty = saturate_value(config['translation_y_s'], self.trans_stf_min, self.trans_stf_max)
            tz = saturate_value(config['translation_z_s'], self.trans_stf_min, self.trans_stf_max)
            rx = saturate_value(config['rotation_x_s'], self.rot_stf_min, self.rot_stf_max)
            ry = saturate_value(config['rotation_y_s'], self.rot_stf_min, self.rot_stf_max)
            rz = saturate_value(config['rotation_z_s'], self.rot_stf_min, self.rot_stf_max)
            ns_stiffness = saturate_value(config['nullspace_stiffness'], self.ns_stf_min, self.ns_stf_max)
            self.set_stiffness([tx, ty, tz, rx, ry, rz, ns_stiffness])

        if config['damping_factors']:
            tx_d = saturate_value(config['translation_x_d'], self.trans_dmp_min, self.trans_dmp_max)
            ty_d = saturate_value(config['translation_y_d'], self.trans_dmp_min, self.trans_dmp_max)
            tz_d = saturate_value(config['translation_z_d'], self.trans_dmp_min, self.trans_dmp_max)
            rx_d = saturate_value(config['rotation_x_d'], self.rot_dmp_min, self.rot_dmp_max)
            ry_d = saturate_value(config['rotation_y_d'], self.rot_dmp_min, self.rot_dmp_max)
            rz_d = saturate_value(config['rotation_z_d'], self.rot_dmp_min, self.rot_dmp_max)
            ns_damping = saturate_value(config['nullspace_damping'], self.ns_dmp_min, self.ns_dmp_max)
            self.set_damping_factors([tx_d, ty_d, tz_d, rx_d, ry_d, rz_d, ns_damping])

        F = np.zeros(6)
        if config['wrench']:
            F = np.array([config['f_x'], config['f_y'], config['f_z'], config['tau_x'], config['tau_y'], config['tau_z']])
            # if not self.transform_wrench(F, self.end_effector, self.root_frame):
            #     rospy.logerr("Could not transform wrench. Not applying it.")
            #     return

            self.apply_wrench(F)

    def init_and_update_rosparam(self):
        self.end_effector = rospy.get_param("~end_effector", self.name+"_gripper")
        self.delta_tau_max = rospy.get_param("~delta_tau_max", 1.0)
        self.filter_params_nullspace_config = rospy.get_param("~filtering/nullspace_config", 1.0)
        self.filter_params_stiffness = rospy.get_param("~filtering/stiffness", 1.0)
        self.filter_params_pose = rospy.get_param("~filtering/pose", 1.0)
        self.filter_params_wrench = rospy.get_param("~filtering/wrench", 1.0)
        self.verbose_print = rospy.get_param("~verbosity/verbose_print", False)
        self.verbose_state = rospy.get_param("~verbosity/state_msgs", False)
        self.verbose_tf = rospy.get_param("~verbosity/tf_frames", False)
        self.eef_ext = rospy.get_param("~end_effector/external_force", False)

    def init_messaging(self):
        if self.eef_ext == True:
            self.sub_cart_wrench_ = rospy.Subscriber("/robot/limb/"+self.name+"/end_effector/wrench", WrenchStamped, self.wrench_command_cb)
        # self.sub_controller_config_ = rospy.Subscriber("set_config", Float64MultiArray, self.controller_config_cb)
        # self.sub_reference_pose_ = rospy.Subscriber("reference_pose", PoseStamped, self.reference_pose_cb)
        return True

    def initialize(self, q):
        if self.init_dynamic_reconfigure() and self.init_and_update_rosparam():
            return False

        # Initialize ROS messaging features
        if not self.init_messaging():
            return False

        self.position_d_target, self.orientation_d_target = self.get_fk(q)
        self.q_d_nullspace_target = q

        self.init_desired_pose(self.position_d_target, self.orientation_d_target)
        self.init_q_d_nullspace(self.q_d_nullspace_target)

        rospy.loginfo("Started Cartesian Impedance Controller")

    def wrench_command_cb(self, msg):
        F = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

        # if msg.header.frame_id and msg.header.frame_id != self.end_effector:
        #     if not self.transform_wrench(F, msg.header.frame_id, self.end_effector):
        #         rospy.logerr("Could not transform wrench. Not applying it.")
        #         return

        self.apply_wrench(F)

    def transform_wrench(self, cartesian_wrench, from_frame, to_frame):
        try:
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time())
            trans = transform.transform.translation
            rot = transform.transform.rotation

            v_f = cartesian_wrench[:3]
            v_t = cartesian_wrench[3:]
            # print(v_f)
            # print(v_t)
            # Rotating the vectors
            quat_vec_f = np.array([0] + list(v_f))
            quat_vec_t = np.array([0] + list(v_t))
            quat_rot = np.array([rot.w, rot.x, rot.y, rot.z])

            v_f_rot = quaternion_multiply(quaternion_multiply(quat_rot, quat_vec_f), np.conj(quat_rot))[1:]
            v_t_rot = quaternion_multiply(quaternion_multiply(quat_rot, quat_vec_t), np.conj(quat_rot))[1:]
            # print("v_f_rot", v_f_rot)
            # print("v_t_rot", v_t_rot)

            cartesian_wrench[:] = np.concatenate([v_f_rot, v_t_rot])

            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logerr("Transform exception: %s", str(ex))
            return False

    def set_stiffness(self, stiffness_new, auto_damping=False):
        stiffness_new = np.array(stiffness_new, dtype=np.int16)
        assert all(v >= 0 for v in stiffness_new), "Stiffness values need to be positive."
        self.cartesian_stiffness_target = np.diag(stiffness_new[:6])
        self.nullspace_stiffness_target = stiffness_new[6]
        if auto_damping:
            self.apply_damping()

    def set_damping_factors(self, damping_new):
        damping_new = np.array(damping_new, dtype=np.int8)
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
        self.orientation_d_target = np.array(orientation_d_target)
        self.orientation_d_target /= np.linalg.norm(self.orientation_d_target)

    def set_q_d_nullspace_target(self, q_d_nullspace_target):
        assert len(q_d_nullspace_target) == self.n_joints, "Nullspace target needs to be the same size as n_joints_"
        self.q_d_nullspace_target = q_d_nullspace_target

    def set_max_torque_delta(self, d, update_frequency=None):
        assert d >= 0, "Allowed torque change must be positive"
        self.delta_tau_max = d
        if update_frequency is not None:
            self.set_update_frequency(update_frequency)

    def apply_wrench(self, cartesian_wrench_target):
        self.cartesian_wrench_target = np.array(cartesian_wrench_target)
        print("apply_wrench ", self.cartesian_wrench_target)

    def update(self, q, dq, q_d):
        #current
        self.q = q
        self.dq = dq
        self.position, self.orientation = self.get_fk(self.q)

        # jacobian_ = self.jacobian
        # duration = rospy.Time.now() - self.cmd_time

        self.jacobian = self._kin.jacobian(self.q)
        self.jacobian_transpose_pinv = np.linalg.pinv(self.jacobian.T)
        # self.djacobian = (self.jacobian - jacobian_) / 0.01
        # print(self.djacobian)
        # print(self.djacobian)
        # self.cart_inertia = self._kin.cart_inertia(self.q)

        #target
        self.q_d_nullspace = q_d
        # self.set_q_d_nullspace_target(self.q_d)

        # self.update_filtered_q_d_nullspace()

        self.position_d, self.orientation_d =  self.get_fk(self.q_d_nullspace)
        # self.position_d_target, self.orientation_d_target =  self.get_fk(self.q_d)

        self.set_reference_pose(self.position_d_target, self.orientation_d_target)

        self.update_filtered_stiffness()
        # self.update_filtered_pose()
        # self.update_filtered_wrench()

        # self.init_and_update_rosparam()

    def compute_output(self):
        # Perform a filtering step

        # current state
        cmd = dict()

        joint_name = self.q_d_nullspace.keys()
        q_d = np.asarray(list(self.q_d_nullspace.values()))
        q = np.asarray(list(self.q.values()))
        dq = np.asarray(list(self.dq.values()))


        # Compute error term
        error = np.zeros(6)
        error[:3] = self.position - self.position_d
        error[3:] = calculate_orientation_error(self.orientation, self.orientation_d)
        error = error[np.newaxis]
        # Kinematic pseudo-inverse

        # Initialize torque vectors
        tau_task = np.zeros(self.n_joints)
        tau_nullspace = np.zeros(self.n_joints)
        # tau_ext = np.zeros(self.n_joints)
        # Calculate task torque
        # extra = self.cart_inertia.dot(self.djacobian.dot(dq.T).T)
        # print(extra)
        # tau_task = self.jacobian.T.dot(-self.cartesian_stiffness.dot(error.T) - self.cartesian_damping.dot(self.jacobian.dot(dq.T).T)
        #                                -extra).T
        tau_task = self.jacobian.T.dot(-self.cartesian_stiffness.dot(error.T) - self.cartesian_damping.dot(self.jacobian.dot(dq.T).T)).T

        # Calculate nullspace torque
        nullspace_projector = np.eye(self.n_joints) - self.jacobian.T.dot(self.jacobian_transpose_pinv)
        tau_nullspace = nullspace_projector.dot(self.nullspace_stiffness * (q_d -q) - self.nullspace_damping * dq)

        # # Calculate external torque
        tau_ext = self.jacobian.T.dot(self.cartesian_wrench_target)
        tau_c = np.asarray(tau_task + tau_nullspace + tau_ext).squeeze()
        # tau_d = np.asarray(tau_task + tau_nullspace).squeeze()
        # tau_d = np.asarray(tau_task).squeeze()

        #tau_d = tau_task
        # self.tau_c = saturate_torque_rate(tau_d, self.tau_c, self.delta_tau_max)

        for idx, joint in enumerate(self.joint_names):
            # spring portion
            # cmd[joint] = np.clip(tau_c[idx], self.tau_c_min, self.tau_c_max)
            cmd[joint] = tau_c[idx]
        #rospy.loginfo(cmd)
        return cmd

    def get_fk(self, q):
        ee_state = self._kin.forward_position_kinematics(q)
        position= np.asarray(ee_state[:3])
        orientation = np.asarray([ee_state[-1], ee_state[3], ee_state[4], ee_state[5]])
        return position, orientation

    # def update_state(self, q, dq, tau_c):
    #     self.q = np.asarray(q)
    #     self.dq = np.asarray(dq)
    #     self.tau_c = np.asarray(tau_c)

    #     q_ = dict()

    #     for idx, jnt in enumerate(self.joint_names):
    #         q_[jnt] = q[idx]

    #     self.jacobian = self._kin.jacobian(q_)
    #     self.position, self.orientation = self.get_fk(self.q)

    def get_last_commands(self):
        return np.copy(self.tau_c)

    def get_applied_wrench(self):
        return np.copy(self.cartesian_wrench)

    def set_update_frequency(self, freq):
        assert freq >= 0, "Update frequency needs to be greater or equal to zero"
        self.update_frequency = max(freq, 0.0)


    def set_filtering(self, update_frequency, filter_params_nullspace_config, filter_params_stiffness,
                     filter_params_pose, filter_params_wrench):
        self.set_update_frequency(update_frequency)
        self.set_filter_value(filter_params_nullspace_config, self.filter_params_nullspace_config)
        self.set_filter_value(filter_params_stiffness, self.filter_params_stiffness)
        self.set_filter_value(filter_params_pose, self.filter_params_pose)
        self.set_filter_value(filter_params_wrench, self.filter_params_wrench)

    def set_filter_value(self, val, saved_val):
        assert 0 < val <= 1.0, "Filter params need to be between 0 and 1."
        saved_val = saturate_value(val, 0.0000001, 1.0)

    def filter_step(self, update_frequency, filter_param):
        return 1.0 - np.exp(-update_frequency / filter_param)


    def filtered_update(self, target, current, step):
        if type(target) == type(dict()):
            target_ = np.asarray(list(target.values()))
            current_ = np.asarray(list(current.values()))
            result = dict(zip(target.keys() ,((1.0 - step) * current_ + step * target_)))
        else:
            target = np.asarray(target)
            current = np.asarray(current)
            result = (1.0 - step) * current + step * target
        return result

    def update_filtered_q_d_nullspace(self):
        if self.filter_params_nullspace_config == 1.0:
            self.q_d_nullspace = copy.copy(self.q_d_nullspace_target)
        else:
            step = self.filter_step(self.update_frequency, self.filter_params_nullspace_config)
            self.q_d_nullspace = self.filtered_update(self.q_d_nullspace_target, self.q_d_nullspace, step)

    def update_filtered_stiffness(self):
        if self.filter_params_stiffness == 1.0:
            self.cartesian_stiffness = np.copy(self.cartesian_stiffness_target)
            self.cartesian_damping = np.copy(self.cartesian_damping_target)
            self.nullspace_stiffness = self.nullspace_stiffness_target
        else:
            step = self.filter_step(self.update_frequency, self.filter_params_stiffness)
            self.cartesian_stiffness = self.filtered_update(self.cartesian_stiffness_target, self.cartesian_stiffness, step)
            self.cartesian_damping = self.filtered_update(self.cartesian_damping_target, self.cartesian_damping, step)
            self.nullspace_stiffness = self.filtered_update(self.nullspace_stiffness_target, self.nullspace_stiffness, step)

    def update_filtered_pose(self):
        if self.filter_params_pose == 1.0:
            self.position_d = np.copy(self.position_d_target)

            self.orientation_d = np.copy(self.orientation_d_target)
        else:
            step = self.filter_step(self.update_frequency, self.filter_params_pose)
            self.position_d = self.filtered_update(self.position_d_target, self.position_d, step)
            self.orientation_d = Quaternion(self.orientation_d)
            self.orientation_d_target = Quaternion(self.orientation_d_target)
            self.orientation_d = quaternion_to_numpy(Quaternion.slerp(self.orientation_d, self.orientation_d_target, step))

    def update_filtered_wrench(self):
        if self.filter_params_wrench == 1.0:
            self.cartesian_wrench = np.copy(self.cartesian_wrench_target)
        else:
            step = self.filter_step(self.update_frequency, self.filter_params_wrench)
            self.cartesian_wrench = self.filtered_update(self.cartesian_wrench_target, self.cartesian_wrench, step)

    def publish_msgs_and_tf(self):
        position = self.position  # Your position data here
        orientation = self.orientation  # Your orientation data here

        if self.verbose_print:
            rospy.loginfo("Cartesian Position: "+position+"\n"+
                          "Cartesian Orientation: "+orientation+"\n"+
                          "Cartesian Stiffness: "+self.cartesian_stiffness+"\n"+
                          "Cartesian Damping: "+self.cartesian_damping+"\n"+
                          "Nullspace Stiffness: "+self.nullspace_stiffness+"\n"+
                          "Cartesian Nullspace q d: "+self.q_d_nullspace+"\n"+
                          "Cartesian tau_d: "+ self.tau_c)  # Your detailed log message here

        if self.verbose_tf:
            self.publish_tf(self.position, self.orientation, self.root_frame, self.end_effector+"_ee_fk")
            self.publish_tf(self.position_d, self.orientation_d, self.root_frame, self.end_effector+"_ee_ref_pose")

    def publish_tf(self, position, orientation, from_frame, to_frame):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = from_frame
        transform.child_frame_id = to_frame
        transform.transform.translation = position  # Use a geometry_msgs/Vector3
        transform.transform.rotation = orientation  # Use a geometry_msgs/Quaternion
        self.tf_broadcaster.sendTransform(transform)
