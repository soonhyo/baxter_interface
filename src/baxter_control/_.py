#!/usr/bin/env python3

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

class CartesianImpedanceControllerRos(CartesianImpedanceController):
    def __init__(self, limb):
        self.limb = limb # baxter interface limb
        self.name = self.limb.name
        self.joint_names = self.limb.joint_names()

        self.jacobian = None

        # Dynamic reconfigure parameters
        self.cartesian_stiffness = np.eye(6)
        self.cartesian_stiffness_target = np.eye(6)

        self.cartesian_damping = np.eye(6)
        self.cartesian_damping_target = np.eye(6)

        self.nullspace_stiffness_target = 0.0
        self.nullspace_damping_target = 0.0

        # robot state
        self.position_d = None # from trajectory point
        self.orientation_d = None # from trajectory point

        self.q_d_nullspace = None
        self.n_joints = None
        self.q = None
        self.dq = None

        self.n_joints = len(self.joint_names)
        self.tau_c = np.zeros(self.n_joints)

        # ros parameters

        # self.set_stiffness([200, 200, 200, 20, 20, 20, 0], False)
        # self.set_damping_factors([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        if self.init_dynamic_reconfigure() and self.init_and_update_rosparam():
            return False

        # Initialize ROS messaging features
        if not self.init_messaging():
            return False

        # baxter dynamic model
        self._kin = baxter_kinematics(self.name)
        self.root_frame = self._kin.get_kdl_chain()[0]

        # ros tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize min and max values for stiffness and damping
        self.trans_stf_min = 0.0
        self.trans_stf_max = 2000.0
        self.dmp_factor_min = 0.0
        self.dmp_factor_max = 1.0

        self.starting()

        rospy.loginfo("Finished initialization.")

    def init_dynamic_reconfigure(self):
        self.dynamic_server_wrench_param = dynamic_reconfigure.client.Client("/CartesianImpedanceJointTrajectoryActionServer", timeout=30, config_callback=self.dynamic_cb)
        return True

    def dynamic_cb(self, config):
        if config['stiffness']:
            tx = saturate_value(config['translation_x'], self.trans_stf_min, self.trans_stf_max)
            ty = saturate_value(config['translation_y'], self.trans_stf_min, self.trans_stf_max)
            tz = saturate_value(config['translation_z'], self.trans_stf_min, self.trans_stf_max)
            rx = saturate_value(config['rotation_x'], self.trans_stf_min, self.trans_stf_max)
            ry = saturate_value(config['rotation_y'], self.trans_stf_min, self.trans_stf_max)
            rz = saturate_value(config['rotation_z'], self.trans_stf_min, self.trans_stf_max)
            ns_stiffness = config['nullspace_stiffness']
            self.set_stiffness([tx, ty, tz, rx, ry, rz, ns_stiffness])

        if config['damping_factors']:
            self.set_damping_factors(
                [config['translation_x'], config['translation_y'], config['translation_z'],
                config['rotation_x'], config['rotation_y'], config['rotation_z'],
                config['nullspace_damping']]
            )

        F = np.zeros(6)
        if config['wrench']:
            F = np.array([config['f_x'], config['f_y'], config['f_z'], config['tau_x'], config['tau_y'], config['tau_z']])
            if not self.transform_wrench(F, self.end_effector_, self.root_frame):
                rospy.logerr("Could not transform wrench. Not applying it.")
                return
            self.apply_wrench(F)

    def init_and_update_rosparam(self):
        self.end_effector_ = rospy.get_param("~end_effector", self.name+"_gripper")
        self.delta_tau_max_ = rospy.get_param("~delta_tau_max", 1.0)
        self.filter_params_nullspace_config_ = rospy.get_param("~filtering/nullspace_config", 0.1)
        self.filter_params_stiffness_ = rospy.get_param("~filtering/stiffness", 0.1)
        self.filter_params_pose_ = rospy.get_param("~filtering/pose", 0.1)
        self.filter_params_wrench_ = rospy.get_param("~filtering/wrench", 0.1)
        self.verbose_print_ = rospy.get_param("~verbosity/verbose_print", False)
        self.verbose_state_ = rospy.get_param("~verbosity/state_msgs", False)
        self.verbose_tf_ = rospy.get_param("~verbosity/tf_frames", False)
        self.eef_ext = rospy.get_param("~end_effector/external_force", False)

    def init_messaging(self):
        if self.eef_ext == True:
            self.sub_cart_wrench_ = rospy.Subscriber("/robot/limb/"+self.name+"/end_effector/wrench", WrenchStamped, self.wrench_command_cb)
        # self.sub_controller_config_ = rospy.Subscriber("set_config", Float64MultiArray, self.controller_config_cb)
        # self.sub_reference_pose_ = rospy.Subscriber("reference_pose", PoseStamped, self.reference_pose_cb)
        return True

    def starting(self):
        self.init_desired_pose(self.position, self.orientation)
        self.init_nullspace_config(self.q_d_nullspace)
        rospy.loginfo("Started Cartesian Impedance Controller")

    def get_fk(self, q):
        ee_state = self._kin.forward_position_kinematics(q)
        return ee_state[:3], ee_state[3:]

    def update_state(self, q, dq, tau_c):
        self.q = q
        self.dq = dq
        self.tau_c = tau_c

        self.jacobian = self._kin.jacobian()
        self.position, self.orientation = self.get_fk(self.q)

    # def controller_config_cb(self, msg):
    #     self.set_stiffness(msg.cartesian_stiffness, msg.nullspace_stiffness, False)
    #     self.set_damping_factors(msg.cartesian_damping_factors, msg.nullspace_damping_factor)

    #     if len(msg.q_d_nullspace) == self.n_joints:
    #         q_d_nullspace = np.array(msg.q_d_nullspace)
    #         self.set_nullspace_config(q_d_nullspace)
    #     else:
    #         rospy.logwarn(f"Nullspace configuration does not have the correct amount of entries. Got {len(msg.q_d_nullspace)} expected {self.n_joints}. Ignoring.")

    # def reference_pose_cb(self, msg):
    #     if msg.header.frame_id and msg.header.frame_id != self.root_frame:
    #         rospy.logwarn(f"Reference poses need to be in the root frame '{self.root_frame}'. Ignoring.")
    #         return

    #     position_d = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    #     last_orientation_d_target = self.orientation_d

    #     orientation_d = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    #     if np.dot(last_orientation_d_target, self.orientation_d) < 0.0:
    #         self.orientation_d = -self.orientation_d

    #     self.set_reference_pose(position_d, orientation_d)

    def wrench_command_cb(self, msg):
        F = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

        if msg.header.frame_id and msg.header.frame_id != self.end_effector_:
            if not self.transform_wrench(F, msg.header.frame_id, self.end_effector_):
                rospy.logerr("Could not transform wrench. Not applying it.")
                return

        self.apply_wrench(F)

    def transform_wrench(self, cartesian_wrench, from_frame, to_frame):
        try:
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time())
            trans = transform.transform.translation
            rot = transform.transform.rotation

            v_f = cartesian_wrench[:3]
            v_t = cartesian_wrench[3:]

            # Rotating the vectors
            quat_vec_f = np.array([0] + list(v_f))
            quat_vec_t = np.array([0] + list(v_t))
            quat_rot = np.array([rot.w, rot.x, rot.y, rot.z])

            v_f_rot = quaternion_multiply(quaternion_multiply(quat_rot, quat_vec_f), np.conj(quat_rot))[1:]
            v_t_rot = quaternion_multiply(quaternion_multiply(quat_rot, quat_vec_t), np.conj(quat_rot))[1:]

            cartesian_wrench[:] = np.concatenate([v_f_rot, v_t_rot])

            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logerr("Transform exception: %s", str(ex))
            return False

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
            self.publish_tf(self.position, self.orientation, self.root_frame, f"{self.end_effector_}_ee_fk")
            self.publish_tf(self.position_d, self.orientation_d, self.root_frame, f"{self.end_effector}_ee_ref_pose")

    def publish_tf(self, position, orientation, from_frame, to_frame):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = from_frame
        transform.child_frame_id = to_frame
        transform.transform.translation = position  # Use a geometry_msgs/Vector3
        transform.transform.rotation = orientation  # Use a geometry_msgs/Quaternion
        self.tf_broadcaster.sendTransform(transform)


if __name__ == '__main__':
    main()
