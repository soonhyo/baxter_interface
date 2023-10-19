#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from baxter_core_msgs.msg import JointCommand
from CartesianImpedanceController import *

class CartesianImpedanceControllerRos:
    def __init__(self):
        self.end_effector_ = rospy.get_param("~end_effector", "right_gripper")
        self.wrench_ee_frame_ = rospy.get_param("~wrench_ee_frame", self.end_effector_)
        self.dynamic_reconfigure = rospy.get_param("~dynamic_reconfigure", True)
        self.enable_trajectories = rospy.get_param("~handle_trajectories", True)
        self.delta_tau_max_ = rospy.get_param("~delta_tau_max", 1.0)
        self.update_frequency_ = rospy.get_param("~update_frequency", 500.0)
        self.filter_params_nullspace_config_ = rospy.get_param("~filtering/nullspace_config", 0.1)
        self.filter_params_stiffness_ = rospy.get_param("~filtering/stiffness", 0.1)
        self.filter_params_pose_ = rospy.get_param("~filtering/pose", 0.1)
        self.filter_params_wrench_ = rospy.get_param("~filtering/wrench", 0.1)
        self.verbose_print_ = rospy.get_param("~verbosity/verbose_print", False)
        self.verbose_state_ = rospy.get_param("~verbosity/state_msgs", False)
        self.verbose_tf_ = rospy.get_param("~verbosity/tf_frames", False)

        self.pub_torques = rospy.Publisher("torques_topic", Float64MultiArray, queue_size=10)
        self.pub_command = rospy.Publisher("command_topic", Float64MultiArray, queue_size=10)
        self.pub_state = rospy.Publisher("state_topic", YourCustomStateMessage, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.root_frame = "root_frame"

        # Initialize min and max values for stiffness and damping
        self.trans_stf_min = 0.0
        self.trans_stf_max = 100.0

        # Assuming other initializations
        self.n_joints = 6
        self.tau_c = np.zeros(self.n_joints)

        # Initialize a tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize ROS messaging features
        if not self.init_messaging() or not self.init_rb_dyn():
            return False

        if self.enable_trajectories and not self.init_trajectories():
            return False

        self.root_frame_ = self.rbdyn_wrapper_.root_link()
        rospy.set_param("root_frame", self.root_frame_)

        if self.n_joints_ < 6:
            rospy.logwarn("Number of joints is below 6. Functions might be limited.")

        if self.n_joints_ < 7:
            rospy.logwarn("Number of joints is below 7. No redundant joint for nullspace.")

        self.tau_m_ = [0.0] * self.n_joints_

        if self.dynamic_reconfigure and not self.init_dynamic_reconfigure():
            return False

        self.traj_as = actionlib.SimpleActionServer('trajectory_action', FollowJointTrajectoryAction, execute_cb=self.traj_goal_cb, auto_start=False)
        self.traj_as.start()

        self.traj_sub = rospy.Subscriber('trajectory_topic', JointTrajectory, self.traj_cb)

        self.traj_running = False
        self.traj_index = 0
        self.traj_duration = rospy.Duration()

        rospy.loginfo("Finished initialization.")

    def init_dynamic_reconfigure(self):
        self.dynamic_server_compliance_param_ = dynamic_reconfigure.client.Client(nh.get_namespace() + "/stiffness_reconfigure", timeout=30)
        self.dynamic_server_compliance_param_.set_callback(self.dynamic_stiffness_cb)

        self.dynamic_server_damping_param_ = dynamic_reconfigure.client.Client(nh.get_namespace() + "/damping_factors_reconfigure", timeout=30)
        self.dynamic_server_damping_param_.set_callback(self.dynamic_damping_cb)

        self.dynamic_server_wrench_param_ = dynamic_reconfigure.client.Client(nh.get_namespace() + "/cartesian_wrench_reconfigure", timeout=30)
        self.dynamic_server_wrench_param_.set_callback(self.dynamic_wrench_cb)

        return True

    def init_messaging(self):
        self.sub_cart_stiffness_ = rospy.Subscriber("set_cartesian_stiffness", Float64MultiArray, self.cartesian_stiffness_cb)
        self.sub_cart_wrench_ = rospy.Subscriber("set_cartesian_wrench", Float64MultiArray, self.wrench_command_cb)
        self.sub_damping_factors_ = rospy.Subscriber("set_damping_factors", Float64MultiArray, self.cartesian_damping_factor_cb)
        self.sub_controller_config_ = rospy.Subscriber("set_config", Float64MultiArray, self.controller_config_cb)
        self.sub_reference_pose_ = rospy.Subscriber("reference_pose", Float64MultiArray, self.reference_pose_cb)

        self.sub_joint_state_ = rospy.Subscriber("/robot/joint_states", JointState, self.update_state)

        self.pub_torques_ = rospy.Publisher("commanded_torques", Float64MultiArray, queue_size=20)
        self.pub_state_ = rospy.Publisher("controller_state", Float64MultiArray, queue_size=10)  # Adjust message type
        self.pub_command_ = rospy.Publisher("/robot/limb/right/joint_command", JointCommand, queue_size=10)  # Adjust topic name based on conditions

        # Initialize other parameters, e.g., joint names and offsets
        if nh.has_param("joints"):
            self.joint_names = nh.get_param("joints")

        # ... (initialize pub_state_ and pub_command_ messages)

        return True

    def init_rb_dyn(self):
        # Get the URDF XML from the parameter server. Wait if needed.
        robot_description_param = "/robot_description"
        if nh.has_param("robot_description"):
            robot_description_param = nh.get_param("robot_description")

        while not nh.has_param(robot_description_param):
            rospy.loginfo_once("Waiting for robot description in parameter %s on the ROS param server.", robot_description_param)
            sleep(0.1)  # Sleep for 100 ms

        urdf_string = nh.get_param(robot_description_param)

        try:
            self.rbdyn_wrapper_.init_rb_dyn(urdf_string, self.end_effector_)
        except RuntimeError as e:
            rospy.logerr("Error when initializing RBDyn: %s", str(e))
            return False

        rospy.loginfo("Number of joints found in urdf: %d", self.rbdyn_wrapper_.n_joints())

        if self.rbdyn_wrapper_.n_joints() < self.n_joints_:
            rospy.logerr("Number of joints in the URDF is smaller than supplied number of joints. %d < %d", self.rbdyn_wrapper_.n_joints(), self.n_joints_)
            return False
        elif self.rbdyn_wrapper_.n_joints() > self.n_joints_:
            rospy.logwarn("Number of joints in the URDF is greater than supplied number of joints: %d > %d. Assuming that the actuated joints come first.", self.rbdyn_wrapper_.n_joints(), self.n_joints_)

        return True

   def init_trajectories(self, nh):
        # Initialize subscribers and action servers
        self.sub_trajectory = rospy.Subscriber("joint_trajectory", JointTrajectory, self.traj_cb)

        self.traj_as = actionlib.SimpleActionServer("follow_joint_trajectory", FollowJointTrajectoryAction, execute_cb=None, auto_start=False)
        self.traj_as.register_goal_callback(self.traj_goal_cb)
        self.traj_as.register_preempt_callback(self.traj_preempt_cb)
        self.traj_as.start()

        return True

     def starting(self, time):
        self.init_desired_pose(self.position_, self.orientation_)
        self.init_nullspace_config(self.q_)
        rospy.loginfo("Started Cartesian Impedance Controller")

    def update(self, time, period):
        if self.traj_running_:
            self.traj_update()

        # Apply control law in base library
        self.calculate_commanded_torques()
        self.publish_msgs_and_tf()

    def get_fk(self, q): # TODO
        # Assuming rbdyn_wrapper object is available and has a method called perform_fk
        if self.rbdyn_wrapper.n_joints != self.n_joints:
            q_rb = np.zeros(self.rbdyn_wrapper.n_joints)
            q_rb[:len(q)] = q
            ee_state = self.rbdyn_wrapper.perform_fk(q_rb)
        else:
            ee_state = self.rbdyn_wrapper.perform_fk(q)

        return ee_state.translation, ee_state.orientation

    def get_jacobian(self, q, dq):
        if self.rbdyn_wrapper.n_joints != self.n_joints:
            q_rb = np.zeros(self.rbdyn_wrapper.n_joints)
            q_rb[:len(q)] = q
            dq_rb = np.zeros(self.rbdyn_wrapper.n_joints)
            dq_rb[:len(dq)] = dq
            jacobian = self.rbdyn_wrapper.jacobian(q_rb, dq_rb)
        else:
            jacobian = self.rbdyn_wrapper.jacobian(q, dq)

        jacobian = np.dot(self.jacobian_perm, jacobian)

        return jacobian

    def update_state(self, msg):
        for i in range(self.n_joints):
            offset = self.joint_index_offset
            self.q[i] = msg.position[i + offset]
            self.dq[i] = msg.velocity[i + offset]
            self.tau_m[i] = msg.effort[i + offset]

        self.jacobian = self.get_jacobian(self.q, self.dq)
        self.position, self.orientation = self.get_fk(self.q)

    def controller_config_cb(self, msg):
        self.set_stiffness(msg.cartesian_stiffness, msg.nullspace_stiffness, False)
        self.set_damping_factors(msg.cartesian_damping_factors, msg.nullspace_damping_factor)

        if len(msg.q_d_nullspace) == self.n_joints:
            q_d_nullspace = np.array(msg.q_d_nullspace)
            self.set_nullspace_config(q_d_nullspace)
        else:
            rospy.logwarn(f"Nullspace configuration does not have the correct amount of entries. Got {len(msg.q_d_nullspace)} expected {self.n_joints}. Ignoring.")

    def cartesian_damping_factor_cb(self, msg):
        self.set_damping_factors(msg, self.damping_factors[6])

    def reference_pose_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.root_frame:
            rospy.logwarn(f"Reference poses need to be in the root frame '{self.root_frame}'. Ignoring.")
            return

        position_d = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        last_orientation_d_target = self.orientation_d

        orientation_d = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

        if np.dot(last_orientation_d_target, self.orientation_d) < 0.0:
            self.orientation_d = -self.orientation_d

        self.set_reference_pose(position_d, orientation_d)

    def cartesian_stiffness_cb(self, msg):
        self.set_stiffness(msg.wrench, self.nullspace_stiffness_target_)

    def set_damping_factors(self, cart_damping, nullspace):
        # Assuming that setDampingFactors is defined in a base class or this class
        self.setDampingFactors(
            self.saturate_value(cart_damping.force.x, self.dmp_factor_min_, self.dmp_factor_max_),
            self.saturate_value(cart_damping.force.y, self.dmp_factor_min_, self.dmp_factor_max_),
            self.saturate_value(cart_damping.force.z, self.dmp_factor_min_, self.dmp_factor_max_),
            self.saturate_value(cart_damping.torque.x, self.dmp_factor_min_, self.dmp_factor_max_),
            self.saturate_value(cart_damping.torque.y, self.dmp_factor_min_, self.dmp_factor_max_),
            self.saturate_value(cart_damping.torque.z, self.dmp_factor_min_, self.dmp_factor_max_),
            self.saturate_value(nullspace, self.dmp_factor_min_, self.dmp_factor_max_)
        )

    def set_stiffness(self, cart_stiffness, nullspace, auto_damping=False):
        # Assuming that setStiffness is defined in a base class or this class
        self.setStiffness(
            self.saturate_value(cart_stiffness.force.x, self.trans_stf_min_, self.trans_stf_max_),
            self.saturate_value(cart_stiffness.force.y, self.trans_stf_min_, self.trans_stf_max_),
            self.saturate_value(cart_stiffness.force.z, self.trans_stf_min_, self.trans_stf_max_),
            self.saturate_value(cart_stiffness.torque.x, self.rot_stf_min_, self.rot_stf_max_),
            self.saturate_value(cart_stiffness.torque.y, self.rot_stf_min_, self.rot_stf_max_),
            self.saturate_value(cart_stiffness.torque.z, self.rot_stf_min_, self.rot_stf_max_),
            self.saturate_value(nullspace, self.ns_min_, self.ns_max_),
            auto_damping
        )

    def wrench_command_cb(self, msg):
        F = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

        if msg.header.frame_id and msg.header.frame_id != self.root_frame_:
            if not self.transform_wrench(F, msg.header.frame_id, self.root_frame_):
                rospy.logerr("Could not transform wrench. Not applying it.")
                return
        elif not msg.header.frame_id:
            if not self.transform_wrench(F, self.wrench_ee_frame_, self.root_frame_):
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
        tau_c_msg = Float64MultiArray()
        tau_c_msg.data = self.tau_c
        self.pub_torques.publish(tau_c_msg)
        self.pub_command.publish(tau_c_msg)

        error = self.get_pose_error()
        position = self.position  # Your position data here
        orientation = self.orientation  # Your orientation data here

        if self.verbose_print:
            rospy.loginfo("Your log message")  # Your detailed log message here

        if self.verbose_tf:
            self.publish_tf(self.position, self.orientation, self.root_frame, f"{self.end_effector}_ee_fk")
            self.publish_tf(self.position_d, self.orientation_d, self.root_frame, f"{self.end_effector}_ee_ref_pose")

        if self.verbose_state:
            state_msg = YourCustomStateMessage()  # Initialize your custom state message
            state_msg.header.stamp = rospy.Time.now()
            # Fill in the rest of your message
            self.pub_state.publish(state_msg)

    def publish_tf(self, position, orientation, from_frame, to_frame):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = from_frame
        transform.child_frame_id = to_frame
        transform.transform.translation = position  # Use a geometry_msgs/Vector3
        transform.transform.rotation = orientation  # Use a geometry_msgs/Quaternion
        self.tf_broadcaster.sendTransform(transform)

    def get_pose_error(self):
        return np.zeros(6)  # Dummy, replace with your logic

    def dynamic_stiffness_cb(self, config):
        if config['update_stiffness']:
            tx = saturate_value(config['translation_x'], self.trans_stf_min, self.trans_stf_max)
            ty = saturate_value(config['translation_y'], self.trans_stf_min, self.trans_stf_max)
            tz = saturate_value(config['translation_z'], self.trans_stf_min, self.trans_stf_max)
            rx = saturate_value(config['rotation_x'], self.trans_stf_min, self.trans_stf_max)
            ry = saturate_value(config['rotation_y'], self.trans_stf_min, self.trans_stf_max)
            rz = saturate_value(config['rotation_z'], self.trans_stf_min, self.trans_stf_max)
            ns_stiffness = config['nullspace_stiffness']
            self.set_stiffness(tx, ty, tz, rx, ry, rz, ns_stiffness)

    def dynamic_damping_cb(self, config):
        if config['update_damping_factors']:
            self.set_damping_factors(
                config['translation_x'], config['translation_y'], config['translation_z'],
                config['rotation_x'], config['rotation_y'], config['rotation_z'],
                config['nullspace_damping']
            )

    def dynamic_wrench_cb(self, config):
        F = np.zeros(6)
        if config['apply_wrench']:
            F = np.array([config['f_x'], config['f_y'], config['f_z'], config['tau_x'], config['tau_y'], config['tau_z']])
            if not self.transform_wrench(F, self.wrench_ee_frame, self.root_frame):
                rospy.logerr("Could not transform wrench. Not applying it.")
                return
            self.apply_wrench(F)

    def traj_cb(self, msg):
        rospy.loginfo('Got trajectory msg from trajectory topic.')

        if self.traj_as.is_active():
            self.traj_as.set_preempted()
            rospy.loginfo('Preempted running action server goal.')

        self.traj_start(msg)

    def traj_goal_cb(self, goal):
        self.traj_as_goal = goal
        rospy.loginfo('Accepted new goal from action server.')

        self.traj_start(goal.trajectory)

    def traj_preempt_cb(self):
        rospy.loginfo('Actionserver got preempted.')
        self.traj_as.set_preempted()

    def traj_start(self, trajectory):
        # Your initialization logic
        self.traj_duration = trajectory.points[-1].time_from_start
        self.traj_index = 0
        self.traj_running = True
        self.traj_start = rospy.Time.now()

        # Call the update function, perhaps in a timer or separate thread
        self.traj_update()

    def traj_update(self):
        if not self.traj_running:
            return

        current_time = rospy.Time.now()

        if current_time > (self.traj_start + self.traj_duration):
            rospy.loginfo('Finished executing trajectory.')
            if self.traj_as.is_active():
                self.traj_as.set_succeeded()
            self.traj_running = False
            return
def main():
  rospy.init_node('cartesian_impedance_controller')

  controller = CartesianImpedanceControllerRos()
  if not controller.init(rospy.get_node_handle()):
      rospy.logerr("Failed to initialize Cartesian impedance controller")
      return

  rate = rospy.Rate(500)  # Set the control loop rate (Hz)
  while not rospy.is_shutdown():
      try:
          controller.update(rospy.Time.now(), rospy.Duration(1.0 / 500))
      except rospy.ROSInterruptException:
          break
      rate.sleep()

if __name__ == '__main__':
    main()
