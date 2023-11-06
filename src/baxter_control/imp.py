import rospy
import numpy as np
import tf.transformations
from baxter_interface.cfg import CartesianImpedanceGainServerConfig
import dynamic_reconfigure.client

class IMP(object):
    # New attributes for stiffness control
    def __init__(self, kin=None, dyn=None):
        self._kin = kin
        self._dyn = dyn

        self._cartesian_D = np.zeros(6)
        self._cartesian_K = np.zeros(6)
        self._cartesian_D_target = np.zeros(6)
        self._cartesian_K_target = np.zeros(6)

        self._stiff_force = np.zeros(6)

        self._cur_time = rospy.get_time()
        self._prev_time = self._cur_time

        self.trans_stf_min = 0.0
        self.trans_stf_max = 10.0
        self.rot_stf_min = 0.0
        self.rot_stf_max = 10.0

        self.trans_dmp_min = 0.0
        self.trans_dmp_max = 1.0
        self.rot_dmp_min = 0.0
        self.rot_dmp_max = 1.0

        self.update_frequency = 100

        self.filter_params = rospy.set_param("~filtering/param", 1.0)

        self.dyn_srv_wrench_param = dynamic_reconfigure.client.Client("position_imp", timeout=30, config_callback=self.dynamic_update)

        self.dynamic_update(self._dyn.config)

    def saturate_value(self, x, x_min, x_max):
        return np.clip(x, x_min, x_max)

    def set_K(self, stiffness_new):
        stiffness_new = np.array(stiffness_new, dtype=np.float16)
        assert all(v >= 0 for v in stiffness_new), "Stiffness values need to be positive."
        self._cartesian_K_target = stiffness_new

    def set_D(self, damping_new):
        damping_new = np.array(damping_new, dtype=np.float16)
        assert all(v >= 0 for v in damping_new), "Damping values need to be positive."
        self._cartesian_D_target = damping_new

    def filter_step(self, update_frequency, filter_param):
        kappa = - 1.0 / np.log(1 - min(filter_param, 0.999999))
        return 1.0 / (kappa * update_frequency + 1.0)

    def filtered_update(self, target, current, step):
        target = np.asarray(target)
        current = np.asarray(current)
        result = (1.0 - step) * current + step * target
        return result

    def update_filtered_gains(self):
        if self.filter_params == 1.0:
            self._cartesian_K = np.copy(self._cartesian_K_target)
            self._cartesian_D = np.copy(self._cartesian_D_target)
        else:
            # print("params: ",self.filter_params)
            step = self.filter_step(self.update_frequency, self.filter_params)
            # print("step: ", step)
            self._cartesian_K = self.filtered_update(self._cartesian_K_target, self._cartesian_K, step)
            self._cartesian_D = self.filtered_update(self._cartesian_D_target, self._cartesian_D, step)

    def dynamic_update(self, config):
        if config['stiffness']:
            tx = self.saturate_value(config['translation_x_s'], self.trans_stf_min, self.trans_stf_max)
            ty = self.saturate_value(config['translation_y_s'], self.trans_stf_min, self.trans_stf_max)
            tz = self.saturate_value(config['translation_z_s'], self.trans_stf_min, self.trans_stf_max)
            rx = self.saturate_value(config['rotation_x_s'], self.rot_stf_min, self.rot_stf_max)
            ry = self.saturate_value(config['rotation_y_s'], self.rot_stf_min, self.rot_stf_max)
            rz = self.saturate_value(config['rotation_z_s'], self.rot_stf_min, self.rot_stf_max)
            self.set_K([tx, ty, tz, rx, ry, rz])
        if config['damping_factors']:
            tx_d = self.saturate_value(config['translation_x_d'], self.trans_dmp_min, self.trans_dmp_max)
            ty_d = self.saturate_value(config['translation_y_d'], self.trans_dmp_min, self.trans_dmp_max)
            tz_d = self.saturate_value(config['translation_z_d'], self.trans_dmp_min, self.trans_dmp_max)
            rx_d = self.saturate_value(config['rotation_x_d'], self.rot_dmp_min, self.rot_dmp_max)
            ry_d = self.saturate_value(config['rotation_y_d'], self.rot_dmp_min, self.rot_dmp_max)
            rz_d = self.saturate_value(config['rotation_z_d'], self.rot_dmp_min, self.rot_dmp_max)
            self.set_D([tx_d, ty_d, tz_d, rx_d, ry_d, rz_d])
        self.filter_params = config['filtering_weight']

        # Method for stiffness calculation
    def _stiff_cal(self, force, joint_vel):
        endpoint_vel = np.asarray(self._kin.jacobian().dot(np.asarray(list(joint_vel.values())).reshape(7,-1))).ravel()
        # print("K", self._cartesian_K)
        # print("D", self._cartesian_D)

        # print("endpoint_vel", endpoint_vel)
        # print("force", force)
        diff_pose = -1.0* (force + self._cartesian_D * endpoint_vel) * self._cartesian_K
        # print("diff pose", diff_pose)
        return diff_pose

    # Method for applying stiffness
    def compute_output(self, joint_names, pointp, joint_vel, force):
        self.update_filtered_gains()

        self._stiff_pose = self._stiff_cal(force, joint_vel)
        jacobian_pseudo_inv = np.linalg.pinv(self._kin.jacobian())

        self._cur_time = rospy.get_time()

        dt = self._cur_time - self._prev_time

        _pointp =np.asarray(pointp + jacobian_pseudo_inv.dot(self._stiff_pose) * dt).ravel()

        self._prev_time = self._cur_time

        return _pointp
