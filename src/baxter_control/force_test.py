#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
import math

def wrench_publisher():
    # 노드 초기화
    rospy.init_node('wrench_publisher_node')

    # Publisher 설정
    pub = rospy.Publisher('/comb/filtered_wrench', WrenchStamped, queue_size=10)

    # 발행할 메시지의 빈 틀 설정
    wrench_stamped = WrenchStamped()
    wrench_stamped.header.frame_id = ''
    wrench_stamped.wrench.force.x = 0.0
    wrench_stamped.wrench.force.z = 0.0
    wrench_stamped.wrench.torque.x = 0.0
    wrench_stamped.wrench.torque.y = 0.0
    wrench_stamped.wrench.torque.z = 0.0

    # 발행 주기 설정 (50Hz)
    rate = rospy.Rate(100)

    gain = 100
    # 시작 시간
    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        # 현재 시간
        current_time = rospy.get_time()

        # y 방향 힘을 사인 함수로 변화
        wrench_stamped.wrench.force.y = gain * math.sin(2 * math.pi * (current_time - start_time))

        # 현재 시각으로 header.stamp 업데이트
        wrench_stamped.header.stamp = rospy.Time.now()

        # 메시지 발행
        pub.publish(wrench_stamped)

        # 주기적으로 대기
        rate.sleep()

if __name__ == '__main__':
    try:
        wrench_publisher()
    except rospy.ROSInterruptException:
        pass
