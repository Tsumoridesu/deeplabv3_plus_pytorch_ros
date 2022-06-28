#!/usr/bin/env python3
# codding = utf-8

import rospy
from geometry_msgs.msg import Twist
from deeplabv3_plus_pytorch_ros.msg import zone_status


class segmentation_cmd_vel:
    def __init__(self):
        self.max_linear = rospy.get_param('~max_linear', 1)  #
        self.max_angular = rospy.get_param('~max_angular', 2)  #

        self.danger_zone_min = rospy.get_param('~danger_zone_min', 0.95)  #
        # self.left_zone_threshold = rospy.get_param('~left_zone_threshold', 0)  #
        # self.right_zone_threshold = rospy.get_param('~right_zone_threshold', 0)  #
        # self.warning_zone_threshold = rospy.get_param('~warning_zone_threshold', 0)  #

        self.linear_gain = rospy.get_param('~linear_gain', 1)  #
        self.angular_gain = rospy.get_param('~angular_gain', 8)  #

        self.change_mode = rospy.get_param('~change_mode', 'default')

        self.zone_status_sub = rospy.get_param('~zone_status_sub', '/Segmentation_image/zone_status')
        self.zone_status = rospy.Subscriber(self.zone_status_sub, zone_status, self.state2cmd)

        self.cmd_vel_pub_topic = rospy.get_param('~cmd_vel_pub', 'icart_mini/cmd_vel')
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_pub_topic, Twist, queue_size=10)

    def state2cmd(self, data):
        msg = Twist()
        # print(data)
        if self.change_mode == "default":
            if data.danger_zone_point > self.danger_zone_min:
                msg.linear.x = data.warning_zone_point * self.linear_gain * self.max_linear
            else:
                msg.linear.x = -0.1

            for i in range(10):
                turn_tootle = 0
                turn = data.left_zone_point - data.right_zone_point
                turn_tootle += turn
            msg.angular.z = turn_tootle * self.angular_gain * self.max_angular


            self.cmd_vel_pub.publish(msg)

        else:
            rospy.loginfo('input your change mode in script')


if __name__ == '__main__':
    try:
        rospy.init_node("segmentation_cmd_vel")
        segmentation_cmd_vel()
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
