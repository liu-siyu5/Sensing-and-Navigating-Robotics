#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def publisher():
    rospy.init_node('publisher', anonymous=True)
    pub = rospy.Publisher('chatter', String, queue_size=20)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        msg_str = "pooooooooo~"
        rospy.loginfo(msg_str)
        pub.publish(msg_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
