#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import serial
import math
import numpy as np
from std_msgs.msg import Header
from vn_driver.msg import Vectornav  # 根据实际消息类型修改

# 初始化串口
def init_serial(port, baudrate):
    try:
        return serial.Serial(port, baudrate, timeout=1)
    except serial.SerialException as e:
        rospy.logerr(f"Can't open：{e}")
        return None

# 读取并验证串口数据
def read_and_validate(serial_conn):
    try:
        data = serial_conn.readline().decode('utf-8').strip()
        if data.startswith("$VNYMR") and len(data.split(',')) == 13:
            return data.split(',')
        rospy.logwarn("Wrong")
    except Exception as e:
        rospy.logwarn(f"Can't read：{e}")
    return None

# 欧拉角转四元数
def euler_to_quaternion(yaw, pitch, roll):
    roll, pitch, yaw = map(math.radians, [roll, pitch, yaw])
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]

# 填充IMU消息
def create_Vectornav_msg(data_split, header):
    msg = Vectornav()
    msg.Header = msg.IMU.header = msg.MagField.header = header

    # 处理四元数、磁场、加速度、角速度数据
    msg.IMU.orientation.x, msg.IMU.orientation.y, msg.IMU.orientation.z, msg.IMU.orientation.w = euler_to_quaternion(*map(float, data_split[1:4]))
    msg.MagField.magnetic_field.x, msg.MagField.magnetic_field.y, msg.MagField.magnetic_field.z = (float(data_split[i]) * 1e-4 for i in range(4, 7))
    msg.IMU.linear_acceleration.x, msg.IMU.linear_acceleration.y, msg.IMU.linear_acceleration.z = map(float, data_split[7:10])
    msg.IMU.angular_velocity.x = float(data_split[10].split('*')[0])
    msg.IMU.angular_velocity.y = float(data_split[11].split('*')[0])
    msg.IMU.angular_velocity.z = float(data_split[12].split('*')[0])


    msg.rawIMUstring = ','.join(data_split)
    return msg

# 主函数
if __name__ == '__main__':
    rospy.init_node('vn_driver', anonymous=True)
    serial_conn = init_serial(rospy.get_param("~port", "/dev/ttyUSB0"), rospy.get_param("~baudrate", 115200))

    if not serial_conn:
        rospy.logerr("Exiting...")
        exit(1)

    pub = rospy.Publisher("imu", Vectornav, queue_size=10)
    rate = rospy.Rate(40)

    try:
        while not rospy.is_shutdown():
            data_split = read_and_validate(serial_conn)
            if not data_split:
                continue

            header = Header(frame_id="imu1_frame", stamp=rospy.Time.now())
            pub.publish(create_Vectornav_msg(data_split, header))
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        if serial_conn:
            serial_conn.close()
            rospy.loginfo("Closed")