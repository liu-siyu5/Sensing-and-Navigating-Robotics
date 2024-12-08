import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz

import rosbag
import numpy as np

def read_imu_data(bag_file):
    # 初始化存储数据的字典
    imu_data = {
        'time': [],
        'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
        'accel_x': [], 'accel_y': [], 'accel_z': [],
        'quat_w': [], 'quat_x': [], 'quat_y': [], 'quat_z': [],
        'magnetic_x': [], 'magnetic_y': [], 'magnetic_z': []
    }
    
    # 打开 ROS bag 文件
    bag = rosbag.Bag(bag_file)
    try:
        for topic, msg, t in bag.read_messages(topics=['/imu']):
            # 时间戳
            imu_data['time'].append(t.to_sec())
            
            # 陀螺仪数据（角速度，单位：度/秒）
            imu_data['gyro_x'].append(np.degrees(msg.imu.angular_velocity.x))
            imu_data['gyro_y'].append(np.degrees(msg.imu.angular_velocity.y))
            imu_data['gyro_z'].append(np.degrees(msg.imu.angular_velocity.z))
            
            # 加速度计数据
            imu_data['accel_x'].append(msg.imu.linear_acceleration.x)
            imu_data['accel_y'].append(msg.imu.linear_acceleration.y)
            imu_data['accel_z'].append(msg.imu.linear_acceleration.z)
            
            # 姿态四元数
            imu_data['quat_w'].append(msg.imu.orientation.w)
            imu_data['quat_x'].append(msg.imu.orientation.x)
            imu_data['quat_y'].append(msg.imu.orientation.y)
            imu_data['quat_z'].append(msg.imu.orientation.z)
            
            # 磁力计数据
            imu_data['magnetic_x'].append(msg.mag_field.magnetic_field.x)
            imu_data['magnetic_y'].append(msg.mag_field.magnetic_field.y)
            imu_data['magnetic_z'].append(msg.mag_field.magnetic_field.z)
    finally:
        # 确保文件关闭
        bag.close()
    
    # 转换为 NumPy 数组
    for key in imu_data:
        imu_data[key] = np.array(imu_data[key])
    return imu_data


def read_gps_data(bag_file):
    gps_data = {
        'time_gps': [],
        'utm_x': [],
        'utm_y': [],
        'altitude': []
    }
    bag = rosbag.Bag(bag_file)
    try:
        for topic, msg, t in bag.read_messages(topics=['/gps_data']):
            # 时间戳
            gps_data['time_gps'].append(t.to_sec())
            gps_data['utm_x'].append(msg.utm_easting)
            gps_data['utm_y'].append(msg.utm_northing)
            gps_data['altitude'].append(msg.altitude)
    finally:
        # 确保文件关闭
        bag.close()
            
    # 转换为 NumPy 数组
    for key in gps_data:
        gps_data[key] = np.array(gps_data[key])
    return gps_data

### hard + soft
def calibrate_magnetometer(mag_data):
    mag_xy = mag_data[:, :2]
    offset = np.mean(mag_xy, axis=0)
    mag_xy_offset = mag_xy - offset

    # 保持变量名 transform 和 mag_xy_calibrated 一致，但省略 soft-iron 校准
    transform = np.eye(2)  # 单位矩阵，表示不进行 soft-iron 变换
    mag_xy_calibrated = mag_xy_offset  # 直接使用偏移校正后的数据

    return offset, transform, mag_xy_calibrated


def calculate_yaw(mag_calibrated):
    yaw = np.arctan2(mag_calibrated[:,1], mag_calibrated[:,0])
    return yaw

def integrate_gyro(time, gyro_z):
    yaw = cumtrapz(gyro_z, time, initial=0)
    return yaw

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def complementary_filter(yaw_magnetometer, yaw_gyro, alpha=1):
    fused_yaw = np.zeros_like(yaw_magnetometer)
    fused_yaw[0] = yaw_magnetometer[0]
    for i in range(1, len(fused_yaw)):
        delta_yaw = yaw_gyro[i] - yaw_gyro[i-1]
        fused_yaw[i] = wrap_angle(fused_yaw[i-1] + delta_yaw)
        fused_yaw[i] = wrap_angle((1 - alpha) * fused_yaw[i] + alpha * yaw_magnetometer[i])
    return fused_yaw

def low_pass_filter(data, cutoff=0.1, fs=50.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def high_pass_filter(data, cutoff=0.1, fs=50.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

if __name__ == "__main__":
    # Step 1: Calibrate magnetometer using calibration bag
    calibration_bag = '/home/sophia/catkin_ws/src/LAB5/data/circle_data.bag'
    imu_data_calib = read_imu_data(calibration_bag)
    time_calib = imu_data_calib['time']
    mag_data_calib = np.vstack((imu_data_calib['magnetic_x'], imu_data_calib['magnetic_y'])).T    
    print("Calibration magnetometer data shape:", mag_data_calib.shape)
    offset, transform, mag_calibrated_calib = calibrate_magnetometer(mag_data_calib)
    print("Hard-Iron Offset:", offset)
    print("Soft-Iron Transformation Matrix:\n", transform)

    # Plot magnetometer data before and after calibration (Calibration Data)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(mag_data_calib[:,0], mag_data_calib[:,1], s=1)
    plt.title('Raw Magnetometer Data (Calibration)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.scatter(mag_calibrated_calib[:,0], mag_calibrated_calib[:,1], s=1, color='r')
    plt.title('Calibrated Magnetometer Data (Calibration)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Step 2: Read driving magnetometer data and apply calibration
    driving_bag = '/home/sophia/catkin_ws/src/LAB5/data/drive.bag'
    imu_data_drive = read_imu_data(driving_bag)
    time_drive = imu_data_drive['time']  # 提取时间戳
    mag_data_drive = np.vstack((imu_data_drive['magnetic_x'], imu_data_drive['magnetic_y'])).T
    mag_calibrated_drive = np.dot(transform, (mag_data_drive - offset).T).T

    # Calculate yaw from raw and calibrated magnetometer data
    yaw_magnetometer_raw = calculate_yaw(mag_data_drive)
    yaw_magnetometer_calibrated = calculate_yaw(mag_calibrated_drive)

    # Step 3: Read and integrate gyro data
    gyro_z = -imu_data_drive['gyro_z']
    yaw_integrated = integrate_gyro(time_drive, gyro_z)

    # 归一化时间戳，让时间从 0 开始
    time_drive_normalized = time_drive - time_drive[0]  # 对 drive 数据的时间进行归一化

    # Step 4: Plot yaw estimations before and after calibration vs time
    plt.figure(figsize=(12, 6))
    plt.plot(time_drive_normalized, yaw_magnetometer_raw, label='Raw Magnetometer Yaw')
    plt.plot(time_drive_normalized, yaw_magnetometer_calibrated, label='Calibrated Magnetometer Yaw')
    plt.title('Magnetometer Yaw Estimation Before and After Calibration')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Step 5: Plot gyro yaw estimation vs time
    plt.figure(figsize=(12, 6))
    plt.plot(time_drive_normalized, yaw_integrated, label='Integrated Gyro Yaw', color='g')
    plt.title('Gyro Yaw Estimation Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Step 6: Apply complementary filter
    fused_yaw = complementary_filter(yaw_magnetometer_calibrated, yaw_integrated, alpha=0.95)

    # Step 7: Apply low pass and high pass filters
    fs = 1 / np.mean(np.diff(time_drive))  # Sampling frequency based on gyro timestamps
    yaw_magnetometer_lp = low_pass_filter(yaw_magnetometer_calibrated, cutoff=0.1, fs=fs, order=4)
    yaw_integrated_hp = high_pass_filter(yaw_integrated, cutoff=0.1, fs=fs, order=4)

    # Read IMU yaw data
    imu_yaw = imu_data_drive['quat_z']

    # Step 8: Plot all filters and IMU yaw as 4 subplots
    plt.figure(figsize=(15, 12))

    plt.subplot(4,1,1)
    plt.plot(time_drive, yaw_magnetometer_lp, label='Low-Pass Filtered Magnetometer Yaw', color='b')
    plt.title('Low-Pass Filtered Magnetometer Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4,1,2)
    plt.plot(time_drive, yaw_integrated_hp, label='High-Pass Filtered Gyro Yaw', color='m')
    plt.title('High-Pass Filtered Gyro Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4,1,3)
    plt.plot(time_drive, fused_yaw, label='Complementary Filter Yaw', color='c')
    plt.title('Complementary Filter Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4,1,4)
    plt.plot(time_drive, imu_yaw, label='IMU Yaw', color='r')
    plt.title('IMU Yaw Estimate')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Step 9: Estimate forward velocity and trajectory using IMU and GPS data
# Step 9.1: Compute forward velocity (raw and filtered)
accel_x = imu_data_drive['accel_x']

# Raw forward velocity (uncorrected)
raw_vel_acc = -cumtrapz(accel_x, time_drive, initial=0)

# Apply a low-pass filter to smooth acceleration data
b, a = butter(4, 0.3, btype='low', fs=fs)  # 0.3 Hz cutoff frequency
accel_x_filtered = filtfilt(b, a, accel_x)

# Compute velocity from filtered acceleration
vel_acc = -cumtrapz(accel_x_filtered, time_drive, initial=0)

# Correct acceleration bias (assuming stationary in the first 100 samples)
accel_bias = np.mean(accel_x[:100])
accel_x_corrected = accel_x - accel_bias

# High-pass filter to remove low-frequency noise
b, a = butter(4, 0.001, btype='high', fs=fs)
accel_x_filtered_hp = filtfilt(b, a, accel_x_corrected)

# Recalculate velocity with corrected acceleration
filtered_velocity = -cumtrapz(accel_x_filtered_hp, time_drive, initial=0)

# Plot forward velocities
plt.figure(figsize=(10, 6))
plt.plot(time_drive, raw_vel_acc, label="Raw Forward Velocity")
plt.plot(time_drive, vel_acc, label="Filtered Forward Velocity")
plt.plot(time_drive, filtered_velocity, label="High-pass Filtered Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Forward Velocity: Raw and Filtered")
plt.legend()
plt.grid()
plt.show()

# Step 2: Read GPS data from driving bag
gps_data = read_gps_data(driving_bag)  # driving_bag 文件路径应该已经定义

# Step 9.2: Compute GPS velocity
utm_x = gps_data['utm_x']
utm_y = gps_data['utm_y']
time_gps = gps_data['time_gps']

delta_x = np.diff(utm_x)
delta_y = np.diff(utm_y)
delta_t = np.diff(time_gps)
delta_s = np.sqrt(delta_x**2 + delta_y**2)
gps_velocity = delta_s / delta_t
gps_velocity = np.insert(gps_velocity, 0, 0)  # Insert initial zero velocity

# Plot GPS velocity
plt.figure(figsize=(10, 6))
plt.plot(time_gps, gps_velocity, label="GPS Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Forward Velocity from GPS")
plt.legend()
plt.grid()
plt.show()

# Step 9.3: Dead reckoning (IMU trajectory calculation)
# Use yaw from complementary filter
yaw_corrected = fused_yaw

# Compute Easting and Northing velocities
v_e = vel_acc * np.cos(yaw_corrected)
v_n = vel_acc * np.sin(yaw_corrected)

# Integrate to get trajectory
x_e = cumtrapz(v_e, time_drive, initial=0)
x_n = cumtrapz(v_n, time_drive, initial=0)

# Adjust IMU trajectory starting point
x_e -= x_e[0]
x_n -= x_n[0]

# Step 9.4: Align IMU trajectory with GPS trajectory
# Adjust initial heading to align IMU and GPS
gps_h = np.arctan2(utm_y[3] - utm_y[0], utm_x[3] - utm_x[0])
imu_h = np.arctan2(x_n[50] - x_n[0], x_e[50] - x_e[0])
heading_offset = gps_h - imu_h

# Rotate IMU trajectory to align with GPS
imu_ec = x_e * np.cos(heading_offset) - x_n * np.sin(heading_offset)
imu_nc = x_e * np.sin(heading_offset) + x_n * np.cos(heading_offset)

# Step 9.5: Scale IMU trajectory to match GPS displacement
gps_displacement = np.sqrt(np.diff(utm_x)**2 + np.diff(utm_y)**2).sum()
imu_displacement = np.sqrt(np.diff(imu_ec)**2 + np.diff(imu_nc)**2).sum()
scaling_factor = gps_displacement / imu_displacement

imu_ec *= scaling_factor
imu_nc *= scaling_factor

# Step 9.6: Plot GPS and IMU trajectories
plt.figure(figsize=(12, 8))

# GPS trajectory
plt.subplot(2, 1, 1)
plt.plot(utm_x, utm_y, label="GPS Trajectory", color="blue")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("GPS Trajectory")
plt.legend()
plt.grid()

# IMU trajectory (aligned and scaled)
plt.subplot(2, 1, 2)
plt.plot(imu_ec, imu_nc, label="IMU Trajectory (Aligned and Scaled)", color="orange")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("IMU Trajectory (Aligned and Scaled)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Print scaling factor for reference
print(f"Scaling Factor: {scaling_factor:.4f}")
