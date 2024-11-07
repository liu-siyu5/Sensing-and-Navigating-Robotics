import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import allantools

# 读取 ROS bag 文件
bag_file = '/home/sophia/Documents/5554/LAB3/data/imu_data.bag'
b = bagreader(bag_file)

# 显示所有 topics
print("Available topics in bag:", b.topic_table)

# 将数据转换为 CSV 并读取 IMU 数据
imu_csv_file = b.message_by_topic('/imu')
imu_data = pd.read_csv(imu_csv_file)
imu_data.head()
print(imu_data.columns)

# 提取和清理数据
orientation_data = imu_data[['IMU.orientation.x', 'IMU.orientation.y', 'IMU.orientation.z', 'IMU.orientation.w']]
orientation_data.columns = ['qx', 'qy', 'qz', 'qw']

angular_data = imu_data[['IMU.angular_velocity.x', 'IMU.angular_velocity.y', 'IMU.angular_velocity.z']]
angular_data.columns = ['angular_x', 'angular_y', 'angular_z']

accel_data = imu_data[['IMU.linear_acceleration.x', 'IMU.linear_acceleration.y', 'IMU.linear_acceleration.z']]
accel_data.columns = ['accel_x', 'accel_y', 'accel_z']

mag_field_data = imu_data[['MagField.magnetic_field.x', 'MagField.magnetic_field.y', 'MagField.magnetic_field.z']]
mag_field_data.columns = ['mag_x', 'mag_y', 'mag_z']

# 转换四元数到欧拉角
def quaternion_to_euler(row):
    qx, qy, qz, qw = row
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))
    return np.degrees([roll, pitch, yaw])

euler_angles = orientation_data.apply(quaternion_to_euler, axis=1, result_type='expand')
euler_angles.columns = ['Roll', 'Pitch', 'Yaw']

## 在angular_data和accel_data处理完后添加打印
print("Second code - Gyro data (first 5 rows):")
print(angular_data.head())
print("Second code - Accel data (first 5 rows):")
print(accel_data.head())

## 在euler_angles计算后添加打印
print("Second code - Euler angles (first 5 rows):")
print(euler_angles.head())

angular_data['angular_x'] = np.degrees(angular_data['angular_x'])
angular_data['angular_y'] = np.degrees(angular_data['angular_y'])
angular_data['angular_z'] = np.degrees(angular_data['angular_z'])


# 生成与第一个代码类似的时间序列图
time = np.arange(len(euler_angles))

fig, axs = plt.subplots(3, 4, figsize=(15, 10))
axs[0, 0].plot(time, euler_angles['Yaw'], label='Yaw')
axs[0, 0].set_title('Yaw changes')
axs[0, 1].plot(time, euler_angles['Pitch'], label='Pitch')
axs[0, 1].set_title('Pitch changes')
axs[0, 2].plot(time, euler_angles['Roll'], label='Roll')
axs[0, 2].set_title('Roll changes')

axs[1, 0].plot(time, angular_data['angular_x'])
axs[1, 0].set_title('Angular Velocity X changes')
axs[1, 1].plot(time, angular_data['angular_y'])
axs[1, 1].set_title('Angular Velocity Y changes')
axs[1, 2].plot(time, angular_data['angular_z'])
axs[1, 2].set_title('Angular Velocity Z changes')

axs[2, 0].plot(time, accel_data['accel_x'])
axs[2, 0].set_title('Linear Acceleration X changes')
axs[2, 1].plot(time, accel_data['accel_y'])
axs[2, 1].set_title('Linear Acceleration Y changes')
axs[2, 2].plot(time, accel_data['accel_z'])
axs[2, 2].set_title('Linear Acceleration Z changes')

axs[0, 3].plot(time, mag_field_data['mag_x'])
axs[0, 3].set_title('Magnetic Field X changes')
axs[1, 3].plot(time, mag_field_data['mag_y'])
axs[1, 3].set_title('Magnetic Field Y changes')
axs[2, 3].plot(time, mag_field_data['mag_z'])
axs[2, 3].set_title('Magnetic Field Z changes')

for ax in axs.flat:
    ax.set(xlabel='Time (1/40s)', ylabel='Value')
    ax.label_outer()

plt.tight_layout()
plt.show()

# Allan Variance 和直方图分析
# Allan Variance 分析（与第二个代码保持一致）
gyro_data = imu_data[['IMU.angular_velocity.x', 'IMU.angular_velocity.y', 'IMU.angular_velocity.z']]
gyro_data.columns = ['GyroX', 'GyroY', 'GyroZ']

def allan_variance_analysis(data, fs=40):
    Allan_dev = allantools.oadev(data, rate=fs, data_type="phase", taus="all")
    tau, allan_dev, *_ = Allan_dev
    allan_var = allan_dev**2
    return tau, allan_var

# 计算各轴的 Allan Variance
tau_x, allan_var_x = allan_variance_analysis(gyro_data['GyroX'])
tau_y, allan_var_y = allan_variance_analysis(gyro_data['GyroY'])
tau_z, allan_var_z = allan_variance_analysis(gyro_data['GyroZ'])

# 绘制 Allan Variance 图
plt.figure()
plt.title('Gyro Allan Variance vs. Tau')
plt.plot(tau_x, allan_var_x, label='Gyro X')
plt.plot(tau_y, allan_var_y, label='Gyro Y')
plt.plot(tau_z, allan_var_z, label='Gyro Z')
plt.xlabel(r'$\tau$ [sec]')
plt.ylabel('Allan Variance [deg^2/sec^2]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 欧拉角直方图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(euler_angles['Roll'], bins=20, color='red', alpha=0.7)
plt.xlabel('Roll (degrees)')
plt.ylabel('Frequency')
plt.title('Histogram of Roll')

plt.subplot(1, 3, 2)
plt.hist(euler_angles['Pitch'], bins=20, color='green', alpha=0.7)
plt.xlabel('Pitch (degrees)')
plt.ylabel('Frequency')
plt.title('Histogram of Pitch')

plt.subplot(1, 3, 3)
plt.hist(euler_angles['Yaw'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Yaw (degrees)')
plt.ylabel('Frequency')
plt.title('Histogram of Yaw')

plt.tight_layout()
plt.show()
