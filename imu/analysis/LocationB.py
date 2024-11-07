import rosbag
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import allantools
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt

# 读取指定的 ROSbag 文件
bag = rosbag.Bag('/home/sophia/Documents/5554/LAB3/data/LocationB.bag')
data = {'time': [], 'gyro_x': [], 'gyro_y': [], 'gyro_z': [], 'accel_x': [], 'accel_y': [], 'accel_z': []}

# 遍历 bag 文件中的消息并提取角速度和加速度数据
for topic, msg, t in bag.read_messages(topics=['/vectornav']):
    data['time'].append(t.to_sec())
    data_values = msg.data.split(',')
    data['gyro_x'].append(float(data_values[1]))
    data['gyro_y'].append(float(data_values[2]))
    data['gyro_z'].append(float(data_values[3]))
    data['accel_x'].append(float(data_values[4]))
    data['accel_y'].append(float(data_values[5]))
    data['accel_z'].append(float(data_values[6]))

bag.close()

# 创建高通滤波器
def highpass_filter(data, cutoff=0.1, fs=40, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# 转换为 DataFrame 便于处理
df = pd.DataFrame(data)
df['time'] = df['time'] - df['time'][0]  # 将时间起点设为 0

# 时间间隔（假设采样率 fs=40Hz）
fs = 40
dt = 1 / fs
print(df.head())  # 查看数据的前几行

# 对陀螺仪和加速度数据进行积分以获得角度
df['angle_gyro_x'] = cumtrapz(df['gyro_x'], dx=dt, initial=0)
df['angle_gyro_y'] = cumtrapz(df['gyro_y'], dx=dt, initial=0)
df['angle_gyro_z'] = cumtrapz(df['gyro_z'], dx=dt, initial=0)

df['angle_accel_x'] = cumtrapz(df['accel_x'], dx=dt, initial=0)
df['angle_accel_y'] = cumtrapz(df['accel_y'], dx=dt, initial=0)
df['angle_accel_z'] = cumtrapz(df['accel_z'], dx=dt, initial=0)

# 对积分后的角度数据进行高通滤波，校正漂移误差
df['angle_gyro_x_filtered'] = highpass_filter(df['angle_gyro_x'])
df['angle_gyro_y_filtered'] = highpass_filter(df['angle_gyro_y'])
df['angle_gyro_z_filtered'] = highpass_filter(df['angle_gyro_z'])

df['angle_accel_x_filtered'] = highpass_filter(df['angle_accel_x'])
df['angle_accel_y_filtered'] = highpass_filter(df['angle_accel_y'])
df['angle_accel_z_filtered'] = highpass_filter(df['angle_accel_z'])

# Allan Deviation 计算函数
def allan_deviation_analysis(data, fs=40):
    Allan_dev = allantools.oadev(data, rate=fs, data_type="phase", taus="all")
    tau, allan_dev, *_ = Allan_dev
    allan_var = allan_dev**2
    return tau, allan_var

# 绘制 Gyroscope Rate 的 Allan 偏差（图 1）
plt.figure(figsize=(10, 8))
tau_gx, oadev_gx = allan_deviation_analysis(df['gyro_x'])
tau_gy, oadev_gy = allan_deviation_analysis(df['gyro_y'])
tau_gz, oadev_gz = allan_deviation_analysis(df['gyro_z'])

plt.loglog(tau_gx, oadev_gx, label="Gyro X Rate", color='red')
plt.loglog(tau_gy, oadev_gy, label="Gyro Y Rate", color='blue')
plt.loglog(tau_gz, oadev_gz, label="Gyro Z Rate", color='yellow')
plt.grid(which="both")
plt.title("Allan Deviation for Gyroscope Rate")
plt.xlabel("Tau (seconds)")
plt.ylabel("Allan Deviation")
plt.legend()
plt.show()

# 绘制 Gyroscope Angle 的 Allan 偏差（图 2）
plt.figure(figsize=(10, 8))
tau_ax, oadev_ax = allan_deviation_analysis(df['angle_gyro_x_filtered'])
tau_ay, oadev_ay = allan_deviation_analysis(df['angle_gyro_y_filtered'])
tau_az, oadev_az = allan_deviation_analysis(df['angle_gyro_z_filtered'])

plt.loglog(tau_ax, oadev_ax, label="Gyro X Angle", color='green')
plt.loglog(tau_ay, oadev_ay, label="Gyro Y Angle", color='purple')
plt.loglog(tau_az, oadev_az, label="Gyro Z Angle", color='orange')
plt.grid(which="both")
plt.title("Allan Deviation for Gyroscope Angle")
plt.xlabel("Tau (seconds)")
plt.ylabel("Allan Deviation")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# 绘制 Accelerometer Rate 的 Allan 偏差（图 3）
plt.figure(figsize=(10, 8))
tau_ax, oadev_ax = allan_deviation_analysis(df['accel_x'])
tau_ay, oadev_ay = allan_deviation_analysis(df['accel_y'])
tau_az, oadev_az = allan_deviation_analysis(df['accel_z'])

plt.loglog(tau_ax, oadev_ax, label="Accel X Rate", color='red')
plt.loglog(tau_ay, oadev_ay, label="Accel Y Rate", color='blue')
plt.loglog(tau_az, oadev_az, label="Accel Z Rate", color='yellow')
plt.grid(which="both")
plt.title("Allan Deviation for Accelerometer Rate")
plt.xlabel("Tau (seconds)")
plt.ylabel("Allan Deviation")
plt.legend()
plt.show()

# 绘制 Accelerometer Angle 的 Allan 偏差（图 4）
plt.figure(figsize=(10, 8))
tau_ax, oadev_ax = allan_deviation_analysis(df['angle_accel_x_filtered'],)
tau_ay, oadev_ay = allan_deviation_analysis(df['angle_accel_y_filtered'])
tau_az, oadev_az = allan_deviation_analysis(df['angle_accel_z_filtered'])

plt.loglog(tau_ax, oadev_ax, label="Accel X Angle", color='green')
plt.loglog(tau_ay, oadev_ay, label="Accel Y Angle", color='purple')
plt.loglog(tau_az, oadev_az, label="Accel Z Angle", color='orange')
plt.grid(which="both")
plt.title("Allan Deviation for Accelerometer Angle")
plt.xlabel("Tau (seconds)")
plt.ylabel("Allan Deviation")
plt.legend()
plt.show()
