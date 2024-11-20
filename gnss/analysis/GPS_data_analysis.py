import rosbag
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 定义函数：从ROS bag中提取GPS数据
def extract_gps_data(bag_path, topic_name='/gps_data'):
    data = {'easting': [], 'northing': [], 'altitude': [], 'time': []}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            data['easting'].append(msg.utm_easting)
            data['northing'].append(msg.utm_northing)
            data['altitude'].append(msg.altitude)
            data['time'].append(t.to_sec())
    return {key: np.array(value) for key, value in data.items()}

# 定义函数：计算偏移量与质心
def process_data(easting, northing):
    easting_offset = easting - easting[0]
    northing_offset = northing - northing[0]
    centroid = (np.mean(easting_offset), np.mean(northing_offset))
    return easting_offset, northing_offset, centroid

# 定义函数：绘制散点图
def plot_scatter(easting_offsets, northing_offsets, centroids, labels, colors, title):
    plt.figure(figsize=(10, 8))
    for easting, northing, centroid, label, color in zip(easting_offsets, northing_offsets, centroids, labels, colors):
        plt.scatter(easting, northing, label=label, marker='o', color=color)
        plt.scatter(*centroid, color='red' if label == 'Open' else 'purple', label=f'Centroid ({label})', marker='+')
        plt.text(centroid[0], centroid[1], f'  Centroid ({label}): ({centroid[0]:.2f}, {centroid[1]:.2f})', 
                 verticalalignment='bottom', horizontalalignment='right', color=color)
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 定义函数：绘制高度-时间变化图
def plot_altitude_vs_time(times, altitudes, labels, colors, title):
    plt.figure(figsize=(10, 5))
    for time, altitude, label, color in zip(times, altitudes, labels, colors):
        plt.plot(time, altitude, label=label, marker='o', color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 定义函数：绘制距离直方图
def plot_histogram(distances, labels, colors, title):
    plt.figure(figsize=(10, 5))
    for distance, label, color in zip(distances, labels, colors):
        plt.hist(distance, bins=10, alpha=0.7, label=label, color=color)
    plt.xlabel('Distance to Centroid (m)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 定义函数：移动数据线性拟合
def plot_moving_fit(easting, northing, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(easting, northing, label='Moving Data', marker='o', color='blue')
    model = LinearRegression().fit(easting.reshape(-1, 1), northing)
    y_fit = model.predict(easting.reshape(-1, 1))
    plt.plot(easting, y_fit, color='red', label='Line of Best Fit', linewidth=2)
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 数据文件路径
bag_paths = {
    'Open': '/home/sophia/Documents/5554/LAB1/gnss/data/ocopen_gps_data.bag',
    'Occluded': '/home/sophia/Documents/5554/LAB1/gnss/data/occluded_gps_data.bag',
    'Moving': '/home/sophia/Documents/5554/LAB1/gnss/data/walking_gps_data.bag'
}

# 数据提取
data_open = extract_gps_data(bag_paths['Open'])
data_occluded = extract_gps_data(bag_paths['Occluded'])
data_moving = extract_gps_data(bag_paths['Moving'])

# 数据处理
open_easting_offset, open_northing_offset, open_centroid = process_data(data_open['easting'], data_open['northing'])
occluded_easting_offset, occluded_northing_offset, occluded_centroid = process_data(data_occluded['easting'], data_occluded['northing'])

# 绘制散点图
plot_scatter(
    [open_easting_offset, occluded_easting_offset],
    [open_northing_offset, occluded_northing_offset],
    [open_centroid, occluded_centroid],
    ['Open', 'Occluded'],
    ['blue', 'orange'],
    'Stationary Northing vs. Easting Scatterplot (Open and Occluded)'
)

# 绘制高度-时间变化图
plot_altitude_vs_time(
    [data_open['time'], data_occluded['time']],
    [data_open['altitude'], data_occluded['altitude']],
    ['Open', 'Occluded'],
    ['blue', 'orange'],
    'Altitude vs. Time Plot (Open and Occluded)'
)

# 计算与质心的距离
open_distances = np.sqrt((open_easting_offset - open_centroid[0])**2 + (open_northing_offset - open_centroid[1])**2)
occluded_distances = np.sqrt((occluded_easting_offset - occluded_centroid[0])**2 + (occluded_northing_offset - occluded_centroid[1])**2)

# 绘制距离直方图
plot_histogram(
    [open_distances, occluded_distances],
    ['Open', 'Occluded'],
    ['blue', 'orange'],
    'Stationary Histogram Plot for Position'
)

# 移动数据拟合与绘图
plot_moving_fit(data_moving['easting'], data_moving['northing'], 'Moving Data Northing vs. Easting Scatterplot with Best Fit Line')

# 绘制移动数据的高度-时间图
plot_altitude_vs_time(
    [data_moving['time']],
    [data_moving['altitude']],
    ['Moving Data'],
    ['green'],
    'Altitude vs. Time Plot (Moving Data)'
)
