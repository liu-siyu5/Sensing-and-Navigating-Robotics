import rosbag
import numpy as np
import matplotlib.pyplot as plt
bag_path = '/home/sophia/catkin_ws/src/data/gps_data_sectry.bag'
easting_open = []
northing_open = []
altitude_open = []
time_open = []
bag = rosbag.Bag(bag_path)
for topic, msg, t in bag.read_messages(topics=['/gps_data']):
    easting_open.append(msg.utm_easting)
    northing_open.append(msg.utm_northing)
    altitude_open.append(msg.altitude)
    time_open.append(t.to_sec())
easting_open = np.array(easting_open)
northing_open = np.array(northing_open)
altitude_open = np.array(altitude_open)
time_open = np.array(time_open)

bag.close()

print("Easting Data: ", easting_open)
print("Northing Data: ", northing_open)
print("Altitude Data: ", altitude_open)
print("Time Data: ", time_open)

easting_open_offset = easting_open - easting_open[0]
northing_open_offset = northing_open - northing_open[0]

centroid_easting_open = np.mean(easting_open_offset)
centroid_northing_open = np.mean(northing_open_offset)

print(f"Centroid (Open): Easting = {centroid_easting_open}, Northing = {centroid_northing_open}")

deviation_easting_open = easting_open_offset - centroid_easting_open
deviation_northing_open = northing_open_offset - centroid_northing_open

plt.scatter(easting_open_offset, northing_open_offset, label='Open', marker='o')

plt.scatter(centroid_easting_open, centroid_northing_open, color='red', label='Centroid (Open)', marker='+')

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Stationary Northing vs. Easting Scatterplot')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(time_open, altitude_open, label='Open', marker='o')

plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Stationary Altitude vs. Time Plot')
plt.legend()
plt.grid(True)
plt.show()


distances_open = np.sqrt(deviation_easting_open**2 + deviation_northing_open**2)

plt.hist(distances_open, bins=10, alpha=0.7, label='Open')
plt.xlabel('Distance to Centroid (m)')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to Centroid (Open)')
plt.grid(True)
plt.show()

easting_moving_open = easting_open
northing_moving_open = northing_open

m_open, b_open = np.polyfit(easting_moving_open, northing_moving_open, 1)

plt.scatter(easting_moving_open, northing_moving_open, label='Moving (Open)', marker='o')
plt.plot(easting_moving_open, m_open * easting_moving_open + b_open, label='Best Fit (Open)', color='red')

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Moving Northing vs. Easting Scatterplot with Best Fit Line')
plt.legend()
plt.grid(True)
plt.show()


plt.plot(time_open, altitude_open, label='Moving (Open)', marker='o')

plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Moving Altitude vs. Time Plot')
plt.legend()
plt.grid(True)
plt.show()

