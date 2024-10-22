import rosbag
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to read GPS data from bag files
def read_gps_data(bag_path):
    easting = []
    northing = []
    altitude = []
    time = []
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages(topics=['gps_data']):
        easting.append(msg.utm_easting)
        northing.append(msg.utm_northing)
        altitude.append(msg.altitude)
        time.append(t.to_sec())
    bag.close()
    return np.array(easting), np.array(northing), np.array(altitude), np.array(time)

# File paths for the bag files
bag_path_open = '/home/sophia/catkin_ws/src/lab2_gps_driver/bag/Stationary_open.bag'
bag_path_occluded = '/home/sophia/catkin_ws/src/lab2_gps_driver/bag/Stationary_occluded.bag'
bag_path_moving_occ = '/home/sophia/catkin_ws/src/lab2_gps_driver/bag/Walking_occluded.bag'
bag_path_moving_open = '/home/sophia/catkin_ws/src/lab2_gps_driver/bag/Walking_open.bag'

# Read the GPS data for all datasets
easting_open, northing_open, altitude_open, time_open = read_gps_data(bag_path_open)
print("Number of easting_open data points:", len(easting_open))
print("Easting_open data:", easting_open)
easting_occluded, northing_occluded, altitude_occluded, time_occluded = read_gps_data(bag_path_occluded)
easting_moving_occ, northing_moving_occ, altitude_moving_occ, time_moving_occ = read_gps_data(bag_path_moving_occ)
easting_moving_open, northing_moving_open, altitude_moving_open, time_moving_open = read_gps_data(bag_path_moving_open)

# Offset the data for Open area
easting_open_offset = easting_open - easting_open[0]
northing_open_offset = northing_open - northing_open[0]
centroid_easting_open = np.mean(easting_open_offset)
centroid_northing_open = np.mean(northing_open_offset)

# Offset the data for Occluded area
easting_occluded_offset = easting_occluded - easting_occluded[0]
northing_occluded_offset = northing_occluded - northing_occluded[0]
centroid_easting_occluded = np.mean(easting_occluded_offset)
centroid_northing_occluded = np.mean(northing_occluded_offset)

# Calculate deviations (standard deviation)
std_easting_open = np.std(easting_open_offset)
std_northing_open = np.std(northing_open_offset)
std_easting_occluded = np.std(easting_occluded_offset)
std_northing_occluded = np.std(northing_occluded_offset)

# Print deviation information
print(f"Open Area Easting Std Dev: {std_easting_open:.2f}, Northing Std Dev: {std_northing_open:.2f}")
print(f"Occluded Area Easting Std Dev: {std_easting_occluded:.2f}, Northing Std Dev: {std_northing_occluded:.2f}")

# Create scatter plot for Northing vs. Easting with Centroids and Deviations
plt.figure(figsize=(10, 8))
plt.scatter(easting_open_offset, northing_open_offset, label='Open', marker='o', color='blue')
plt.scatter(easting_occluded_offset, northing_occluded_offset, label='Occluded', marker='x', color='orange')

# Mark the centroids
plt.scatter(centroid_easting_open, centroid_northing_open, color='red', label='Centroid (Open)', marker='+')
plt.scatter(centroid_easting_occluded, centroid_northing_occluded, color='purple', label='Centroid (Occluded)', marker='*')

# Add standard deviation annotations
plt.text(centroid_easting_open, centroid_northing_open, f'  Std Dev (Open): E={std_easting_open:.2f}, N={std_northing_open:.2f}', 
         verticalalignment='bottom', horizontalalignment='right', color='red')
plt.text(centroid_easting_occluded, centroid_northing_occluded, f'  Std Dev (Occluded): E={std_easting_occluded:.2f}, N={std_northing_occluded:.2f}', 
         verticalalignment='bottom', horizontalalignment='right', color='purple')

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Stationary Northing vs. Easting Scatterplot (Open and Occluded) with Std Dev')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Altitude vs. Time Plot for Open and Occluded
plt.figure(figsize=(10, 5))
plt.plot(time_open, altitude_open, label='Open', marker='o', color='blue')
plt.plot(time_occluded, altitude_occluded, label='Occluded', marker='x', color='orange')

plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Time Plot (Open and Occluded)')
plt.legend()
plt.grid(True)
plt.show()

# Plot histograms for distance from centroid for Open and Occluded
distances_open = np.sqrt((easting_open_offset - centroid_easting_open) ** 2 + 
                         (northing_open_offset - centroid_northing_open) ** 2)
distances_occluded = np.sqrt((easting_occluded_offset - centroid_easting_occluded) ** 2 + 
                             (northing_occluded_offset - centroid_northing_occluded) ** 2)

plt.figure(figsize=(10, 5))
plt.hist(distances_open, bins=10, alpha=0.7, label='Open', color='blue')
plt.xlabel('Distance to Centroid (m)')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to Centroid (Open)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(distances_occluded, bins=10, alpha=0.7, label='Occluded', color='orange')
plt.xlabel('Distance to Centroid (m)')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to Centroid (Occluded)')
plt.legend()
plt.grid(True)
plt.show()

# Moving Data: Northing vs. Easting Scatterplot for Open and Occluded areas
plt.figure(figsize=(10, 8))
plt.scatter(easting_moving_open, northing_moving_open, label='Moving Open', marker='o', color='green')
plt.scatter(easting_moving_occ, northing_moving_occ, label='Moving Occluded', marker='x', color='red')

# Fit lines to moving data for Open and Occluded
X_moving_open = easting_moving_open.reshape(-1, 1)
X_moving_occ = easting_moving_occ.reshape(-1, 1)
model_open = LinearRegression().fit(X_moving_open, northing_moving_open)
model_occ = LinearRegression().fit(X_moving_occ, northing_moving_occ)
y_fit_open = model_open.predict(X_moving_open)
y_fit_occ = model_occ.predict(X_moving_occ)

# Plot the line of best fit
plt.plot(easting_moving_open, y_fit_open, color='blue', label='Best Fit (Open)', linewidth=2)
plt.plot(easting_moving_occ, y_fit_occ, color='orange', label='Best Fit (Occluded)', linewidth=2)

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Moving Northing vs. Easting Scatterplot with Best Fit Line')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Moving Data: Altitude vs. Time Plot for Open and Occluded
plt.figure(figsize=(10, 5))
plt.plot(time_moving_open, altitude_moving_open, label='Moving Open', marker='o', color='green')
plt.plot(time_moving_occ, altitude_moving_occ, label='Moving Occluded', marker='x', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Time Plot (Open and Occluded)')
plt.legend()
plt.grid(True)
plt.show()

