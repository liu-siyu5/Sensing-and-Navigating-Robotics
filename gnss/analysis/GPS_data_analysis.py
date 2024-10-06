import rosbag
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Funcion para leer los datos del GPS dentro del archivo bag
def read_gps_data(bag_path):
    easting = []
    northing = []
    altitude = []
    time = []
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages(topics=['/gps_data']):
        easting.append(msg.utm_easting)
        northing.append(msg.utm_northing)
        altitude.append(msg.altitude)
        time.append(t.to_sec())
    bag.close()
    return np.array(easting), np.array(northing), np.array(altitude), np.array(time)

# Con esto le digo donde tiene que buscar los archivos bag
bag_path_open = 'src/data/ocopen_gps_data.bag'
bag_path_occluded = 'src/data/occluded_gps_data.bag'
bag_path_moving = 'src/data/walking_gps_data.bag'

# Lectura de los datos 
easting_open, northing_open, altitude_open, time_open = read_gps_data(bag_path_open)
easting_occluded, northing_occluded, altitude_occluded, time_occluded = read_gps_data(bag_path_occluded)
easting_moving, northing_moving, altitude_moving, time_moving = read_gps_data(bag_path_moving)

# Procesamiento de los datos recogidos en una zona abierta
easting_open_offset = easting_open - easting_open[0]
northing_open_offset = northing_open - northing_open[0]
centroid_easting_open = np.mean(easting_open_offset)
centroid_northing_open = np.mean(northing_open_offset)

# Procesamiento de los datos recogidos en una zona cerrada
easting_occluded_offset = easting_occluded - easting_occluded[0]
northing_occluded_offset = northing_occluded - northing_occluded[0]
centroid_easting_occluded = np.mean(easting_occluded_offset)
centroid_northing_occluded = np.mean(northing_occluded_offset)

# Create scatter plot for Northing vs. Easting
plt.figure(figsize=(10, 8))
plt.scatter(easting_open_offset, northing_open_offset, label='Open', marker='o', color='blue')
plt.scatter(easting_occluded_offset, northing_occluded_offset, label='Occluded', marker='x', color='orange')

# Disposicion de los centroides
plt.scatter(centroid_easting_open, centroid_northing_open, color='red', label='Centroid (Open)', marker='+')
plt.scatter(centroid_easting_occluded, centroid_northing_occluded, color='purple', label='Centroid (Occluded)', marker='*')

plt.text(centroid_easting_open, centroid_northing_open, f'  Centroid (Open): ({centroid_easting_open:.2f}, {centroid_northing_open:.2f})', 
         verticalalignment='bottom', horizontalalignment='right', color='red')
plt.text(centroid_easting_occluded, centroid_northing_occluded, f'  Centroid (Occluded): ({centroid_easting_occluded:.2f}, {centroid_northing_occluded:.2f})', 
         verticalalignment='bottom', horizontalalignment='right', color='purple')

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Stationary Northing vs. Easting Scatterplot (Open and Occluded)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Utilizo la misma escala para ambas
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

# Calculo de las distancias
distances_open = np.sqrt((easting_open_offset - centroid_easting_open) ** 2 + 
                          (northing_open_offset - centroid_northing_open) ** 2)

distances_occluded = np.sqrt((easting_occluded_offset - centroid_easting_occluded) ** 2 + 
                              (northing_occluded_offset - centroid_northing_occluded) ** 2)

# Plot histogram for Open
plt.figure(figsize=(10, 5))
plt.hist(distances_open, bins=10, alpha=0.7, label='Open', color='blue')
plt.xlabel('Distance to Centroid (m)')
plt.ylabel('Frequency')
plt.title('Stationary Histogram Plot for Position (Open)')
plt.legend()
plt.grid(True)
plt.show()

# Plot histogram for Occluded
plt.figure(figsize=(10, 5))
plt.hist(distances_occluded, bins=10, alpha=0.7, label='Occluded', color='orange')
plt.xlabel('Distance to Centroid (m)')
plt.ylabel('Frequency')
plt.title('Stationary Histogram Plot for Position (Occluded)')
plt.legend()
plt.grid(True)
plt.show()

# Graficas de los datos en movimiento: Northing vs. Easting Scatterplot with Line of Best Fit
plt.figure(figsize=(10, 8))
plt.scatter(easting_moving, northing_moving, label='Moving Data', marker='o', color='blue')

# Fit line to moving data
X_moving = easting_moving.reshape(-1, 1)  # Reshape for LinearRegression
y_moving = northing_moving
model = LinearRegression().fit(X_moving, y_moving)
y_fit = model.predict(X_moving)

# Plot the line of best fit
plt.plot(easting_moving, y_fit, color='red', label='Line of Best Fit', linewidth=2)

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Moving Data Northing vs. Easting Scatterplot with Line of Best Fit')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Moving Data: Altitude vs. Time Plot
plt.figure(figsize=(10, 5))
plt.plot(time_moving, altitude_moving, label='Moving Data', marker='o', color='green')

plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Time Plot (Moving Data)')
plt.legend()
plt.grid(True)
plt.show()