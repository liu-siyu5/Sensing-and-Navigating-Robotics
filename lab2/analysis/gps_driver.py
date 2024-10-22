#!/usr/bin/env python3
import serial
import rospy
from lab2_gps_driver.msg import Customgps 
from std_msgs.msg import Header
import time
import utm 


file_path = '/home/sophia/catkin_ws/src/lab2_gps_driver/data/walking_open_gps_open.ubx'
# Function to read and filter NMEA lines
def read_filtered_lines(file_path):
    gngga_lines = []

    try:
        with open(file_path, 'rb') as file:  # Open in binary mode
            while True:
                chunk = file.read(1024)  # Read 1024 bytes at a time
                if not chunk:
                    break  # End of file

                # Decode the chunk to a string
                try:
                    decoded_chunk = chunk.decode('ascii', errors='ignore')  # Ignore decoding errors
                except UnicodeDecodeError as e:
                    print(f"Decoding error: {e}")
                    continue

                # Split the decoded chunk into lines
                lines = decoded_chunk.splitlines()

                # Filter lines for GNGGA sentences
                for line in lines:
                    if 'GNGGA' in line:
                        gngga_lines.append(line.strip())

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return gngga_lines


def isGNGGAinStr(inputString):
    if '$GNGGA' in inputString:
        print('Great Success')
        return True
    else:
        print('GNGGA not found in string')
        return False


def degMinstoDegDec(LatOrLong):
    deg = int(LatOrLong // 100)  
    mins = LatOrLong % 100 
    degDec = mins / 60  
    return deg + degDec


def LatLongSignConvetion(LatOrLong, LatOrLongDir):
    if LatOrLongDir == "W" or LatOrLongDir == "S":
        return -LatOrLong
    return LatOrLong

#BLOCK2
def UTCtoUTCEpoch(UTC):
    TimeSinceEpoch = time.time()
    CurrentTimeSec = int(TimeSinceEpoch) 
    CurrentTimeNsec = int((TimeSinceEpoch - CurrentTimeSec) * 1e9)
    return [CurrentTimeSec, CurrentTimeNsec]

#BLOCK3
def convertToUTM(LatitudeSigned, LongitudeSigned):
    UTMVals = utm.from_latlon(LatitudeSigned, LongitudeSigned)
    return [UTMVals[0], UTMVals[1], UTMVals[2], UTMVals[3]] 

#BLOAKC4
rospy.init_node('gps_driver_node')
gps_pub = rospy.Publisher('gps_data', Customgps, queue_size=10)


while not rospy.is_shutdown():
    gngga_lines = read_filtered_lines(file_path)

    if not gngga_lines:
        rospy.logwarn("No GNGGA lines found.")
        continue

    for gnggaRead in gngga_lines:
        if not isGNGGAinStr(gnggaRead):
            rospy.logwarn("GNGGA not found in string.")
            continue

        gnggaSplit = gnggaRead.split(",")
    # Rest of your processing code here

    print("yes iam running")

    
    if len(gnggaSplit) < 9:
        rospy.logerr("Incomplete GNGGA sentence received.")
        continue

    try:
        UTC = float(gnggaSplit[1])
        Latitude = float(gnggaSplit[2])
        LatitudeDir = gnggaSplit[3]
        Longitude = float(gnggaSplit[4])
        LongitudeDir = gnggaSplit[5]
        FixQuality = int(gnggaSplit[6])  # GNSS fix quality
        NumSatellites = int(gnggaSplit[7])  # Optional: Number of satellites
        HDOP = float(gnggaSplit[8])
        Altitude = float(gnggaSplit[9])
    except ValueError:
        rospy.logerr("Error parsing GNGGA data.")
        continue

    


    
    Latitude = degMinstoDegDec(Latitude)
    Longitude = degMinstoDegDec(Longitude)

   
    LatitudeSigned = LatLongSignConvetion(Latitude, LatitudeDir)
    LongitudeSigned = LatLongSignConvetion(Longitude, LongitudeDir)

 
    CurrentTime = UTCtoUTCEpoch(UTC)


    utm_format = convertToUTM(LatitudeSigned, LongitudeSigned)


    gps_message = Customgps()
    gps_message.header.frame_id = 'GPS2_Frame'
    gps_message.header.stamp.secs = CurrentTime[0]
    gps_message.header.stamp.nsecs = CurrentTime[1]
    gps_message.latitude = LatitudeSigned
    gps_message.longitude = LongitudeSigned
    gps_message.altitude = Altitude  
    gps_message.utm_easting = utm_format[0]
    gps_message.utm_northing = utm_format[1]
    gps_message.zone = utm_format[2]
    gps_message.letter = utm_format[3]
    gps_message.hdop = HDOP
    gps_message.fix_quality = FixQuality

   
    gps_pub.publish(gps_message)

    rospy.sleep(1)

