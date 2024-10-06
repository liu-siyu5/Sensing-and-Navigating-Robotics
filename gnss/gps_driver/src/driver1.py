#!/usr/bin/env python3
import serial
import rospy
from gps_driver.msg import Customgps  # Import the custom GPS message
from std_msgs.msg import Header
import time
import utm 

# Function to read data from serial
def ReadFromSerial():
    try:
        serialPort = serial.Serial('/dev/ttyUSB0', baudrate=4800, timeout=1)
        gpggaRead = serialPort.readline().decode('ascii', errors='replace').strip()  # Decode byte string to regular string
        print(gpggaRead)
        serialPort.close()
        return gpggaRead
    except serial.SerialException as e:
        rospy.logerr(f"Serial Exception: {e}")
        return None

# Function to check if the string contains a GPGGA sentence
def isGPGGAinStr(inputString):
    if '$GPGGA' in inputString:
        print('Great Success')
        return True
    else:
        print('GPGGA not found in string')
        return False

# Function to convert degree minutes to decimal degrees
def degMinstoDegDec(LatOrLong):
    deg = int(LatOrLong // 100)  # Extract degrees
    mins = LatOrLong % 100  # Extract minutes
    degDec = mins / 60  # Convert minutes to decimal degrees
    return deg + degDec

# Function to apply sign based on direction (W/S are negative)
def LatLongSignConvetion(LatOrLong, LatOrLongDir):
    if LatOrLongDir == "W" or LatOrLongDir == "S":
        return -LatOrLong
    return LatOrLong

# Function to convert UTC to epoch time
def UTCtoUTCEpoch(UTC):
    TimeSinceEpoch = time.time()
    CurrentTimeSec = int(TimeSinceEpoch)  # Total seconds since epoch
    CurrentTimeNsec = int((TimeSinceEpoch - CurrentTimeSec) * 1e9)
    return [CurrentTimeSec, CurrentTimeNsec]

# Function to convert Latitude and Longitude to UTM
def convertToUTM(LatitudeSigned, LongitudeSigned):
    UTMVals = utm.from_latlon(LatitudeSigned, LongitudeSigned)
    return [UTMVals[0], UTMVals[1], UTMVals[2], UTMVals[3]]  # Easting, Northing, Zone, Letter

# Initialize ROS node and publisher
rospy.init_node('gps_driver_node')
gps_pub = rospy.Publisher('gps_data', Customgps, queue_size=10)

# Main loop
while not rospy.is_shutdown():
    # Read GPS data from serial port
    gpggaRead = ReadFromSerial()

    # Skip if no data is read
    if gpggaRead is None:
        continue

    # Check if it's a GPGGA sentence, skip if it's not
    if not isGPGGAinStr(gpggaRead):
        continue

    # Split the GPGGA sentence into fields
    gpggaSplit = gpggaRead.split(",")

    # Check if the GPGGA sentence has the expected number of fields (usually 15 fields)
    if len(gpggaSplit) < 9:
        rospy.logerr("Incomplete GPGGA sentence received.")
        continue

    try:
        # Parse the GPGGA sentence
        UTC = float(gpggaSplit[1])
        Latitude = float(gpggaSplit[2])
        LatitudeDir = gpggaSplit[3]
        Longitude = float(gpggaSplit[4])
        LongitudeDir = gpggaSplit[5]
        HDOP = float(gpggaSplit[8])
    except ValueError:
        rospy.logerr("Error parsing GPGGA data.")
        continue

    # Convert Latitude and Longitude to decimal degrees
    Latitude = degMinstoDegDec(Latitude)
    Longitude = degMinstoDegDec(Longitude)

    # Apply direction sign to Latitude and Longitude
    LatitudeSigned = LatLongSignConvetion(Latitude, LatitudeDir)
    LongitudeSigned = LatLongSignConvetion(Longitude, LongitudeDir)

    # Get current epoch time
    CurrentTime = UTCtoUTCEpoch(UTC)

    # Convert to UTM
    utm_format = convertToUTM(LatitudeSigned, LongitudeSigned)

    # Create GPS message and populate data
    gps_message = Customgps()
    gps_message.header.frame_id = 'GPS1_Frame'
    gps_message.header.stamp.secs = CurrentTime[0]
    gps_message.header.stamp.nsecs = CurrentTime[1]
    gps_message.latitude = LatitudeSigned
    gps_message.longitude = LongitudeSigned
    gps_message.altitude = float(gpggaSplit[9])  # Set if available
    gps_message.utm_easting = utm_format[0]
    gps_message.utm_northing = utm_format[1]
    gps_message.zone = utm_format[2]
    gps_message.letter = utm_format[3]
    gps_message.hdop = HDOP

    # Publish the GPS message
    gps_pub.publish(gps_message)

    # Sleep for 1 second (publishing at 1 Hz)
    rospy.sleep(1)

