#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
rp = rospkg.RosPack()

class ColourDetector:
    def __init__(self):
        self.ros_image = Image()
        self.cv_image_data = np.array([])
        self.cv_image = cv2.Mat(self.cv_image_data)
        self.green_result, self.red_result, self.blue_result = None, None, None
        self.bridge = CvBridge()

        self.scale = 0.5
        self.MAX_AREA = 12000
        self.up = 2.0
        self.low = 1.3

        rospy.init_node('detect_colour_node', anonymous=True)

        self.target_colour = rospy.get_param('~target_colour', '').lower()

        rospy.loginfo('TASK 4 BEACON: %s', self.target_colour)
        self.path = rp.get_path('com2009_team26') + '/snaps/'
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        
    def image_callback(self, msg):
        self.ros_image = msg
        self.cv_image = self.bridge.imgmsg_to_cv2(self.ros_image, desired_encoding='bgr8')
        # print(self.cv_image)
        # For colour detection
        height, width = self.cv_image.shape[:2]
        self.cv_image = cv2.resize(self.cv_image, (round(self.scale * width), round(self.scale * height)))
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        if not self.target_colour:
            return
        if self.target_colour == 'red':
            # RED
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([5, 255, 255])
            mask_red = cv2.inRange(self.hsv_image, lower_red, upper_red)
            self.red_result = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask_red)

            contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(self.cv_image, [cnt], -1, (100, 100, 255), 3)
                area = cv2.contourArea(cnt)
                if area > self.MAX_AREA:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    # print('====H--W====')
                    # print(h, w)
                    # print('====X--Y====')
                    # print(x, y)
                    if self.up > (h / w) > self.low and 600 > x > 150 and y > 0:
                        # print('SAVING')
                        cv2.imwrite(self.path + 'task4_beacon.jpg', self.cv_image)
                    cv2.rectangle(self.cv_image, (x, y), (x+w, y+h), (100, 100, 255), 2)

        elif self.target_colour == 'green':
            # GREEN
            lower_green = np.array([45, 50, 50])
            upper_green = np.array([75, 255, 255])
            mask_green = cv2.inRange(self.hsv_image, lower_green, upper_green)
            self.green_result = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask_green)

            contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(self.cv_image, [cnt], -1, (100, 255, 100), 3)
                area = cv2.contourArea(cnt)
                if area > self.MAX_AREA:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    if self.up > (h / w) > self.low and 600 > x > 150 and y > 0:
                        # print('SAVING')
                        cv2.imwrite(self.path + 'task4_beacon.jpg', self.cv_image)
                    cv2.rectangle(self.cv_image, (x, y), (x+w, y+h), (100, 255, 100), 2)

        elif self.target_colour == 'blue':
            # BLUE
            lower_blue = np.array([120, 50, 50])
            upper_blue = np.array([150, 255, 255])
            mask_blue = cv2.inRange(self.hsv_image, lower_blue, upper_blue)
            self.blue_result = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask_blue)

            contours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(self.cv_image, [cnt], -1, (255, 100, 100), 3)
                area = cv2.contourArea(cnt)
                # print(area)
                if area > self.MAX_AREA:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    # print(h, w)
                    if self.up > (h / w) > self.low and 600 > x > 150 and y > 0:
                        # print('SAVING')
                        cv2.imwrite(self.path + 'task4_beacon.jpg', self.cv_image)
                    cv2.rectangle(self.cv_image, (x, y), (x+w, y+h), (255, 100, 100), 2)
        elif self.target_colour == 'yellow':
            # YELLOW
            lower_yellow = np.array([20, 100, 100]) 
            upper_yellow = np.array([30, 255, 255]) 
            mask_yellow = cv2.inRange(self.hsv_image, lower_yellow, upper_yellow)
            self.yellow_result = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask_yellow)

            contours, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(self.cv_image, [cnt], -1, (100, 255, 255), 3)
                area = cv2.contourArea(cnt)
                if area > self.MAX_AREA:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    if self.up > (h / w) > self.low and 600 > x > 150 and y > 0:
                        # print('SAVING')
                        cv2.imwrite(self.path + 'task4_beacon.jpg', self.cv_image)
                    cv2.rectangle(self.cv_image, (x, y), (x+w, y+h), (100, 255, 255), 2)

if __name__ == '__main__':
    try:
        cd = ColourDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
