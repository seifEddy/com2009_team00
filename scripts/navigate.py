#!/usr/bin/env python3
import rospy
import random
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalID
from nav_msgs.msg import Odometry
from tuos_msgs.srv import SetBool, SetBoolResponse, TimedMovement, TimedMovementResponse
import tf
import numpy as np
from scipy.spatial import cKDTree

from nav_msgs.msg import OccupancyGrid
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
import subprocess
import rospkg
rp = rospkg.RosPack()

import threading

class Navigator:
    def __init__(self):
        rospy.init_node('navigator_node', anonymous=True)

        # Task 4 description
        self.TASK_TIME_LIMIT = rospy.Duration(180)
        self.AVERAGE_ROOM_SIZE = 1.5 # in meters

        self.rotating = False

        self.n_waypoints = 10
        # self.waypoints = [[random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)] for _ in range(self.n_waypoints)]
        self.waypoints = []
        self.trajectory = []
        self.free_space_list = None
        self.reduced_free_space_list = None
        self.goal = MoveBaseGoal()
        
        self.pose = [0.0, 0.0]
        self.theta = 0.0
        self.GOAL_TOLERANCE = 0.7
        rospy.Subscriber('/odom', Odometry, self.pose_callback)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        self.costmap_data = None
        self.map_data = None
        self.map_array = None
        self.laser_data = None
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        self.goals_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server(self.TASK_TIME_LIMIT)
        rospy.loginfo("Connected to move_base server")

        service_name = 'move_service'
        self.service = rospy.Service(service_name, SetBool, self.srv_callback)
        rospy.loginfo(f"the '{service_name}' Server is ready to be called...") 
        timed_service_name = 'timed_move_service'
        self.timed_service = rospy.Service(timed_service_name, TimedMovement, self.timed_srv_callback)
        rospy.loginfo(f"the '{timed_service_name}' Server is ready to be called...")

        self.update_trajectory_thread = threading.Thread(target=self.update_trajectory)
        # rospy.Timer(rospy.Duration(2.0), self.update_trajectory)
        self.check_task_time_thread = threading.Thread(target=self.check_task_time)
        self.time = None

    def scan_callback(self, msg):
        """
            LaserScan:
                float32 angle_min
                float32 angle_max
                float32 angle_increment
                float32 time_increment
                float32 scan_time
                float32 range_min
                float32 range_max
                float32[] ranges
                float32[] intensities
        """
        self.laser_data = msg
        self.is_inside_room()

    def is_inside_room(self):
        if self.laser_data:
            angle_min = self.laser_data.angle_min
            angle_max = self.laser_data.angle_max
            angle_increment = self.laser_data.angle_increment
            range_min = self.laser_data.range_min
            range_max = self.laser_data.range_max
            ranges = self.laser_data.ranges
            num_ranges = len(ranges)
            count = 0
            
            MAX_COUNT = int(0.8 * num_ranges)   # 80% of the ranges are less than the 'self.AVERAGE_ROOM_SIZE'
            for range in ranges:
                if range <= self.AVERAGE_ROOM_SIZE:
                    count += 1
            

            # print(count, num_ranges, 100 * (count / num_ranges))
            return count >= MAX_COUNT
        else:
            return False


    def timed_srv_callback(self, request):
        """
            string movement_request  # the type of movement to perform
            int32 duration           # the time (in seconds) to perform the movement for
            ---
            bool success             # a boolean response to indicate that the service has completed

        """
        """
            movement_request:
                "fwd": Move forwards. linear velocity = 0.15, angular velocity = 0.0
                "back": Move backwards. linear velocity = -0.15, angular velocity = 0.0
                "left": Turn left. linear velocity = 0.0, angular velocity = 1.0
                "right": Turn right. linear velocity = 0.0, angular velocity = -1.0

        """

        response = TimedMovementResponse()
        duration = request.duration
        
        move_type = request.movement_request
        if move_type == "fwd":
            self.move(lin_vel=0.15, ang_vel=0.0, duration=duration)
            response.success = True
        elif move_type == "back":
            self.move(lin_vel=-0.15, ang_vel=0.0, duration=duration)
            response.success = True
        elif move_type == "left":
            self.move(lin_vel=0.0, ang_vel=1.0, duration=duration)
            response.success = True
        elif move_type == "right":
            self.move(lin_vel=0.0, ang_vel=-1.0, duration=duration)
            response.success = True
        else:
            self.move(lin_vel=0.0, ang_vel=0.0, duration=duration)
            response.success = False

        return response

    def srv_callback(self, request):
        response = SetBoolResponse()
        if request.request_signal == True:
            self.execute_trajectory()
            response.response_signal = True
            response.response_message = 'Request complete'
        else:
            self.client.cancel_all_goals()
            self.cmd_vel_pub.publish(Twist())
            response.response_signal = False
            response.response_message = "Nothing happened, set request_signal to 'true' next time."
        
        return response

    def move(self, lin_vel, ang_vel, duration=0):
        
        cmd = Twist()
        cmd.linear.x = lin_vel
        cmd.angular.z = ang_vel
        timeout = rospy.Time.now() + rospy.Duration(duration)
        r = rospy.Rate(10.0)
    
        while rospy.Time.now() < timeout:
            self.cmd_vel_pub.publish(cmd)
            r.sleep()
        else:
            self.cmd_vel_pub.publish(Twist())
            rospy.logerr('TIMED MOVEMENT ENDED')


    def costmap_callback(self, data):
        self.costmap_data = data
        # print("COSTMAP RECEIVED ", self.costmap_data.info.resolution)
        
    def is_clear_around(self, x, y, radius):
        width, height = self.map_array.shape[1], self.map_array.shape[0]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if self.map_array[ny, nx] > 0:
                        return False
        return True

    def get_free_cells(self):
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        self.map_array = np.array(self.map_data.data).reshape((height, width))
        radius_in_meters = 0.26  # Distance in meters you want to be clear of obstacles
        radius = int(radius_in_meters / resolution)

        free_indices = np.argwhere(self.map_array == 0)
        self.free_indices = np.array([idx for idx in free_indices if self.is_clear_around(idx[1], idx[0], radius)])

    def map_index_to_world(self, map_info):
        resolution = map_info.resolution
        origin = map_info.origin.position

        world_coordinates = []
        for index in self.free_indices:
            world_x = index[1] * resolution + origin.x
            world_y = index[0] * resolution + origin.y
            if self.is_valid_waypoint(world_x, world_y) & self.is_not_explored_waypoint(world_x, world_y): 
                world_coordinates.append((world_x, world_y))
        return np.array(world_coordinates)
    
    def world_xy_to_map_indices(self, map_info, x, y):
        resolution = map_info.resolution
        origin = map_info.origin.position

        map_x = (x - origin.x) / resolution
        map_y = (y - origin.y) / resolution
        index = int(map_y * map_info.width + map_x)
        return int(map_x), int(map_y), index

    def get_xy_cost(self, map_info, x, y):
        return self.costmap_data.data[self.world_xy_to_map_indices(map_info, x, y)[2]]

    def map_callback(self, data):
        self.map_data = data
    
    def process_free_cells(self):
        if not self.map_data or not self.costmap_data:
            return
        self.get_free_cells()
        self.free_space_list = self.map_index_to_world(self.map_data.info)
        self.reduced_free_space_list = self.remove_close_points(self.free_space_list, threshold=0.15)

    def publish_markers(self, points):
        for i, p in enumerate(points):
            # Create a marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = ""
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set the pose of the marker
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1.0

            # Set the scale of the marker
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            # Set the color
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            # Publish the marker
            self.goals_pub.publish(marker)
    
    def publish_trajectory_markers(self, points):
        for i, p in enumerate(points):
            # Create a marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = ""
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set the pose of the marker
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1.0

            # Set the scale of the marker
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Set the color
            marker.color.a = 1.0
            marker.color.r = 0.3
            marker.color.g = 1.0
            marker.color.b = 0.0

            # Publish the marker
            self.goals_pub.publish(marker)

    def remove_close_points(self, points, threshold=0.5):
        if points is None:
            return np.empty((0, points.shape[1]), dtype=points.dtype)

        if points.shape:
            if points.size >= 2:
                tree = cKDTree(points)
                to_remove = set()

                for i in range(len(points)):
                    if i not in to_remove:
                        # Query the points within the threshold distance
                        indices = tree.query_ball_point(points[i], r=threshold)
                        # Exclude the point itself from the results
                        indices = [idx for idx in indices if idx != i]
                        # Add the found indices to the removal set
                        to_remove.update(indices)

                # Filter out the points by keeping those not in the removal set
                filtered_points = np.array([point for idx, point in enumerate(points) if idx not in to_remove])

                return filtered_points
        return None

    def sample_point(self, free_space_list):
        if free_space_list is not None:
            idx = np.random.choice(free_space_list.shape[0], size=1, replace=False)
            return free_space_list[idx], idx
        else:
            return None, None

    def pose_callback(self, msg):
        self.pose[0] = msg.pose.pose.position.x
        self.pose[1] = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def send_goal_and_wait(self, goal, duration):

        self.client.send_goal(goal)
        rospy.sleep(0.5)
        # self.client.cancel_goals_at_and_before_time(duration)
        self.client.wait_for_result()

        # Check the result
        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal reached!")
            return True
        else:
            rospy.loginfo("Goal not reached. State: %d " % self.client.get_state())
            return False

    def update_trajectory(self):
        previous_pose = []
        while not rospy.is_shutdown():
            if self.pose and not self.rotating:
                if not previous_pose:
                    previous_pose = [self.pose[0], self.pose[1]]
                    continue
                save_to_trajectory = (abs(previous_pose[0] - self.pose[0]) > 0.5) | (abs(previous_pose[1] - self.pose[1]) > 0.5) 
                if save_to_trajectory:
                    self.trajectory.append([self.pose[0], self.pose[1]])
                    # self.publish_trajectory_markers(self.trajectory)
                    previous_pose = [self.pose[0], self.pose[1]]
            rospy.sleep(0.05)

    def check_task_time(self):
        self.time = rospy.Time.now().secs
        while not rospy.is_shutdown():
            if rospy.Time.now().secs-self.time > self.TASK_TIME_LIMIT.secs:
                self.cmd_vel_pub.publish(Twist())
                rospy.sleep(0.5)
                rospy.logerr('TASK TIME LIMIT REACHED.')
                self.client.cancel_all_goals()
                rospy.sleep(0.5)
                # rospy.signal_shutdown('Finished operations')
                break
            else:
                pass
                # rospy.logerr(rospy.Time.now().secs-self.time)
            
            rospy.sleep(1.0)

    def execute_trajectory(self):
        self.update_trajectory_thread.start()
        self.check_task_time_thread.start()
        while not rospy.is_shutdown():

            self.process_free_cells()

            if self.reduced_free_space_list is None:
                continue
            p, idx = self.sample_point(self.reduced_free_space_list)

            if p is None:
                continue
            x, y = p[0, 0], p[0, 1]

            rospy.logerr('ADDING A NEW VALID WAYPOINT')

            self.waypoints.append([x, y])

            self.publish_markers(self.waypoints)
            rospy.loginfo('Waypoint x= %f, y= %f', x, y)
            self.goal.target_pose.header.frame_id = "map"
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.goal.target_pose.pose.position.x = x
            self.goal.target_pose.pose.position.y = y
            self.goal.target_pose.pose.orientation.w = 1.0
            done = self.send_goal_and_wait(goal=self.goal, duration=rospy.Duration(self.TASK_TIME_LIMIT.secs - rospy.Time.now().secs + self.time))

            self.rotate_to_explore()

            self.save_map()

    def is_valid_waypoint(self, x, y):
        accept_goal = True
        for w in self.waypoints:
            if (abs(w[0] - x) < self.GOAL_TOLERANCE and abs(w[1] - y) < self.GOAL_TOLERANCE):
                accept_goal = False
                break

        return accept_goal
    
    def is_not_explored_waypoint(self, x, y):
        accept_goal = True
        for p in self.trajectory:
            if (abs(p[0] - x) < 0.75 * self.GOAL_TOLERANCE and abs(p[1] - y) < 0.75 * self.GOAL_TOLERANCE):
                accept_goal = False
                break

        return accept_goal

    def rotate_to_explore(self):
        if self.is_inside_room():
            robot_radius_in_meters = 0.2
            robot_radius = int(robot_radius_in_meters / self.costmap_data.info.resolution)
            
            map_x, map_y, _ = self.world_xy_to_map_indices(map_info=self.map_data.info, x=self.pose[0], y=self.pose[1])
            if not self.is_clear_around(map_x, map_y, radius=robot_radius):
                rospy.loginfo('NOT SURE IF INSIDE A ROOM: Not rotating')
                return
            self.rotating = True
            rospy.loginfo('INSIDE A ROOM: Rotating to explore')
            cmd = Twist()
            cmd.angular.z = 1.2
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(5.0)
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.rotating = False
        else:
            self.rotating = False
            rospy.loginfo('NOT INSIDE A ROOM: Moving to the next goal')

    def save_map(self, filename='task4_map'):
        path = rp.get_path('com2009_team00') + '/maps/'
        command = f"cd {path} && rosrun map_server map_saver -f {filename}"
        try:
            output = subprocess.check_output(command, shell=True)
            rospy.loginfo("Map saved successfully:\n%s", output.decode())
        except subprocess.CalledProcessError as e:
            rospy.logerr("Failed to save map: %s", e)
            
if __name__ == '__main__':

    navigator = Navigator()
    rospy.spin()
