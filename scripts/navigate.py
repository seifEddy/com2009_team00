#!/usr/bin/env python3
import rospy
import random
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalID
from nav_msgs.msg import Odometry
import tf
import numpy as np
from scipy.spatial import cKDTree

from nav_msgs.msg import OccupancyGrid
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from functools import lru_cache

class Navigator:
    def __init__(self):
        rospy.init_node('navigator_node', anonymous=True)
        self.n_waypoints = 10
        # self.waypoints = [[random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)] for _ in range(self.n_waypoints)]
        self.waypoints = []
        self.free_space_list = None
        self.reduced_free_space_list = None
        self.goal = MoveBaseGoal()
        
        self.pose = [0.0, 0.0]
        self.theta = 0.0
        self.GOAL_TOLERANCE = 0.7
        rospy.Subscriber('/odom', Odometry, self.pose_callback)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        self.client.cancel_goals_at_and_before_time(rospy.Duration(60.0))
        rospy.loginfo("Connected to move_base server")
        self.costmap_data = None
        self.map_data = None
        self.map_array = None
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        self.goals_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    def costmap_callback(self, data):
        self.costmap_data = data
        # print("COSTMAP RECEIVED ", self.costmap_data.info.resolution)

    @lru_cache(maxsize=None)
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
        radius_in_meters = 0.3  # Distance in meters you want to be clear of obstacles
        radius = int(radius_in_meters / resolution)

        free_indices = np.argwhere(self.map_array == 0)
        self.free_indices = np.array([idx for idx in free_indices if self.is_clear_around(idx[1], idx[0], radius)])
        # self.free_indices = free_indices
        # print(np.unique(map_array))
        # print(np.min(map_array))
    
    def get_zero_cost_cells(self):
        width = self.costmap_data.info.width
        height = self.costmap_data.info.height
        costmap_array = np.array(self.costmap_data.data).reshape((height, width))

        self.free_indices = np.argwhere(costmap_array == 0)
        # print(np.unique(costmap_array))

    def get_free_zero_cost_cells(self):
        width = self.map_data.info.width
        height = self.map_data.info.height
        map_array = np.array(self.map_data.data).reshape((height, width))

        costmap_array = np.array(self.costmap_data.data).reshape((height, width))

        self.free_indices = np.argwhere((map_array < 99) & (map_array > -1) & (costmap_array >= 0) & (costmap_array < 3))
        free_indices = np.argwhere((map_array == 0))
        print(self.free_indices.shape[0], free_indices.shape[0])

    def shuffle_indices(self):
        np.random.shuffle(self.free_indices)

    def map_index_to_world(self, map_info):
        resolution = map_info.resolution
        origin = map_info.origin.position

        world_coordinates = []
        for index in self.free_indices:
            world_x = index[1] * resolution + origin.x
            world_y = index[0] * resolution + origin.y
            world_coordinates.append((world_x, world_y))
        return np.array(world_coordinates)

    def map_callback(self, data):
        self.map_data = data
    
    def process_free_cells(self):
        if not self.map_data or not self.costmap_data:
            return
        # self.get_free_zero_cost_cells()
        self.get_free_cells()
        self.free_space_list = self.map_index_to_world(self.map_data.info)
        self.reduced_free_space_list = self.remove_close_points(self.free_space_list, threshold=0.15)
        # self.reduced_free_space_list = self.free_space_list
        # self.publish_markers(self.free_space_list)
        # self.plot_points()

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
        
    def reduce_points_with_dbscan(self, eps, min_samples):
        coords = self.free_space_list

        # Applying DBSCAN to reduce points
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        unique_labels = set(clustering.labels_)
        
        # Extracting the centroid of each cluster
        self.centroids = []

        for k in unique_labels:
            class_member_mask = (clustering.labels_ == k)
            xy = coords[class_member_mask]
            cluster_center = xy.mean(axis=0)
            self.centroids.append(tuple(cluster_center))

        # Update the free_space_list with reduced points
        self.reduced_free_space_list = np.array(self.centroids)
        # print("Reduced Goal Positions:", self.free_space_list)
        # return clustered_points

    def plot_points(self):
        # Extract coordinates for plotting
        free_space_coords = self.free_space_list
        # centroids = np.array(self.centroids)
        centroids = self.reduced_free_space_list

        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.scatter(free_space_coords[:, 0], free_space_coords[:, 1], c='blue', label='Free Space')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.title('Map Free Space and Centroids')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)

    def remove_close_points(self, points, threshold=0.5):
        if points is None:
            return np.empty((0, points.shape[1]), dtype=points.dtype)

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

    def sample_point(self, free_space_list):
        if free_space_list is not None:
            idx = np.random.choice(free_space_list.shape[0], size=1)
            # if self.is_zero_cost_space(free_space_list[idx, 0], free_space_list[idx, 0]):
                # return free_space_list[idx], idx
            # else:
                # return None, None
            return free_space_list[idx], idx
        else:
            return None, None
        
    def is_zero_cost_space(self, x, y):
        if self.costmap_data is None:
            return False

        mx = int((x - self.costmap_data.info.origin.position.x) / self.costmap_data.info.resolution)
        my = int((y - self.costmap_data.info.origin.position.y) / self.costmap_data.info.resolution)

        if mx < 0 or my < 0 or mx >= self.costmap_data.info.width or my >= self.costmap_data.info.height:
            return False

        index = my * self.costmap_data.info.width + mx
        # print("COST: ", self.costmap_data.data[index])
        return self.costmap_data.data[index] == 0

    def is_free_space(self, x, y):
        if self.map_data is None:
            return False

        mx = int((x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        my = int((y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)

        if mx < 0 or my < 0 or mx >= self.map_data.info.width or my >= self.map_data.info.height:
            return False

        index = my * self.map_data.info.width + mx
        # print("COST: ", self.map_data.data[index])
        return self.map_data.data[index] == 0


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

    def send_goal_and_wait(self, goal):

        self.client.send_goal(goal)
        self.client.wait_for_result()

        # Check the result
        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal reached!")
        else:
            rospy.loginfo("Goal not reached. State: %d" % self.client.get_state())

    def execute_trajectory(self):
        
        while not rospy.is_shutdown():
            # x = self.pose[0] + random.uniform(-2.0, 2.0)
            # y = self.pose[1] + random.uniform(-2.0, 2.0)
            # free_space_list = self.remove_close_points(self.free_space_list, threshold=0.2)
            self.process_free_cells()
            # free_space_list = self.free_space_list
            # print(free_space_list.shape)
            # if free_space_list.shape[0] < 2:
            if self.reduced_free_space_list is None:
                continue
            p, idx = self.sample_point(self.reduced_free_space_list)
            if p is None:
                continue
            print(p)
            x, y = p[0, 0], p[0, 1]
            # print(x, y)
            accept_goal = False
            # print(self.is_zero_cost_space(x, y))
            # waypoints_tmp = self.waypoints
            # if self.is_zero_cost_space(x, y):

            if not self.waypoints:
                self.waypoints.append([x, y])
                accept_goal = True
            else:
                accept_goal = self.is_valid_waypoint(x, y)
                # for w in self.waypoints:
                    # if abs(w[0] - x) < self.GOAL_TOLERANCE and abs(w[1] - y) < self.GOAL_TOLERANCE:
                        # accept_goal = False

            if not accept_goal:
                # print(x, y)
                # print(self.waypoints)
                continue
            rospy.logerr('ADDING A NEW VALID WAYPOINT')

            self.waypoints.append([x, y])

            self.publish_markers(self.waypoints)
            rospy.loginfo('Waypoint x= %f, y= %f', x, y)
            self.goal.target_pose.header.frame_id = "map"
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.goal.target_pose.pose.position.x = x
            self.goal.target_pose.pose.position.y = y
            self.goal.target_pose.pose.orientation.w = 1.0
            self.send_goal_and_wait(goal=self.goal)
            self.rotate_to_explore()

            # pop the element
            # self.free_space_list = np.delete(self.free_space_list, idx, axis=0)

    def is_valid_waypoint(self, x, y):
        accept_goal = True
        for w in self.waypoints:
            if (abs(w[0] - x) < self.GOAL_TOLERANCE and abs(w[1] - y) < self.GOAL_TOLERANCE) or (w[0] == x and w[1] == y):
                accept_goal = False

        return accept_goal

    def rotate_to_explore(self):
        rospy.loginfo('Rotating to explore')
        cmd = Twist()
        cmd.angular.z = 1.0
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(5.0)
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
            
navigator = Navigator()
navigator.execute_trajectory()
# rospy.spin()
# for w in navigator.waypoints:
#     print("x= %.3f, y= %.3f"%(w[0], w[1]))
