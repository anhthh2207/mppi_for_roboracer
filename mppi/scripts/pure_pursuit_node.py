#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
import pandas as pd
import yaml

import os
base_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(base_dir, 'config.yaml')
config_path = os.path.abspath(config_path)

map_dir = os.path.join(base_dir, 'waypoints')
map_dir = os.path.abspath(map_dir)
map_path = os.path.join(base_dir, 'waypoints', 'map_info.txt')
map_path = os.path.abspath(map_path)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

waypoint_path = os.path.join(map_dir, config['wpt_path'])


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self, speed=2.0, lookahead=1.5):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers
        odom_topic = '/ego_racecar/odom' if config['is_sim'] else 'pf/pose/odom'
        drive_topic = '/drive'

        if __name__ == '__main__':
            self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
            self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        else:
            self.odom_subscriber = None
            self.publisher = None
        
        self.find_localwp = False

        # visualize trajectory and target waypoint
        # self.goal_visualizer = self.create_publisher(Marker, 'pp_target', 10)
        self.goal_visualizer = self.create_publisher(Marker, 'target', 10)
        self.speed_visualizer = self.create_publisher(Marker, 'speed', 10)

        # load trajectory
        wp_path = waypoint_path

        df = pd.read_csv(wp_path, sep=';', skiprows=2)
        self.trajectory = np.array(df)[:, 1:]

        self.look_ahead = lookahead ## NOTE: increase this to reduce responsiveness, less wobbling, old: 2.0
        self.speed = speed

        self.drive_msg = None

    def calculate_drive_msg(self, pose_msg, speed_profiling=False, lookahead_profiling=False):
        # trajectory = self.trajectory[::5, :] ## NOTE: Downsample to reduce responsiveness 
        trajectory = self.trajectory 
        

        # TODO: find the current waypoint to track using methods mentioned in lecture
        vehicle_pose = self.locate_vehicle(pose_msg)
        closest_point_idx = self.find_closetpoint(trajectory, vehicle_pose)
        if lookahead_profiling:
            lookahead = trajectory[closest_point_idx, -1]
        else:
            lookahead = self.look_ahead       # fox fix loodahead
 
        # print(f'x {vehicle_pose[0]:.4f}, y {vehicle_pose[1]:.4f}, yaw {vehicle_pose[2]:.4f}, speed {vehicle_pose[3]:.4f}')
        target_pose = self.find_waypoint(trajectory, vehicle_pose, lookahead)
        self.visualize([target_pose[:2]], color=(1.0, 0.0, 0.0), duration=1)

        # TODO: transform goal point to vehicle frame of reference
        # print(f'target_pose[:2] shape {target_pose[:2].shape}')
        vehicle_coordinate = self.global2vehicle_frame(np.expand_dims(target_pose[:2], axis=0), vehicle_pose)
        vehicle_coordinate = np.squeeze(vehicle_coordinate, axis=0)
        # print(f'vehicle_coordinate shape {vehicle_coordinate.shape}')


        # TODO: calculate curvature/steering angle
        curvature = self.calculate_curvature(vehicle_coordinate)
        speed, steering_angle = self.calculate_speed_n_angle(curvature)

        ############################
        if speed_profiling:
            speed = trajectory[closest_point_idx, -1] ## NOTE: the last col
        else:
            speed = 5.0 
        # speed = np.clip(speed * 1.1, 0, 5.5)

        self.visualize([trajectory[closest_point_idx, :2]], color=(0.0, 1.0, 0.0), duration=1, target=False)


        # print(f'look ahead idx = {}')

        # speed = target_pose[-1]
        steering_angle = steering_angle + self.stanley_term(vehicle_pose, trajectory, speed, k=1.5)
        # print(f'stanley term contrition {(self.stanley_term(vehicle_pose, trajectory, speed)/steering_angle)*100:.2f}')
        # speed = 5.0
        # TODO: publish drive message, don't forget to limit the steering angle.

        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = speed


    def locate_vehicle(self, msg: Odometry):
        """Extract vehicle pose from the Odometry message.

        Returns:
            np.array: [x, y, yaw, speed]
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y

        quaternion = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])
        yaw = euler_from_quaternion(quaternion)[2]
        speed = np.linalg.norm(np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]))

        return np.array([x, y, yaw, speed])

    def find_closetpoint(self, trajectory, vehicle_pose):
        x, y, yaw, speed = vehicle_pose[0], vehicle_pose[1], vehicle_pose[2], vehicle_pose[3]

        position = np.array([[x, y]] * len(trajectory))
        diff = position - trajectory[:, :2]
        dist = np.sqrt(np.sum(diff**2, axis=1))
         
        closest_point_idx = np.argmin(dist)
        return closest_point_idx

    def find_waypoint(self, trajectory, vehicle_pose, lookahead):
        """Find the current waypoint to track.

        Args:
            trajectory (np.array): (n, 4) for [x, y, yaw, speed]
            vehicle_pose (np.array): (4,) for [x, y, yaw, speed]
            lookahead: lookahead distance in meters

        Returns:
            np.array: (4, 0) for [x, y, yaw, speed]
        """
        x, y, yaw, speed = vehicle_pose[0], vehicle_pose[1], vehicle_pose[2], vehicle_pose[3]

        # calculate euclid distances from vehicle to each waypoint
        position = np.array([[x, y]] * len(trajectory))
        diff = position - trajectory[:, :2]
        dist = np.sqrt(np.sum(diff**2, axis=1))

        local_indices = np.where(dist <= lookahead)[0]
        if len(local_indices)==0:
            self.find_localwp = False
            self.get_logger().warn("Cannot detect any local point, targeting the closest point!")
            return trajectory[np.argmin(dist), :2]
        self.find_localwp =True

        # Convert candidate waypoints to the vehicle coordinate frame
        # vehicle_coordinate = []
        # for candidate in trajectory[local_indices]:
        #     vehicle_coordinate.append(self.global2vehicle_frame(candidate[:2], vehicle_pose))
        # vehicle_coordinate = np.array(vehicle_coordinate)
        vehicle_coordinate = self.global2vehicle_frame(trajectory[local_indices, :2], vehicle_pose)
        # print(f'vehicle_coord {vehicle_coordinate.shape}')

        # Choose the candidate with maximum x (farthest in front)
        target_idx = local_indices[np.argmax(vehicle_coordinate[:,0])] ## max x
        target_pose = trajectory[target_idx]

        return target_pose

    # def global2vehicle_frame(self, global_coordinate, vehicle_pose):
    #     """Transform a global coordinate to the vehicle's frame.

    #     Args:
    #         global_coordinate (np.array): (2,) global coordinate (x, y)
    #         vehicle_pose (np.array): (4,) vehicle pose [x, y, yaw, speed]

    #     Returns:
    #         np.array: (2,) coordinate in the vehicle frame.
    #     """
    #     diff = global_coordinate - vehicle_pose[:2]
    #     dx, dy = diff[0], diff[1]
    #     yaw = vehicle_pose[2]
    #     vehicle_coordinate = np.array([np.cos(yaw) * dx + np.sin(yaw) * dy, 
    #                                    -np.sin(yaw) * dx + np.cos(yaw) * dy])
    #     return vehicle_coordinate

    def global2vehicle_frame(self, global_coordinates, vehicle_pose):
        """Transform global coordinates to the vehicle's frame using vectorized operations.

        Args:
            global_coordinates (np.array): (n, 2) array of global coordinates (x, y)
            vehicle_pose (np.array): (4,) vehicle pose [x, y, yaw, speed]

        Returns:
            np.array: (n, 2) array of coordinates in the vehicle frame.
        """
        diff = global_coordinates - vehicle_pose[:2]  # Difference in x, y (n, 2)
        dx, dy = diff[:, 0], diff[:, 1]  # Extract x and y differences

        yaw = vehicle_pose[2]
        
        # Vectorized transformation using broadcasting
        vehicle_coordinates = np.column_stack((
            np.cos(yaw) * dx + np.sin(yaw) * dy,  # x in vehicle frame
            -np.sin(yaw) * dx + np.cos(yaw) * dy  # y in vehicle frame
        ))
    
        return vehicle_coordinates
    

    def calculate_curvature(self, vehicle_coordinate):
        """Calculate curvature based on the target in the vehicle frame.

        Args:
            vehicle_coordinate (np.array): (2,) (x, y) in vehicle frame

        Returns:
            float: curvature
        """
        L_2 = vehicle_coordinate[0]**2 + vehicle_coordinate[1]**2
        curvature = 2 * vehicle_coordinate[1] / L_2
        return curvature
    
    def calculate_speed_n_angle(self, curvature):
        if self.find_localwp:
            speed = 6.0
        else:
            speed = 1.5
    
        L = 0.5 # NOTE: Vehicle wheelbase, decrease this to advoid sharp turn
        steering_angle = np.arctan(L * curvature)
        return speed, steering_angle

    def pose_callback(self, pose_msg):
        self.get_logger().info(f"Node {self.get_name()}, pose callback is running")
        self.calculate_drive_msg(pose_msg)
        self.publisher.publish(self.drive_msg)

    def visualize(self, points, color=(0.0, 1.0, 0.0), duration=0, target=True):
        """Visualize points

        Args:
            points: array-like shape (n, 2)
            color: tuple-like (R, G, B) 
        """
        marker = Marker()
        marker.header.frame_id = "map"  # or "odom", "base_link", etc.
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns, marker.id, marker.type, marker.action = "point", 0, Marker.SPHERE_LIST, Marker.ADD

        # Scale of the sphere (in meters)
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2

        # setup color and duration
        marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
        marker.color.a = 1.0  
        marker.lifetime = Duration(sec=duration)
        marker.points = [Point(x=x, y=y, z=0.0) for x, y in points]
        
        if target:
            self.goal_visualizer.publish(marker)
        else:
            self.speed_visualizer.publish(marker)

    def stanley_term(self, vehicle_pose, trajectory, speed, k=1.0):
        diff = vehicle_pose[:2] - trajectory[:, :2]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        err = np.min(dist)
        offset = np.arctan((k*err)/(speed+1e-10))
        return offset


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
