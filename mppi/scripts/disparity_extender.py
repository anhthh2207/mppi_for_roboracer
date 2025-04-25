#!/usr/bin/env python3
"""
from shreyas-dev
"""
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import yaml


class DisparityExtender(Node):
    ROBOT_WIDTH = 0.3 # m
    PREPROCESS_CONV_SIZE = 3
    MAX_LIDAR_DIST = 3   # m
    STRAIGHTS_SPEED = 4.0
    CORNERS_SPEED = 1.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees
    DISPARITY_THRESHOLD = 0.05
    PASS_THRESHOLD = 0.5

    def __init__(self):

        super().__init__('reactive_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        if __name__ == '__main__':
            self.subscriber = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
            self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        else:
            self.subscriber = None
            self.publisher = None

        self.radians_per_elem = None

    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """
        Preprocess the LiDAR scan array with the following pipeline:
            1. Clipping the ranges viewed
            2. Setting each value to the mean over some window
            3. Rejecting high values above 3 m
        
        Args:
            ranges (np.ndarray): Ranges to be preprocessed.
        
        Returns:
            out (np.ndarray): Ranges after preprocessing.
        """

        # we won't use the LiDAR data from directly behind us
        # trying to take the forward cone values
        proc_ranges = np.array(ranges[200:-200])

        # Moving average filter - sensor noise is filtered, smoothing filter here
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        # this helps with nans and inf values
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)

        return proc_ranges

    def get_disparities(self, ranges: np.ndarray) -> np.ndarray:
        """
        Get the locations of the disparities in the range information
        
        Args:
            ranges (np.ndarray): Range data.
        
        Returns:
            out (np.ndarray): Indices in the range data at which there are disparities.
        """
        return np.argwhere(np.abs(np.diff(ranges)) > self.DISPARITY_THRESHOLD)
    
    def extend_disparities(self, ranges: np.ndarray) -> np.ndarray:
        """
        Extend the disparities in the range data

        Args:
            ranges (np.ndarray): Preprocessed range data.
        
        Returns:
            out (np.ndarray): Virtual range data on which to find gap.
        """
        disparities = self.get_disparities(ranges)
        for d in disparities:
            # arc = np.atan(0.9 * self.ROBOT_WIDTH / np.minimum(ranges[d-1], ranges[d+1]))
            arc = np.arctan(0.9 * self.ROBOT_WIDTH / np.minimum(ranges[d-1], ranges[d+1]))
            min_idx = int(np.maximum(0, d - (arc / self.radians_per_elem)))
            max_idx = int(np.minimum(d + (arc / self.radians_per_elem), len(ranges)))
            ranges[min_idx:max_idx] = ranges[d]
        
        return ranges

    def get_angle(self, range_index: int, range_len: int) -> float:
        """
        Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        
        Args:
            range_index (int): Index of the range data of the target angle.
            range_len (int): Length of the range data array.
        
        Returns:
            out (float): Steering angle of the vehicle.
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle
    
    def finetune_steering_angle(self, ranges, steering_angle, angle_min):
        """
        Adjusts the vehicle's steering angle based on LiDAR range data (similar to finetune_velocity()). 
        If any point is below a safe distance on the side of the car in the direction the car is turning, 
        stop turning and just keep going straight. Make a sharp turn when approaching an obstacle.

        Args:
            ranges (np.array): LiDAR range data.
            steering_angle (float): current velocity of the vehicle.

        Returns:
            float: Adjusted steering_angle based look-ahead distance.
        """
        opening = np.radians(30)
        i = int((-opening-angle_min)/self.radians_per_elem)
        j = int((opening-angle_min)/self.radians_per_elem)
        look_ahead = np.mean(ranges[i:j])
        look_ahead = look_ahead - 0.4
        look_ahead = max(look_ahead, 0.5)

        left_dist = ranges[int((np.pi/2-angle_min)/self.radians_per_elem)-10]
        right_dist = ranges[int((-np.pi/2-angle_min)/self.radians_per_elem)+10]
        if left_dist < 0.2:
            steering_angle = min(steering_angle, 0.0)
        if right_dist < 0.2:
            steering_angle = max(steering_angle, 0.0)

        if look_ahead < 1.5:
            steering_angle = steering_angle * 20
        
        return steering_angle

    def calculate_drive_msg(self, scan_msg: LaserScan):
        self.get_logger().info(f"Node {self.get_name()}, lidar callback is running")
        # Set angular resolution of the LiDAR
        if self.radians_per_elem is None:
            self.radians_per_elem = scan_msg.angle_increment        

        ranges = scan_msg.ranges

        # Preprocess the Lidar Information
        proc_ranges = self.preprocess_lidar(ranges)

        # Extend disparities
        disp_ranges = self.extend_disparities(proc_ranges)

        # Get the final steering angle and speed value
        steering_angle = self.get_angle(np.argmax(disp_ranges), len(proc_ranges))
        steering_angle = self.finetune_steering_angle(ranges, steering_angle, scan_msg.angle_min)
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED

        # Publish Drive message, speed and steering angle
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = speed
    
    def lidar_callback(self, scan_msg):
        self.get_logger().info(f"Node {self.get_name()}, lidar callback is running")
        self.calculate_drive_msg(scan_msg)
        self.publisher.publish(self.drive_msg)



def main(args=None):
    rclpy.init(args=args)
    print("Gapfollow Initialized")
    reactive_node = DisparityExtender()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
