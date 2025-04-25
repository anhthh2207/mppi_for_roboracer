#!/usr/bin/env python3
import time, os, sys, yaml
import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
from sensor_msgs.msg import LaserScan

from infer_env import InferEnv
from mppi_tracking import MPPI
import utils.utils as utils
from utils.jax_utils import numpify
import utils.jax_utils as jax_utils
from utils.Track import Track
from pure_pursuit_node import PurePursuit
import time as time

from mppi_node import MPPI_Node
from pure_pursuit_node import PurePursuit
from reactive_node import ReactiveFollowGap
from disparity_extender import DisparityExtender


# The following line is commented out as it is not currently in use: #########################
# jax.config.update("jax_compilation_cache_dir", "/home/nvidia/jax_cache") 



class MasterNode(Node):
    def __init__(self, is_sim=True):
        super().__init__('land_switching')
        self.pure_pursuit = PurePursuit(speed=4.0, lookahead=2.0)
        self.lane_selector = ReactiveFollowGap()

        self.mppi_node0 = MPPI_Node(index=0)
        self.mppi_node1 = MPPI_Node(index=1)
        self.mppi_node2 = MPPI_Node(index=2)
        self.mppi_node1 = MPPI_Node(index=3)

        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                depth=1,
                                reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        # create subscribers
        if is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, qos)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, qos)
        
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)
        
        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)

    def scan_callback(self, scan_msg):
        pass 

    def pose_callback(self, pose_msg):
        pass


def main(args=None):
    # return
    rclpy.init(args=args)
    mppi_node = MasterNode()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()





    