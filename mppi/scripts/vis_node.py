#!/usr/bin/env python3
import numpy as np
import os, sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
# from ros_np_multiarray import to_multiarray_f32, to_numpy_f32
# from QtMatplotlib import QtPlotter
from utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
import matplotlib.pyplot as plt
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

class QtPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        plt.ion()
        self.fig.show()

    def scatter(self, x, y, c=None, s=20, plot_num=0, live=1):
        self.ax.clear()
        scatter = self.ax.scatter(x, y, c=c, s=s)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class Visualizer_Node(Node):
    def __init__(self, index=0):
        super().__init__(f'visualizer_node{index}')
        self.obs_max_num = 108
        self.reference_max_num = 100
        self.opt_traj_max_num = 10
        self.qt_plotter = QtPlotter()
        self.opt_traj_f = None

        # publishers
        self.reference_pub = self.create_publisher(MarkerArray, f"/vis/reference{index}", 1)
        self.opt_traj_pub = self.create_publisher(MarkerArray, f"/vis/opt_traj{index}", 1)
        self.obstacle_pub = self.create_publisher(MarkerArray, f"/vis/obstacle{index}", 1)
        
        self.frenet_pose_sub = self.create_subscription(Float32MultiArray, "/frenet_pose", self.frenet_pose_callback, 1)
        self.reference_sub = self.create_subscription(Float32MultiArray, f"/reference_arr{index}", self.reference_callback, 1)
        self.opt_traj_sub = self.create_subscription(Float32MultiArray, f"/opt_traj_arr{index}", self.opt_traj_callback, 1)
        self.obstacle_sub = self.create_subscription(Float32MultiArray, "/obstacle_arr_xy", self.obstacle_callback, 1)
        self.reward_sub = self.create_subscription(Float32MultiArray, "/reward_arr", self.reward_callback, 1)

    def reward_callback(self, arr_msg):
        reward_arr = to_numpy_f32(arr_msg)
        min_values = np.min(reward_arr[:, 0])
        self.qt_plotter.scatter(-reward_arr[:, 1], reward_arr[:, 0] - min_values, c=reward_arr[:, 2], s=20, plot_num=0, live=1)
        if self.opt_traj_f is not None:
            self.qt_plotter.scatter(-self.opt_traj_f[:, 1], self.opt_traj_f[:, 0] - min_values, s=20, plot_num=1, live=1)

    def reference_callback(self, arr_msg):
        reference_arr = to_numpy_f32(arr_msg)
        reference_msg = self.waypoints_to_markerArray(reference_arr, self.reference_max_num, 0, 1, r=0.0, g=0.0, b=1.0)
        self.reference_pub.publish(reference_msg)
        
        
    def frenet_pose_callback(self, arr_msg):
        reference_arr = to_numpy_f32(arr_msg)
        
    def obstacle_callback(self, arr_msg):
        obstacle_arr = to_numpy_f32(arr_msg)
        obstacle_msg = self.waypoints_to_markerArray(obstacle_arr, self.obs_max_num, 0, 1, r=1.0, g=0.0, b=0.0)
        self.obstacle_pub.publish(obstacle_msg)

    def opt_traj_callback(self, arr_msg):
        opt_traj_arr = to_numpy_f32(arr_msg)
        # self.qt_plotter.scatter(opt_traj_arr[:, 0], opt_traj_arr[:, 1], s=20, plot_num=0, live=1)
        opt_traj_msg = self.waypoints_to_markerArray(opt_traj_arr[:10], self.opt_traj_max_num, 0, 1, r=0.0, g=1.0, b=0.0)
        self.opt_traj_f = opt_traj_arr[10:]
        self.opt_traj_pub.publish(opt_traj_msg)

    def waypoints_to_markerArray(self, waypoints, max_num, xind, yind, r=0.0, g=1.0, b=0.0, mode="line"):
        array = MarkerArray()

        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()

        if mode == "line":
            line_marker = Marker()
            line_marker.header = header
            line_marker.ns = "waypoint_line"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.1  # Line width
            line_marker.color.a = 1.0
            line_marker.color.r = r
            line_marker.color.g = g
            line_marker.color.b = b
            line_marker.pose.orientation.w = 1.0

            for i in range(min(max_num, waypoints.shape[0])):
                pt = Point()
                pt.x = float(waypoints[i, xind])
                pt.y = float(waypoints[i, yind])
                pt.z = 0.0
                line_marker.points.append(pt)

            array.markers.append(line_marker)

        elif mode == "points": 
            for i in range(max_num):
                marker = Marker()
                marker.header = header
                marker.ns = "waypoint_spheres"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD if i < waypoints.shape[0] else Marker.DELETE
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b

                if i < waypoints.shape[0]:
                    marker.pose.position.x = float(waypoints[i, xind])
                    marker.pose.position.y = float(waypoints[i, yind])
                else:
                    marker.pose.position.x = marker.pose.position.y = 0.0

                marker.pose.position.z = 0.0
                array.markers.append(marker)
                
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented in waypoints_to_markerArray.")

        return array

def main(args=None):
    rclpy.init(args=args)
    print("Track Visualizer Mode Initialized")
    lmppi_node = Visualizer_Node()
    rclpy.spin(lmppi_node)

    lmppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# python3 <vis_node.py> --ros-args -p index:=0