#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class ReactiveFollowGap(Node):
    BUBBLE_RADIUS            = 40             #how many indices do we want to clear
    PREPROCESS_CONV_SIZE     = 5
    BEST_POINT_CONV_SIZE     = 80
    MAX_LIDAR_DIST           = 3 
    STRAIGHTS_SPEED          = 6.0
    CORNERS_SPEED            = 3.0
    STRAIGHTS_STEERING_ANGLE = 0.15708 # np.pi / 20 

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

        self.get_logger().info("Naive GapFollow Activated!!!")

        self.drive_msg = None


    def preprocess_lidar(self, ranges, data):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        crop_indices = 125

        #angular resolution of the lidar
        self.radians_per_elem = data.angle_increment

        # we won't use the LiDAR data from directly behind us
        # trying to take the forward cone values
        proc_ranges = np.array(ranges[crop_indices:-crop_indices]) #changing this- changes a lot in the sim

        # Moving average filter - sensor noise is filtered, smoothing filter here
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        # this helps with nans and inf values
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)

        return proc_ranges
    

    # input is an array like:  [1, 2, 3, 0, 0, 0, 0, 5, 6, 7, 8, 10, 29]

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """

        # mask the bubble- [1 2 3 -- -- -- -- 5 6 7 8 10 29]
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)

        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)

        # max_length will not work, i want upto 5 or 6 values only 

        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl

            
        # for slice in slices[1:]:


        return chosen_slice.start, chosen_slice.stop



    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        #can be changed- doesnt work if changed

        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i


    #this is wrong ig?? - works on an empty map though
    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle


    def calculate_drive_msg(self, scan_msg):
        ranges = scan_msg.ranges

        # Preprocess the Lidar Information
        proc_ranges = self.preprocess_lidar(ranges, scan_msg)

        # Find index of closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)- this can be modified to improve 
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        # preventing out of bounds error
        if min_index < 0: 
            min_index = 0
        if max_index >= len(proc_ranges): 
            max_index = len(proc_ranges) - 1

        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)


        # Find the best point in the gap - yes the largest 
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Get the final steering angle and speed value--?????
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED


        #Publish Drive message, speed and steering anlhr
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.drive.steering_angle = steering_angle 
        self.drive_msg.drive.speed = speed


    def lidar_callback(self, scan_msg):
        self.get_logger().info(f"Node {self.get_name()}, lidar callback is running")
        self.calculate_drive_msg(scan_msg)
        self.publisher.publish(self.drive_msg)
        



def main(args=None):
    rclpy.init(args=args)
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()