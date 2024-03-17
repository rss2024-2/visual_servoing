#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Header
from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDrive,AckermannDriveStamped

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone", 
            self.relative_cone_callback, 1)

        self.parking_distance = .75 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.cone_distance = 0

        self.get_logger().info("Parking Controller Initialized")

        self.L1=1
        self.L=0.3
        self.velocity=1.0

        self.max_parking_error=0.15
        self.max_angle_error=np.pi/30

        self.hysteris_lb=self.parking_distance
        self.hysteris_ub=1.5 * self.parking_distance
        self.dir=-1

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        self.cone_distance=np.sqrt(self.relative_x**2+self.relative_y**2)

        if (self.cone_distance<self.hysteris_lb and self.dir==1) or (self.cone_distance>self.hysteris_ub and self.dir==-1):
            self.dir*=-1

        eta=np.arctan2(self.relative_y,self.relative_x)

        header=Header()
        header.stamp=self.get_clock().now().to_msg()
        header.frame_id="base_link"
        drive=AckermannDrive()
        drive.steering_angle=self.dir*np.arctan2(2*self.L*np.sin(eta),self.L1)
        drive.steering_angle_velocity=0.0
        drive.speed=self.dir*self.velocity
        drive.acceleration=0.0
        drive.jerk=0.0
        if np.abs(self.cone_distance-self.parking_distance)<=self.max_parking_error and np.abs(eta)<=self.max_angle_error:
            drive.speed=0.0
        stamped_msg=AckermannDriveStamped()
        stamped_msg.header=header
        stamped_msg.drive=drive

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd

        #################################

        self.drive_pub.publish(stamped_msg)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        error_msg.x_error=self.relative_x-self.parking_distance
        error_msg.y_error=self.relative_y
        error_msg.distance_error=np.sqrt(error_msg.x_error**2+error_msg.y_error**2)

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        #################################
        
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()