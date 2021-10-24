#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy.linalg import multi_dot
import rospy
import math
import time
from pcl_obstacle_detection.msg import ObjectPoint
from nav_msgs.msg import Odometry


class KalmanFilterTrack:
    def __init__(self):
        self.sub_ = rospy.Subscriber("/cluster",ObjectPoint,self.clusterCallback)
        self.pub_ = rospy.Publisher("/kalman_estimate",Odometry,queue_size=100)

        # Kalman filter parameters
        self.reset_flag = True
        self.num_s = 4 # The number of State veriable,
        self.num_m = 3 # The number of Observable(Measurement)state variables

        self.Fk = np.empty([self.num_s, self.num_s]) #State transtion matrix

        self.Pk_1 = np.empty([self.num_s, self.num_s]) # Covariance of a priori estimate
        self.Xk_1 = np.empty([self.num_s, 1]) # State variable of a priori estimate
        
        self.Pk = np.empty([self.num_s, self.num_s]) # Covariance of predicted estimate
        self.Xk = np.empty([self.num_s, 1]) #State variable of predicted estimate

        self.Pk_new = np.empty([self.num_s, self.num_s]) #Optimal estimate
        self.Xk_new = np.empty([self.num_s, 1])

        self.Zk = np.empty([self.num_m, 1]) # Measurement
        self.Hk = np.empty([self.num_m, self.num_s]) # State-to-Measurement

        self.Q = np.empty([self.num_s, self.num_s]) # System noise
        self.R = np.empty([self.num_m, self.num_m]) # measurement noise

        self.K = np.empty([self.num_s, self.num_m])
        self.X_new = np.empty([self.num_s, 1])
        self.P_new = np.empty([self.num_s, self.num_s])

        # init Obstacle
        self.obstacle_Ux = 0
        self.obstacle_Uy = 0
        self.obstacle_Vx = 0
        self.obstacle_Vy = 0
        self.obstacle_speed = 0

        # init time
        self.dt = 0
        self.finish_time_previous_scan = time.time()
        self.finish_time_current_scan = time.time() 


    def clusterCallback(self,msg):
        rospy.loginfo("Cluster recieved! (%f %f) r: %f",msg.center.x,msg.center.y,msg.radius)
        self.obstacle_Ux = msg.center.x
        self.obstacle_Uy = msg.center.y
        self.obstacle_speed = math.sqrt(math.pow(self.obstacle_Vx,2) + math.pow(self.obstacle_Vy,2))
        self.start_kalman()

    def init_kalman(self):

        self.Hk = np.array([[1,0,0,0],
                            [0,1,0,0]])

        self.Q = np.array([[0.0004, 0.0,    0.0,    0.0],
                           [0.0,    0.0004, 0.0,    0.0],
                           [0.0,    0.0,    0.0001, 0.0],
                           [0.0,    0.0,    0.0,    0.0001]])

        self.R = np.array([[0.0016, 0.0], 
                           [0.0,    0.0016]])

    def reset_kalman(self):

        self.obstacle_Vx = 0
        self.obstacle_Vy = 0

        self.Xk_1 =np.array([[self.obstacle_Ux],
                             [self.obstacle_Uy],
                             [self.obstacle_Vx],
                             [self.obstacle_Vy]])

        self.Pk_1 = np.array([[0.0016, 0.0,    0.0,    0.0],
                              [0.0,    0.0016, 0.0,    0.0],
                              [0.0,    0.0,    0.0004, 0.0],
                              [0.0,    0.0,    0.0,    0.0004]])
        
        self.Zk = np.array([[self.obstacle_Ux],
                            [self.obstacle_Uy]])


    def update_kalman(self):
        self.finish_time_current_scan = time.time()
        self.dt = self.finish_time_current_scan - self.finish_time_previous_scan # in sec

        self.Fk  = np.array([[1, 0,  self.dt, 0],
                            [0, 1,  0,       self.dt],
                            [0, 0,  1,       0],
                            [0, 0,  0,       1]])
        #Predictioni
        rospy.loginfo("type fk: %s xk_1:%s",type(self.Fk),type(self.Xk_1))
        #rospy.loginfo("xk:%f xk_1:%f",self.Xk.shape,self.Xk_1.shape)
        self.Xk = np.dot(self.Fk, self.Xk_1)
        self.Pk = multi_dot([self.Fk,self.Pk_1,self.Fk.T]) + self.Q
        #self.Pk = np.dot(self.Fk, np.dot(self.Pk_1, self.Fk.T)) + self.Q

        #Measurement
        self.Zk = np.array([[self.obstacle_Ux],
                            [self.obstacle_Uy]])

        #Kalman gain
        self.K = np.dot(np.dot(self.Pk,self.Hk.T),np.linalg.inv(multi_dot([self.Hk,self.Pk,self.Hk.T])+ self.R))

        #calaculate the optimal estimate
        self.X_new = self.Xk + np.dot(self.K, (self.Zk - np.dot(self.Hk,self.Xk)))
        self.P_new = self.Pk - multi_dot([self.K,self.Hk,self.Pk])

        #use X_new as the Obstacle's position and velocity
        self.obstacle_Ux = self.X_new[0]
        self.obstacle_Uy = self.X_new[1]
        self.obstacle_Vx = self.X_new[2]
        self.obstacle_Vy = self.X_new[3]
        self.obstacle_speed = math.sqrt(math.pow(self.obstacle_Vx,2) + math.pow(self.obstacle_Vy,2))

        #pub
        k_est = Odometry()
        k_est.pose.pose.position.x = self.obstacle_Ux 
        k_est.pose.pose.position.y = self.obstacle_Uy
        k_est.twist.twist.linear.x = self.obstacle_Vx
        k_est.twist.twist.linear.y = self.obstacle_Vy
        self.pub_.publish(k_est)

        rospy.loginfo("Kalman estimate: p:(%f,%f) v:(%f,%f)",self.obstacle_Ux,self.obstacle_Uy,self.obstacle_Vx,self.obstacle_Vy)

        #Feed the new optimal estimate as a prior for next prediction
        self.Xk_1 = self.X_new
        self.Pk_1 = self.P_new

        #Set current estimated position as previous position to calculate velocity
        self.finish_time_previous_scan = self.finish_time_current_scan

    def start_kalman(self):
        if self.reset_flag:
            self.reset_flag = False
            self.reset_kalman()
            self.finish_time_previous_scan = time.time()

        else:
            self.update_kalman()

    def run_it(self):
        self.init_kalman()
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("kalman_filter_node")
    app = KalmanFilterTrack()
    app.run_it()
