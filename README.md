# 3-DOF Planar Robot Trajectory Controller for ROS. ROBT403 Lab 3

## 1. Overview

This code implements a multi-segment trajectory controller for a 3-DOF planar robotic arm for use in the Gazebo simulation environment.

The main controller node, lab3_final.py, takes a list of Cartesian "via points" as input. It then calculates the necessary joint-space path using inverse kinematics and generates a smooth, "fly-by" trajectory using cubic polynomials. The node publishes joint commands to move the robot in Gazebo and saves plots of the resulting joint and end-effector paths.

## 2. Key Features

Inverse Kinematics (IK): Includes a full IK solver (inverse_kinematics_with_orientation) that calculates the required joint angles $(q_1, q_3, q_5)$ for a given end-effector position $(x, y)$ and orientation $(\phi)$.

Multi-Point Trajectory: Accepts a list of Cartesian via points as a ROS parameter.

Cubic Polynomial Spline: Uses cubic polynomials to generate a smooth "fly-by" trajectory that passes through all via points without stopping. It calculates optimal joint velocities at each via point to ensure continuity.

Visualization: Automatically generates two plot files upon completion:

A Cartesian-space plot (Y vs. X) showing the end-effector's path.

A Joint-space plot (Angles vs. Time) showing the motion of each joint.

Workspace Analysis: A MATLAB script (.m file) is included to calculate and visualize the robot's reachable workspace based on its link lengths and joint limits.

## 3. Robot Configuration

The controller is configured for a 3-DOF planar manipulator with the following link lengths:

Link 1 ($L_1$): 135 mm

Link 2 ($L_2$): 135 mm

End-Effector Offset ($L_3$): 46.7 mm

## 4. Dependencies

ROS Noetic

Gazebo

Python 3 (rospy, numpy, matplotlib)

MATLAB (for running the workspace analysis script)

Dynamixel Motors Git

Robot Move Git is necessary since lab3_final.py should be located in src of this package

## 5. How to Run

Launch Gazebo: In a terminal, launch your Gazebo simulation with the 3-DOF robot.

Run the Controller: In a second terminal, run the task3.py script using rosrun. The via points are passed as the _targets parameter. There is a possibility to execute a trajectory without via points, just state only one target point. 

rosrun robot_move lab3_final.py _targets:="[[0.2,0.1,1.5],[0.1,0.2,1.5],[0.0,0.25,1.5]]"


Parameter Explanation

_targets (string, required)

A YAML-style list of via points for the end-effector to pass through.

Format: _targets:="[[x1, y1, phi1], [x2, y2, phi2], ...]

x: X-position in meters.

y: Y-position in meters.

phi: End-effector orientation in radians.

The robot will start at its home position (fully extended along the X-axis) and move through each via point. Each segment of the trajectory has a fixed duration of 5 seconds.

## 6. Example Results

Running the command above will generate two plot files (e.g., traj_all_joints_joint.png and traj_all_joints_cartesian.png).

Cartesian Trajectory (Y vs. X)

Joint Trajectories (Angles vs. Time)

Workspace Plots

The included .m file generates plots of the robot's reachable workspace.

Workspace (Joint Limits: ±45°)

Workspace (Joint Limits: ±90°)



