# Lab 9: Model Predictive Control

## YouTube video link
[Demo 1 (simulation and on real car)](https://youtu.be/FDkbGUZcvgI)

[Demo 2 (simulation)](https://youtu.be/H36UxK5p5_c)

In the demo, the green line indicates the reference and the red line is the path predicted and selected by MPC.


## Running instruction
- The reference waypoint can be found in [mpc/waypoints/slam_levine_wp.csv](mpc/waypoints/slam_levine_wp.csv)
- You should change the directory of the waypoint `wp_path` in the `waypoint_visualizer.py` and `mpc_node.py` to your own path
- We run the simulation on our SLAM-recorded levine 2nd floor map, the map can be found in [mpc/maps](mpc/maps)
