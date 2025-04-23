# f1tenth-final-race
The code repository for the final race project of ESE 6150: F1/10 Autonomous Racing.

**Note:** To enable JAX to run on a GPU, you may need to uncomment line 23 in [mppi_node.py](mppi/scripts/mppi_node.py).

## Run MPPI on your own waypoints
- First, you need to have a CSV file that records Cartesian coordinates (I assume it has two columns: `x_position` and `y_position`). This can be obtained using a `waypoint_logger` or the drawing tool (available in the [Pure Pursuit lab](https://github.com/f1tenth-class/slam-and-pure-pursuit-team9)).
- Next, run `process_waypoint.py` in [mppi/scripts/waypoints](mppi/scripts/waypoints). This program converts the raw waypoint file into a CSV file formatted for use with MPPI.
- Update the `wpt_path` in [mppi/scripts/config.yaml](mppi/scripts/config.yaml) and [mppi/scripts/waypoints/map_info.txt](mppi/scripts/waypoints/map_info.txt) to point to the waypoint file you just generated.

## To-Do:
- [x] Try run MPPI on other map (slam_levine, available [here](https://github.com/f1tenth-class/model-predictive-control-team9/tree/main/mpc/maps)).
- [ ] Integrate waypoint drawing tool to this repo. 
- [ ] Obstacle advoidance.
- [ ] Setup MPPI on F1tenth car.
- [ ] Finetune for the race.