# COM2009 Challenge: TurtleBot3 Exploration and Color Detection

This repository contains the code and resources for the COM2009 challenge, where we used ROS and Gazebo to explore and map a workspace using the TurtleBot3. The project involves autonomous navigation and color detection within an unknown 2D environment.

## Project Overview

In this project, the TurtleBot3 explores an unknown workspace consisting of multiple rooms, each containing a uniquely colored cylinder. The main objectives are:

1. **Color Specification**: The user specifies the color of the cylinder they want to locate.
2. **Autonomous Exploration**: The TurtleBot3 starts exploring the environment to locate the specified colored cylinder.
3. **Image Capture**: Once the cylinder is located, the robot captures an image of it.
4. **Map Saving**: Simultaneously, the robot maps the environment, and the map is saved for further analysis.

## Key Features

- **ROS Integration**: Leveraged the power of ROS (Robot Operating System) for robot control and navigation.
- **Gazebo Simulation**: Utilized Gazebo for simulating the TurtleBot3 and the environment.
- **Autonomous Navigation**: Implemented algorithms for autonomous exploration and navigation within the 2D map.
- **Color Detection**: Integrated color detection capabilities to identify and locate the specified cylinder.
- **Mapping**: Real-time mapping of the environment during the exploration process.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
   In your catkin workspace src folder:

   ```bash
   git clone https://github.com/seifEddy/seifEddy.git
   cd com2009_team00
   ```

2. **Install Dependencies**:
   Ensure you have ROS and Gazebo installed. Follow the instructions [here](http://wiki.ros.org/ROS/Installation) for ROS installation and [here](http://gazebosim.org/tutorials?tut=install_ubuntu) for Gazebo. You also need to install tuos_ros package [here](https://github.com/tom-howard/tuos_ros.git). Finally, set the TURTLEBOT3_MODEL environment variable to waffle for instance ```export TURTLEBOT3_MODEL=waffle```.

3. **Run the Simulation**:
   Launch the simulation environment:
   ```bash
   roslaunch com2009_simulations task4.launch
   ```
   and start the TurtleBot3 exploration:
   ```bash
   roslaunch com2009_team00 task4.launch target_colour:<desired-color>
   ```
   start the exploration:
   ```bash
   rosservice call /move_service "request_signal: true" 
   ```

## Usage

- Specify the color of the cylinder you want the TurtleBot3 to locate.
- The robot will autonomously explore the environment, locate the cylinder, capture its image, and save the map.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


![](https://github.com/seifEddy/com2009_team00/blob/main/gifs/task4_1.gif)
