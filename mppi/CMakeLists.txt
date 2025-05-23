cmake_minimum_required(VERSION 3.5)
project(mppi)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Include Cpp "include" directory
include_directories(include)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

# Create Cpp executable
add_executable(mppi_node src/mppi_node.cpp)

ament_target_dependencies(mppi_node 
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros
)

# Install Cpp executables
install(TARGETS
  mppi_node
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/mppi_node.py
  scripts/vis_node.py
  scripts/infer_env.py
  scripts/mppi_tracking.py
  scripts/config.yaml
  DESTINATION lib/${PROJECT_NAME}
)

# Install utils directory
install(DIRECTORY
  scripts/utils/
  DESTINATION lib/${PROJECT_NAME}/utils
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install dynamics_models directory
install(DIRECTORY
  scripts/dynamics_models/
  DESTINATION lib/${PROJECT_NAME}/dynamics_models
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install waypoints directory
install(DIRECTORY
  scripts/waypoints/
  DESTINATION lib/${PROJECT_NAME}/waypoints
  FILES_MATCHING
  PATTERN "*"
  PATTERN "__pycache__" EXCLUDE
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()
