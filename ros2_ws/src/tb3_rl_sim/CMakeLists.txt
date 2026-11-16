cmake_minimum_required(VERSION 3.8)
project(tb3_rl_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(gazebo_ros REQUIRED)

install(
  DIRECTORY launch worlds scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/evaluate_benchmark.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
