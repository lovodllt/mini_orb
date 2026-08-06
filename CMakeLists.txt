cmake_minimum_required(VERSION 3.0.2)
project(mini_orb_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  sensor_msgs
  std_msgs
  geometry_msgs
)

add_subdirectory(thirdparty/DBoW2)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o REQUIRED)

catkin_package()

include_directories(
  include
  thirdparty/DBoW2
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
)

add_executable(mini_orb_node
  src/mini_orb_slam.cpp
  src/frontend.cpp
  src/initializer.cpp
  src/tracker.cpp
  src/camera.cpp
  src/frame.cpp
  src/orb_extractor.cpp
  src/matcher.cpp
  src/pose_optimizer.cpp
  src/local_mapper.cpp
  src/map_point.cpp
  src/map.cpp
  src/bow_vocabulary.cpp
  src/keyframe_database.cpp
  src/loop_closer.cpp
)

target_link_libraries(mini_orb_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Sophus::Sophus
  DBoW2
  g2o::core
  g2o::types_sba
  g2o::types_sim3
  g2o::solver_eigen
)
