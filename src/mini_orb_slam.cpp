#include <ros/ros.h>

#include "frontend.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mini_orb_node");
    ros::NodeHandle nh("~");

    mini_orb_slam::Frontend frontend(nh);
    if (!frontend.init())
    {
        ROS_ERROR("Failed to initialize MiniOrbNode.");
        return 1;
    }

    frontend.run();
    return 0;
}
