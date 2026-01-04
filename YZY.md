conda activate arobot
cd /media/yzy/2tb/nus/new_virtual_rls/rls-digital-twin-vamp/rls_fetch_ws
roslaunch low_level_planning push_chair_env.launch
<!-- roslaunch low_level_planning rls_env.launch -->

# 新终端
conda activate arobot
roslaunch fetch_drivers whole_body_controller.launch


# 新终端
conda activate real_robot_policy
cd /media/yzy/2tb/nus/new_virtual_rls/rls-digital-twin-vamp/push_policy
python main/main.py