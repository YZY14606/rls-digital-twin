# Install this repository
```
git clone https://github.com/YZY14606/rls-digital-twin.github
git switch yzy_test
```

# Create the python environment for our method
`This instruction is in the push_policy/README.md `


# The environmen 'arobot' is created for launch rls_fetch_ws. I forget the details.
```
It can reference the INSTALL.md
```

# This is the steps to run our method in gazebo.
```
conda activate arobot
cd /media/yzy/2tb/nus/fork_rls/rls-digital-twin/rls_fetch_ws
roslaunch low_level_planning push_chair_env.launch
<!-- roslaunch low_level_planning rls_env.launch -->

# 新终端
conda activate arobot
roslaunch fetch_drivers whole_body_controller.launch

# 新终端
conda activate real_robot_policy
cd /media/yzy/2tb/nus/fork_rls/rls-digital-twin/push_policy
python main/main.py
```

