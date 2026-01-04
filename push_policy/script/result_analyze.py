from pathlib import Path
import os
import numpy as np
import json


def main():
    json_traj_dir = '/home/yzy/nus/git_rop/Object-centric-Manipulation-Policy/Baseline_pushnet/push_record/test8'
    # Get all trajectories
    root_dir = Path(json_traj_dir)
    # 使用 pathlib 的 rglob 方法查找所有 .npy 文件
    traj_paths = list(root_dir.rglob('**/traj_*/'))
    traj_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))

    # Get the times of error or success
    motion_plan_error = 0
    object_falled = 0
    path_plan_error = 0
    RRT_conduct_error = 0
    code_error = 0
    push_length_error = 0
    success_times = 0
    # all_success_steps
    all_success_steps = 0
    # All times
    traj_number = len(traj_paths)
    for traj_path in traj_paths:
        json_path = os.path.join(traj_path,'log.json')
        # 打开并读取 JSON 文件
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["success_state"] == "True" or data["success_state"] == "tensor([True])":
            success_times += 1
            all_success_steps += int(data['push_steps'])
            continue
        if data["fall_state"] == 'Fall':
            print('In this trajectory, the object falled. The path is:',traj_path)
            print('This trajectory finishing state:',data)
            object_falled += 1
            continue
        if data["fall_state"] == 'Motion planner error':
            motion_plan_error += 1
            print('In this trajectory, motion planner error. The path is:',traj_path)
            continue
        if data["fall_state"] == 'Conduct RRT planner error':
            RRT_conduct_error += 1
            continue
        if data["fall_state"] == 'Code error':
            code_error += 1
            continue
        if data["path_plan_state"] == 'Error':
            path_plan_error += 1
            print('In this trajectory, path plan error. The path is:',traj_path)
            continue
        if data['push_length_error'] == 'Error':
            push_length_error += 1
            print('In this trajectory, push length error. The path is:',traj_path) 
    

    # Calculate the number of valid trajectories
    valid_traj_num = traj_number - RRT_conduct_error
    # Success rate
    success_rate = success_times / valid_traj_num
    # Success average steps
    if success_times > 0:
        success_average_steps = all_success_steps / success_times
    else:
        success_average_steps = 0

    print(f"The trajectory number is {traj_number}. Valid trajectory number is {valid_traj_num}.")
    print(f"The object falled number is {object_falled}.")
    print(f"The motion planner error number is {motion_plan_error}.")
    print(f"The conduct RRT planner error number is {RRT_conduct_error}.")
    print(f"The code error number is {code_error}.")
    print(f"The path plan error number is {path_plan_error}.")
    print(f"The push length error number is {push_length_error}.")
    print(f"The success number is {success_times}.")
    print(f"The success rate is {success_rate}.")
    print(f"The success average steps are {success_average_steps}.")



if __name__ == "__main__":
    main()