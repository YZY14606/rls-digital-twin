#!/bin/bash
set -euo pipefail  # 启用严格模式

run_experiment() {
    local obj_idx=$1 num_traj=$2 object_type=$3 scene_difficulty=$4
    echo "启动任务：obj_idx=$obj_idx | num_traj=$num_traj | object_type=$object_type | scene_difficulty=$scene_difficulty "
    python simulation/run.py --obj-idx "$obj_idx" --num-traj "$num_traj" --object_type "$object_type" \
    --scene_difficulty "$scene_difficulty" --num-procs 1 --save-video
}


echo "Begin test."

# params=()
# for i in $(seq 8 12); do
#     params+=("$i 100 test easy")
# done
# for param_pair in "${params[@]}"; do
#     IFS=' ' read -r obj_idx num_traj object_type scene_difficulty<<< "$param_pair"
#     run_experiment "$obj_idx" "$num_traj" "$object_type" "$scene_difficulty"
# done



# params=()
# for i in $(seq 1 7); do
#     params+=("$i 100 test easy")
# done
# for param_pair in "${params[@]}"; do
#     IFS=' ' read -r obj_idx num_traj object_type scene_difficulty<<< "$param_pair"
#     run_experiment "$obj_idx" "$num_traj" "$object_type" "$scene_difficulty"
# done


params=()
for i in $(seq 2 10); do
    params+=("$i 100 train easy")
done
for param_pair in "${params[@]}"; do
    IFS=' ' read -r obj_idx num_traj object_type scene_difficulty<<< "$param_pair"
    run_experiment "$obj_idx" "$num_traj" "$object_type" "$scene_difficulty"
done

