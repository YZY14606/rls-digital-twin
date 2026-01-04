#!/bin/bash
#SBATCH --job-name=object-centric-policy
#SBATCH --output=logs/push_train.log
#SBATCH --cpus-per-task=70
#SBATCH --mem=72000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuanzy@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane6

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints/push_model
mkdir -p visualizations/push_model

# Set data paths
DATA_DIR=(
    '/data/home/share/object_centric_policy/train_data_1/scene_data'
    )

# Set training parameters
NUM_WORKS=64
BATCH_SIZE=512
POINT_DIM=3
TRANSFORMATION_DIM=7
CONDITION_DIM=128
LATENT_DIM=128
HIDDEN_DIM=256
EPOCHS=200
LR=0.001
WEIGHT_DECAY=0.00001
KLD_WEIGHT=0.01
SEED=42
VAL_SPLIT=0.1
vis_interval=10000 # Don't visulize the interval

# Set logging and checkpoint parameters
CHECKPOINT_DIR="checkpoints/push_model"
LOG_LEVEL="info"
VIS_DIR="visualizations/push_model"
WANDB_PROJECT="push-prediction"
WANDB_NAME="push_cvae_$(date +%Y%m%d_%H%M%S)"


# Print training configuration
echo "=== Training Configuration ==="
echo "Data directory: ${DATA_DIR[@]}"
echo "Number of workers: $NUM_WORKS"
echo "Batch size: $BATCH_SIZE"
echo "Point dimension: $POINT_DIM"
echo "Transformation dimension: $TRANSFORMATION_DIM"
echo "Condition dimension: $CONDITION_DIM"
echo "Latent dimension: $LATENT_DIM"
echo "Hidden dimension: $HIDDEN_DIM"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Weight decay: $WEIGHT_DECAY"
echo "KLD weight: $KLD_WEIGHT"
echo "Random seed: $SEED"
echo "Validation split: $VAL_SPLIT"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Visualization directory: $VIS_DIR"
echo "W&B project: $WANDB_PROJECT"
echo "W&B run name: $WANDB_NAME"


# Run training script
echo "=== Starting Training ==="
python -u train.py \
    --data-dir ${DATA_DIR[@]} \
    --batch-size $BATCH_SIZE \
    --point-dim $POINT_DIM \
    --condition-dim $CONDITION_DIM \
    --latent-dim $LATENT_DIM \
    --hidden-dim $HIDDEN_DIM \
    --epochs $EPOCHS \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --seed $SEED \
    --validation-split $VAL_SPLIT \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log-level $LOG_LEVEL \
    --vis-dir $VIS_DIR \
    --wandb-project $WANDB_PROJECT \
    --wandb-name $WANDB_NAME \
    --log-interval 10 \
    --checkpoint-interval 10 \
    --vis-interval $vis_interval \
    --num-workers $NUM_WORKS\
    --transformation-dim $TRANSFORMATION_DIM\

echo "=== Training Complete ==="