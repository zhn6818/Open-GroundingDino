# GPU_NUM=$1
# CFG=$2
# DATASETS=$3
# OUTPUT_DIR=$4
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# # Change ``pretrain_model_path`` to use a different pretrain. 
# # (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# # If you don't want to use any pretrained model, just ignore this parameter.

# python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
#         --output_dir ${OUTPUT_DIR} \
#         -c ${CFG} \
#         --datasets ${DATASETS}  \
#         --pretrain_model_path /path/to/groundingdino_swint_ogc.pth \
#         --options text_encoder_type=/path/to/bert-base-uncased

CFG=$1
DATASETS=$2
OUTPUT_DIR=$3

# Set the environment variable for CUDA
export CUDA_VISIBLE_DEVICES=1,2,4,5,6

python -m torch.distributed.launch  --nproc_per_node=5 main.py \
    --config_file ${CFG} \
    --datasets ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain_model_path ./groundingdino_swint_ogc.pth \
    --options text_encoder_type="../bert-base-uncased/"

# python main.py \
#     --config_file ${CFG} \
#     --datasets ${DATASETS} \
#     --output_dir ${OUTPUT_DIR} \
#     --pretrain_model_path ./groundingdino_swint_ogc.pth \
#     --options text_encoder_type="./bert"
# ./train_dist.sh "./config/cfg_odvg.py" "./config/datasets_mixed_odvg.json" "./output_coco2" 