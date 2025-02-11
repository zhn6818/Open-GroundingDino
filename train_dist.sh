CFG=$1
DATASETS=$2
OUTPUT_DIR=$3

# Set the environment variable for CUDA
export CUDA_VISIBLE_DEVICES=1,2,3

python -m torch.distributed.launch  --nproc_per_node=3 main.py \
    --config_file ${CFG} \
    --datasets ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain_model_path ./groundingdino_swint_ogc.pth \
    --options text_encoder_type="./bert"

# python main.py \
#     --config_file ${CFG} \
#     --datasets ${DATASETS} \
#     --output_dir ${OUTPUT_DIR} \
#     --pretrain_model_path ./groundingdino_swint_ogc.pth \
#     --options text_encoder_type="./bert"
