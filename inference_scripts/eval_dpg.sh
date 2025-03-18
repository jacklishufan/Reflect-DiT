
ANNOTATION='data/dpg/'

BASE_MODEL_PATH=ckpts/Sana_1600M_1024px_MultiLing_diffusers
CKPT=ckpts/released_weights/dit
OUTOUT_PATH=outputs
VLM_PATH=ckpts/released_weights/vlm
DATASET=dpg 

export NCCL_P2P_DISABLE=1

accelerate launch --main_process_port 30000 --num_processes 1 \
    inference_scripts/eval.py \
    --cfg 4.5 \
    --ckpt $CKPT \
    --base_model_path $BASE_MODEL_PATH \
    --name reflect_dit_dpg \
    --shift 3 \
    --ema \
    --cont \
    --num_samples 20 \
    --output_path $OUTOUT_PATH \
    -i $ANNOTATION \
    --vlm_path $VLM_PATH \
    --dataset $DATASET \
    ${@} \

# cont