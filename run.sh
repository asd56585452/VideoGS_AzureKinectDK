#!/bin/bash
set -e

DATASETS_NAME_1="HiFi4G_Dataset"
DATASETS_NAME_2="4K_Actor1_Greeting"
CUDA_ID=0
SH_DEGREE=3
INTERVAL=1
GROUP_SIZE=20
RESOLUTION=4
QP=0

DATA_DIR="/media/cgvmis418/2TBSSD/VideoGS/datasets/${DATASETS_NAME_1}/${DATASETS_NAME_2}_preprocess"
OUTPUT_DIR="/media/cgvmis418/2TBSSD/VideoGS/output/${DATASETS_NAME_2}"

# Automatically determine the total number of frames
# Frame folders are named 0, 1, 2, ..., N in DATA_DIR
FRAME_START=0
# Use -mindepth 1 to prevent counting the parent DATA_DIR (which starts with '4' in 4K_Actor1...)
FRAME_END=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | wc -l)
echo "Detected $FRAME_END total frames in $DATA_DIR"

python train_sequence.py --start $FRAME_START --end $FRAME_END --cuda $CUDA_ID --data "$DATA_DIR" --output "$OUTPUT_DIR" --sh $SH_DEGREE --interval $INTERVAL --group_size $GROUP_SIZE --resolution $RESOLUTION --eval
python compress/compress_ckpt_2_image_precompute.py --frame_start $FRAME_START --frame_end $FRAME_END --group_size $GROUP_SIZE --interval $INTERVAL --ply_path "$OUTPUT_DIR/checkpoint/" --output_folder "$OUTPUT_DIR/feature_image" --sh_degree $SH_DEGREE
python compress/compress_image_2_video.py --frame_start $FRAME_START --frame_end $FRAME_END --group_size $GROUP_SIZE --output_path "$OUTPUT_DIR" --qp $QP
python compress/decompress_video_2_ckpt.py   --model_path "$OUTPUT_DIR" --sh $SH_DEGREE --frame_start $FRAME_START --frame_end $FRAME_END
python evaluate_compression.py   --source_path "$DATA_DIR"   --model_path "$OUTPUT_DIR"   --ply_path "$OUTPUT_DIR/decompress_output" --frame_start $FRAME_START --frame_end $FRAME_END  --sh $SH_DEGREE  --group_size $GROUP_SIZE --resolution $RESOLUTION --eval
python metrics.py   --model_path "$OUTPUT_DIR"   --renders_folder decompressed_renders
