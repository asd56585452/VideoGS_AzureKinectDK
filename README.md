# [SIGGRAPH Asia 2024] V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians
Official implementation for _V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians_.

**[Penghao Wang*](https://authoritywang.github.io/), [Zhirui Zhang*](https://github.com/zhangzhr4), [Liao Wang*](https://aoliao12138.github.io/), [Kaixin Yao](https://yaokxx.github.io/), [Siyuan Xie](https://simonxie2004.github.io/about/), [Jingyi Yu†](http://www.yu-jingyi.com/cv/), [Minye Wu†](https://wuminye.github.io/), [Lan Xu†](https://www.xu-lan.com/)**

**SIGGRAPH Asia 2024 (ACM Transactions on Graphics)**

| [Webpage](https://authoritywang.github.io/v3/) | [Paper](https://arxiv.org/pdf/2409.13648) | [Video](https://youtu.be/Z5La9AporRU?si=P95fDRxVYhXZEzYT) | [Training Code](https://github.com/AuthorityWang/VideoGS) | [SIBR Viewer Code](https://github.com/AuthorityWang/VideoGS_SIBR_viewers) | [IOS Viewer Code](https://github.com/zhangzhr4/VideoGS_IOS_viewers) |<br>
![Teaser image](assets/teaser.jpg)

This repository contains the official implementation for our paper, which adapts the original [VideoGS](https://github.com/AuthorityWang/VideoGS) framework to work with custom datasets, such as those captured with an Azure Kinect DK. This guide provides a complete workflow from data preprocessing to rendering.

## 1. Installation

### 1.1. Environment Setup
We recommend using Conda to manage the environment.

```bash
conda create -n videogs python=3.9
conda activate videogs
```

### 1.2. PyTorch
First, install PyTorch. Our code was evaluated on **CUDA 11.6** and **PyTorch 1.13.1**. Please install a version compatible with your system.

```bash
# Example installation for CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 1.3. Python Dependencies
Install the required Python packages using `requirements.txt` and then install the submodules for differential Gaussian rasterization and k-NN.

```bash
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### 1.4. External Dependencies
This project relies on several external tools for a full workflow.

#### Nerfstudio (for Camera Pose Optimization)
[Nerfstudio](https://docs.nerf.studio/en/latest/index.html) is used for robust camera pose estimation, which is crucial for high-quality results.

```bash
pip install nerfstudio
```

#### SIBR Viewer (for Rendering)
We use a custom SIBR-based viewer to visualize the trained dynamic Gaussian splats.

```bash
# Clone the viewer repository
git clone https://github.com/AuthorityWang/VideoGS_SIBR_viewers.git

# Build the viewer
cd VideoGS_SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
```

#### NeuS2 (for Keyframe Point Cloud Generation)
The original workflow uses a modified version of [NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) for keyframe point cloud generation.

```bash
# Clone to an 'external' folder
mkdir external && cd external
git clone --recursive https://github.com/AuthorityWang/NeuS2_K.git
cd NeuS2_K
cmake . -B build
cmake --build build --config RelWithDebInfo -j
```

## 2. End-to-End Workflow

This section details the full process, from raw data to a viewable result. We provide example commands using variables for clarity.

### 2.1. Initial Setup
First, define the base paths and parameters for your project.

```bash
# --- Define all required variables ---

# Activate the correct conda environment
conda activate videogs

# --- Path variables (modify for your project) ---
BASE_DIR="/path/to/your/project"
INPUT_NAME="raw_data"
PROCESSED_NAME="processed_data"
OUTPUT_NAME="training_output"

# --- Parameters (modify for your project) ---
FRAME_START=0
FRAME_END=300
GROUP_SIZE=30
INTERVAL=1
RESOLUTION=4   # 1=Full, 2=Half, 4=Quarter, 8=Eighth
SH_DEGREE=0
QP=25          # Video compression quality (lower is higher quality)
CUDA_DEVICE=0
```

### 2.2. Step 1: Preprocess the Dataset
This step converts your raw data (e.g., from the HiFi4G format) into the format required by our trainer. The `hifi4g_process.py` script handles this conversion.

The `--point3d` flag is used when your dataset provides an initial `points3D.bin` file from COLMAP.

```bash
echo "Step 1: Running preprocessing..."
cd preprocess
python hifi4g_process.py \
    --input "${BASE_DIR}/${INPUT_NAME}" \
    --output "${BASE_DIR}/${PROCESSED_NAME}" \
    --point3d
cd ..
```

### 2.3. Step 2 (Optional but Recommended): Camera Pose Optimization
Accurate camera poses are vital. If your initial poses from COLMAP are not perfect, you can refine them using Nerfstudio's optimizer.

**2.3.1. Activate Nerfstudio Environment**
This process requires the `nerfstudio` environment.

```bash
conda activate nerfstudio
```

**2.3.2. Run Nerfstudio Pose Optimization**
This involves processing the data, training a temporary `splatfacto` model with camera optimization enabled, and exporting the refined camera poses.

```bash
# Convert COLMAP model to the required format
colmap model_converter \
    --input_path "${BASE_DIR}/${INPUT_NAME}/colmap/sparse/0" \
    --output_path "${BASE_DIR}/${INPUT_NAME}/colmap/sparse/0" \
    --output_type BIN

# Process data for Nerfstudio
ns-process-data images \
    --data "${BASE_DIR}/${INPUT_NAME}/image_undistortion_white/0" \
    --output-dir "${BASE_DIR}/${INPUT_NAME}/nerfstudio_data/0" \
    --colmap-model-path "${BASE_DIR}/${INPUT_NAME}/colmap/sparse/0" \
    --verbose --skip-colmap

# Train splatfacto to optimize cameras
ns-train splatfacto \
    --data "${BASE_DIR}/${INPUT_NAME}/nerfstudio_data/0" \
    --output-dir "${BASE_DIR}/${INPUT_NAME}/splat_projects" \
    --experiment-name "fix-exmatrix" \
    --pipeline.model.camera-optimizer.mode SO3xR3 \
    --vis wandb \
    --timestamp "latest" \
    nerfstudio-data --downscale-factor 1 --eval-mode all

# Export the optimized cameras
ns-export cameras \
    --load-config "${BASE_DIR}/${INPUT_NAME}/splat_projects/fix-exmatrix/splatfacto/latest/config.yml" \
    --output-dir "${BASE_DIR}/${INPUT_NAME}/splat_projects/fix-exmatrix/splatfacto/latest"
```

**2.3.3. Apply Optimized Poses to VideoGS Dataset**
The `process_poses.py` script converts the exported Nerfstudio poses back into the format used by our project.

```bash
# Activate your project environment again
conda activate videogs

# Restore, update, and distribute the new poses
python process_poses.py restore \
    --dataparser-transforms "${BASE_DIR}/${INPUT_NAME}/splat_projects/fix-exmatrix/splatfacto/latest/dataparser_transforms.json" \
    --transforms-train "${BASE_DIR}/${INPUT_NAME}/splat_projects/fix-exmatrix/splatfacto/latest/transforms_train.json" \
    --output "${BASE_DIR}/${INPUT_NAME}/splat_projects/fix-exmatrix/splatfacto/latest/restored_poses.json"

python process_poses.py update-videogs \
    --source-transforms "${BASE_DIR}/${INPUT_NAME}/nerfstudio_data/0/transforms.json" \
    --target-transforms "${BASE_DIR}/${PROCESSED_NAME}/transforms.json" \
    --output "${BASE_DIR}/${PROCESSED_NAME}/transforms.json"

python process_poses.py distribute \
    --source-file "${BASE_DIR}/${PROCESSED_NAME}/transforms.json" \
    --target-dir "${BASE_DIR}/${PROCESSED_NAME}"
```

### 2.4. Step 3: Train the Model
Now, run the main training script `train_sequence.py`. This script trains a Gaussian Splatting model for each frame sequence.

```bash
echo "Step 3: Starting training..."
python train_sequence.py \
    --start ${FRAME_START} \
    --end ${FRAME_END} \
    --cuda ${CUDA_DEVICE} \
    --data "${BASE_DIR}/${PROCESSED_NAME}" \
    --output "${BASE_DIR}/${OUTPUT_NAME}" \
    --sh ${SH_DEGREE} \
    --interval ${INTERVAL} \
    --group_size ${GROUP_SIZE} \
    --resolution ${RESOLUTION} \
    --point3d \
    --random_background
```

### 2.5. Step 4: Compress Checkpoints into Video
The trained checkpoints are large. We provide scripts to compress them into a streamable video format.

**2.5.1. Convert Checkpoints to Feature Images**
This script precomputes features from the Gaussian point clouds and saves them as images.

```bash
echo "Step 4.1: Converting checkpoints to feature images..."
cd compress
python compress_ckpt_2_image_precompute.py \
    --frame_start ${FRAME_START} \
    --frame_end ${FRAME_END} \
    --group_size ${GROUP_SIZE} \
    --interval ${INTERVAL} \
    --ply_path "${BASE_DIR}/${OUTPUT_NAME}/checkpoint/" \
    --output_folder "${BASE_DIR}/${OUTPUT_NAME}/feature_image" \
    --sh_degree ${SH_DEGREE}
```

**2.5.2. Convert Feature Images to Video**
This script encodes the feature images into highly compressed `.mp4` video files. **Note:** This requires a Linux OS with a compatible video codec.

```bash
echo "Step 4.2: Compressing feature images to video..."
python compress_image_2_video.py \
    --frame_start ${FRAME_START} \
    --frame_end ${FRAME_END} \
    --group_size ${GROUP_SIZE} \
    --output_path "${BASE_DIR}/${OUTPUT_NAME}" \
    --qp ${QP}
cd ..
```

### 2.6. Step 5: View the Results
You can now view your volumetric video using the SIBR viewer.

**2.6.1. Host the Video Files (Optional)**
For web-based viewing, you need to host the `feature_video` directory on a web server (e.g., Nginx).

```bash
echo "Step 5.1: Copying results to web server..."
# Example: Copying to a local Nginx server directory
sudo rm -rf "/var/www/html/files/${OUTPUT_NAME}_feature_video"
sudo cp -r "${BASE_DIR}/${OUTPUT_NAME}/feature_video" "/var/www/html/files/${OUTPUT_NAME}_feature_video"
```

**2.6.2. Run the SIBR Viewer**
Launch the viewer, pointing it to the checkpoint of the first frame you want to view.

```bash
echo "Step 5.2: Launching SIBR Viewer..."
cd VideoGS_SIBR_viewers
./install/bin/SIBR_gaussianViewer_app -m "${BASE_DIR}/${OUTPUT_NAME}/checkpoint/0"
cd ..
```

## Acknowledgement
Our code is based on the original [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation. We also refer to [NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) for fast keyframe point cloud generation and [3DGStream](https://sjojok.top/3dgstream/) for the inspiration of our fast training strategy.

Thanks to [Zhehao Shen](https://github.com/moqiyinlun) for his help on dataset processing.

## Citation
If you find our work useful in your research, please consider citing our paper.
```bibtex
@article{wang2024v,
  title={V\^{} 3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians},
  author={Wang, Penghao and Zhang, Zhirui and Wang, Liao and Yao, Kaixin and Xie, Siyuan and Yu, Jingyi and Wu, Minye and Xu, Lan},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--13},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```