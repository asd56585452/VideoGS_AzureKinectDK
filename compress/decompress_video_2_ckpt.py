import os
import numpy as np
import plyfile
import json
import cv2

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path containing feature_video and feature_image")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=20)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    input_path = args.model_path
    
    start = args.frame_start
    end = args.frame_end
    group_size = 20
    interval = 1
    qp = "manual"
    sh_degree = args.sh_degree
    SH_N = (sh_degree + 1) * (sh_degree + 1)
    sh_number = SH_N * 3
    
    # Calculate total attributes and video count based on SH degree
    # Attributes: 3 (pos) + 3 (n) + 3 (dc) + (sh_number - 3) (rest) + 1 (op) + 3 (scale) + 4 (rot)
    # Note: sh_number includes DC (3) + Rest (45 for deg 3).
    # Total attributes = 62 for deg 3.
    total_attributes = 3 + 3 + 3 + (sh_number - 3) + 1 + 3 + 4
    num_video = total_attributes + 3
    output_path = os.path.join(input_path, "decompress_output") # Changed structure slightly or keep original? Keeping closer to original but cleaner.
    # Original: f"/data/.../decompress/qp_{qp}"
    # New: input_path/decompress_output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    feature_video_path = os.path.join(input_path, "feature_video", "png_all_25") # Note: previous script wrote to png_all_{qp}, defaulting to 25 or checking dir?
    # Let's try to auto-detect or default to 25 based on compress_image_2_video default
    if not os.path.exists(feature_video_path):
        # try to find any png_all_*
        possible_dirs = [d for d in os.listdir(os.path.join(input_path, "feature_video")) if d.startswith("png_all_")]
        if possible_dirs:
            feature_video_path = os.path.join(input_path, "feature_video", possible_dirs[0])
            print(f"Auto-detected video path: {feature_video_path}")
    
    feature_image_path = os.path.join(input_path, "feature_image")
    min_max_path = os.path.join(feature_image_path, "min_max.json")

group_idx = 0

def inverse_sigmoid(x):
    # 限制 x 範圍避免 log 報錯
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def inverse_exp(x):
    x = np.clip(x, 1e-6, None)
    return np.log(x)

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    return frames

def get_attribute():
    attribute_names = []
    attribute_names.append('x')
    attribute_names.append('y')
    attribute_names.append('z')
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    for i in range(45):
        attribute_names.append('f_rest_' + str(i))
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    return attribute_names

def denormalize_uint8(data, min_val, max_val):
    return data / 255.0 * (max_val - min_val) + min_val

def denormalize_uint16(data, min_val, max_val):
    return data / (2 ** 16 - 1) * (max_val - min_val) + min_val

def reconstruct_ply_from_images(frame, num_attributes, image_size, input_folder, min_max_info):
    reconstructed_data = np.zeros((image_size * image_size, num_attributes), dtype=float)
    
    for i in range(num_attributes):
        img_path = os.path.join(input_folder, f"{frame}_{i}.png")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        min_val = float(min_max_info[f'{frame}_{i}_min'])
        max_val = float(min_max_info[f'{frame}_{i}_max'])
        
        img_denormalized = denormalize_uint16(img, min_val, max_val)
        
        reconstructed_data[:, i] = img_denormalized.flatten()
        
    actual_num_points = min_max_info[f'{frame}_num']
    reconstructed_data = reconstructed_data[:actual_num_points]
    
    return reconstructed_data, actual_num_points

def save_ply(residual, output_file):
    n, k = residual.shape

    attribute_names = []
    attribute_names.append('x')
    attribute_names.append('y')
    attribute_names.append('z')
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    for i in range(sh_number - 3):
        attribute_names.append('f_rest_' + str(i))
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    assert k == len(attribute_names)

    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % n)
        
        for attribute_name in attribute_names:
            ply_file.write(b"property float %s\n" % attribute_name.encode())
        
        ply_file.write(b"end_header\n")
        
        for i in range(n):
            vertex_data = residual[i].astype(np.float32).tobytes()
            ply_file.write(vertex_data)

with open(min_max_path, "r") as f:
    min_max_info = json.load(f)

for frame in range(start, end, group_size * interval):
    group_start = frame
    group_end = min(frame + group_size * interval, end)
    print(group_start, group_end)

    group_video_path = os.path.join(feature_video_path, f"group{group_idx}")
    group_video_data = []
    for video_idx in range(num_video):
        video_path = os.path.join(group_video_path, f"{video_idx}.mp4")
        frames = read_video(video_path)
        group_video_data.append(frames)
    group_idx += 1

    group_frame_idx = 0
    # reconstruct a group
    for group_frame in range(group_start, group_end, interval):
        group_frame_data = np.zeros((min_max_info[f'{group_frame}_num'], num_video - 3), dtype=float)
        # position
        for att in range(3):
            # concat uint8 to uint16
            image_even = group_video_data[att * 2][group_frame_idx]
            image_odd = group_video_data[att * 2 + 1][group_frame_idx]
            # image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
            # image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)
            image_even = image_even.astype(np.uint16)
            image_odd = image_odd.astype(np.uint16)
            image = image_even + (image_odd << 8)
            # denormalize
            min_val = float(min_max_info[f'{group_frame}_{att}_min'])
            max_val = float(min_max_info[f'{group_frame}_{att}_max'])
            # print(denormalize_uint16(image, min_val, max_val).shape)
            # print(group_frame_data[:, att].shape)
            group_frame_data[:, att] = denormalize_uint16(image, min_val, max_val).flatten()[:min_max_info[f'{group_frame}_num']]
        for att in range(3, total_attributes):
            # Skip DC if handled differently? No, compress script treats all i > 2 same way.
            # But the loop logic below uses denormalize_uint8.
            # In compress script: if i > 2: normalize_uint8.
            # So DC (indices 3,4,5) are also uint8.
            # So we don't need to skip anything.
            
            image = group_video_data[att + 3][group_frame_idx]
            # denormalize
            min_val = float(min_max_info[f'{group_frame}_{att}_min'])
            max_val = float(min_max_info[f'{group_frame}_{att}_max'])
            group_frame_data[:, att] = denormalize_uint8(image, min_val, max_val).flatten()[:min_max_info[f'{group_frame}_num']]

        # --- Apply inverse transformations ---
        # 1. Reverse SH Rearrangement
        if sh_degree > 0:
            shs_data = np.zeros((group_frame_data.shape[0], sh_number), dtype=group_frame_data.dtype)
            shs_data[:, 0] = group_frame_data[:, 6]
            shs_data[:, 1] = group_frame_data[:, 7]
            shs_data[:, 2] = group_frame_data[:, 8]
            for j in range(1, SH_N):
                shs_data[:, (j - 1) + 3]             = group_frame_data[:, j * 3 + 0 + 6]
                shs_data[:, (j - 1) + SH_N + 2]      = group_frame_data[:, j * 3 + 1 + 6]
                shs_data[:, (j - 1) + 2 * SH_N + 1]  = group_frame_data[:, j * 3 + 2 + 6]
            group_frame_data[:, 6:6+sh_number] = shs_data

        # 2. Opacity: inverse sigmoid (logit)
        # In compression: opacity_data = 1 / (1 + np.exp(-current_data[:, -8]))
        opacity_data = group_frame_data[:, -8]
        opacity_data = np.clip(opacity_data, 1e-6, 1 - 1e-6)
        group_frame_data[:, -8] = np.log(opacity_data / (1 - opacity_data))

        # 3. Scale: inverse exp (log)
        # In compression: scale_data = np.exp(current_data[:, -7:-4])
        scale_data = group_frame_data[:, -7:-4]
        scale_data = np.clip(scale_data, 1e-6, None)
        group_frame_data[:, -7:-4] = np.log(scale_data)
        # -----------------------------------

        # save ply
        save_ply(group_frame_data, os.path.join(output_path, f"{group_frame}.ply"))

        group_frame_idx += 1
