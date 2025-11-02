#!/usr/bin/env python3
# 複製 point_cloud.ply 範例腳本
# (已修改：增加 Y 軸反轉 + 繞 X 軸旋轉 -10 度)

import os
import sys
import re
import shutil
import glob
import numpy as np
from plyfile import PlyData, PlyElement

def process_and_save_ply(src_path, dst_path, x_rotation_degrees=-10.0):
    """
    讀取 .ply 檔案，執行以下轉換，然後儲存：
    1. Y 軸鏡像 (y = -y, rot_1 = -rot_1, rot_3 = -rot_3)
    2. 繞 X 軸旋轉 (例如 -10 度)
    """
    try:
        # 讀取 PLY 檔案
        plydata = PlyData.read(src_path)
        
        if 'vertex' not in plydata:
            print(f"  [警告] 找不到 'vertex' 元素在 {src_path}，將直接複製。")
            shutil.copy(src_path, dst_path)
            return

        vertex_element = plydata['vertex']
        data = vertex_element.data
        
        # 檢查必要的屬性是否存在
        required_props = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        if not all(prop in data.dtype.names for prop in required_props):
            print(f"  [警告] 缺少必要的屬性 (x,y,z,rot_0-3) 在 {src_path}，將直接複製。")
            shutil.copy(src_path, dst_path)
            return

        # --- 1. Y 軸鏡像 (Mirror) ---
        
        # 1a. 位置 Y 軸鏡像
        pos_mirror = np.stack([
            data['x'],
            data['y'] * -1.0,
            data['z']
        ], axis=1) # Shape (N, 3)

        # 1b. 旋轉 Y 軸鏡像 (q_mirror = (w, -x, y, -z))
        q_mirror = np.stack([
            data['rot_0'],
            data['rot_1'] * -1.0,
            data['rot_2'],
            data['rot_3'] * -1.0
        ], axis=1) # Shape (N, 4)

        # --- 2. 繞 X 軸旋轉 -10 度 ---
        
        angle_rad = np.radians(x_rotation_degrees)
        
        # 2a. 位置旋轉 (使用旋轉矩陣)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 繞 X 軸的旋轉矩陣 R_x
        R_x = np.array([
            [1.0, 0.0,   0.0  ],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a,  cos_a]
        ])
        
        # 將 R_x 應用於鏡像後的位置
        # (N, 3) @ (3, 3) -> (N, 3)
        pos_final = pos_mirror @ R_x.T 

        # 2b. 旋轉 (使用四元數乘法: q_final = q_rot * q_mirror)
        
        # 繞 X 軸旋轉的四元數 q_rot = (cos(a/2), sin(a/2), 0, 0)
        half_angle = angle_rad / 2.0
        q_rot_w = np.cos(half_angle)
        q_rot_x = np.sin(half_angle)
        # q_rot_y = 0.0
        # q_rot_z = 0.0

        # 向量化四元數乘法 (q1 = q_rot, q2 = q_mirror)
        w1, x1 = q_rot_w, q_rot_x
        w2, x2, y2, z2 = q_mirror[:, 0], q_mirror[:, 1], q_mirror[:, 2], q_mirror[:, 3]

        # q_final_w = w1*w2 - x1*x2 - y1*y2 - z1*z2  (y1=0, z1=0)
        q_final_w = w1*w2 - x1*x2
        # q_final_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        q_final_x = w1*x2 + x1*w2
        # q_final_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        q_final_y = w1*y2 + x1*z2
        # q_final_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        q_final_z = w1*z2 - x1*y2

        # --- 3. 更新 PLY data ---
        
        # 更新位置
        data['x'] = pos_final[:, 0]
        data['y'] = pos_final[:, 1]
        data['z'] = pos_final[:, 2]
        
        # 更新旋轉
        data['rot_0'] = q_final_w
        data['rot_1'] = q_final_x
        data['rot_2'] = q_final_y
        data['rot_3'] = q_final_z

        # 寫入修改後的 PLY 檔案 (確保是 binary_little_endian 格式)
        plydata.write(dst_path)
        
    except Exception as e:
        print(f"  [錯誤] 處理 {src_path} 時發生錯誤: {e}，將直接複製。")
        shutil.copy(src_path, dst_path)

# --- 以下是您原始腳本的主要邏輯 ---

BASE = sys.argv[1] if len(sys.argv) > 1 else "/home/cgvmis418/VideoGS_AzureKinectDK/datasets/Group4_process_ubuntu_output_90_frame_5_GROUP_SIZE_real_full_iter_1_scale"
DEST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(BASE),"supersplate")
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 90

pattern = re.compile(r".*/checkpoint/(\d+)/point_cloud/iteration_(\d+)/point_cloud\.ply$")
paths = glob.glob(os.path.join(BASE, "checkpoint", "*", "point_cloud", "iteration_*", "point_cloud.ply"))

# ...existing code...
# 變更：對每個 checkpoint 取最大 iteration 的檔案
max_per_ck = {}
for p in paths:
    m = pattern.match(p.replace("\\\\", "/"))
    if m:
        ck = int(m.group(1))
        it = int(m.group(2))
        prev = max_per_ck.get(ck)
        if prev is None or it > prev[0]:
            max_per_ck[ck] = (it, p)

# 轉成 (ck, it, path) 並依 checkpoint 編號排序
items = sorted(((ck, it, p) for ck, (it, p) in max_per_ck.items()))

# 複製/編號
os.makedirs(DEST, exist_ok=True)
count = 0
for i, (ck, it, p) in enumerate(items):
    if i >= LIMIT:
        break
    
    src_path = p
    dst_path = os.path.join(DEST, f"{count:06d}.ply")
    print(f"[{i+1}/{len(items)}] CK={ck} IT={it} {src_path} -> {dst_path}")
    
    # *** 變更點：使用我們的新函式進行處理和複製 (預設 -10 度) ***
    process_and_save_ply(src_path, dst_path, x_rotation_degrees=-10.0) 
    
    count += 1

print(f"處理並複製 {count} 個檔案到 {DEST}")