import json
import numpy as np
import os
import argparse
import shutil
from pathlib import Path
from PIL import Image

# --- 通用函式 ---

def load_json(filepath):
    """載入 JSON 檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath, indent=4):
    """儲存 JSON 檔案"""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

# --- 步驟 1: 還原姿態 (來自 restore_script.py) ---

def restore_extrinsics(dataparser_transforms_path, transforms_train_path, output_path):
    """
    還原原始相機外部參數。
    """
    print("--- 步驟 1: 開始還原原始相機姿態 ---")
    dp_data = load_json(dataparser_transforms_path)
    T_dp_3x4 = np.array(dp_data["transform"])
    s_dp = dp_data["scale"]

    T_dp_4x4 = np.eye(4)
    T_dp_4x4[:3, :] = T_dp_3x4
    try:
        inv_T_dp_4x4 = np.linalg.inv(T_dp_4x4)
    except np.linalg.LinAlgError:
        print("錯誤：dataparser_transforms.json 中的轉換矩陣是奇異的，無法計算逆矩陣。")
        return

    train_data = load_json(transforms_train_path)
    
    restored_frames = []
    for frame_data in train_data:
        P_ns_3x4 = np.array(frame_data["transform"])
        
        P_ns_4x4 = np.eye(4)
        P_ns_4x4[:3, :] = P_ns_3x4

        P_intermediate_4x4 = np.copy(P_ns_4x4)
        P_intermediate_4x4[:3, 3] /= s_dp
        
        P_original_4x4 = inv_T_dp_4x4 @ P_intermediate_4x4
        P_original_3x4 = P_original_4x4[:3, :]
        
        restored_frame = {
            "file_path": frame_data.get("file_path", ""),
            "transform": P_original_3x4.tolist()
        }
        restored_frames.append(restored_frame)

    save_json(restored_frames, output_path)
    print(f"✅ 還原後的相機參數已儲存到: {output_path}\n")

# --- 步驟 2: 更新 Nerfstudio transforms.json (來自 update_script.py) ---

def convert_3x4_list_to_4x4_list(matrix_3x4_list):
    """將 3x4 的列表矩陣轉換為 4x4 的列表齊次矩陣"""
    if not (isinstance(matrix_3x4_list, list) and len(matrix_3x4_list) == 3 and
            all(isinstance(row, list) and len(row) == 4 for row in matrix_3x4_list)):
        raise ValueError("輸入必須是一個 3x4 的列表矩陣。")
    
    matrix_4x4_list = [row[:] for row in matrix_3x4_list]
    matrix_4x4_list.append([0.0, 0.0, 0.0, 1.0])
    return matrix_4x4_list

def update_nerfstudio_transforms(restored_poses_path, target_transforms_path, output_path):
    """
    使用還原後的姿態更新目標 transforms.json。
    """
    print("--- 步驟 2: 開始更新 Nerfstudio transforms.json ---")
    restored_data_list = load_json(restored_poses_path)
    restored_poses_map = {os.path.basename(frame_info["file_path"]): frame_info["transform"]
                          for frame_info in restored_data_list if "file_path" in frame_info and "transform" in frame_info}

    target_data = load_json(target_transforms_path)
    
    updated_frames_count = 0
    not_found_frames_log = []
    
    for frame_in_target in target_data.get("frames", []):
        target_base_filename = os.path.basename(frame_in_target.get("file_path", ""))
        if target_base_filename in restored_poses_map:
            try:
                pose_3x4_list = restored_poses_map[target_base_filename]
                pose_4x4_list = convert_3x4_list_to_4x4_list(pose_3x4_list)
                frame_in_target["transform_matrix"] = pose_4x4_list
                updated_frames_count += 1
            except ValueError as e:
                print(f"警告：處理影格 '{target_base_filename}' 時出錯：{e}。已跳過。")
        else:
            not_found_frames_log.append(target_base_filename)

    if not_found_frames_log:
        print(f"警告：在還原的姿態檔案中找不到以下 {len(not_found_frames_log)} 個影像的對應姿態：{', '.join(not_found_frames_log)}")

    identity_3x4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    target_data["applied_transform"] = identity_3x4

    save_json(target_data, output_path)
    print(f"已更新 {updated_frames_count} 個影格。")
    print(f"✅ 更新後的 Nerfstudio transforms 檔案已儲存到: {output_path}\n")

# --- 步驟 3: 更新 VideoGS transforms.json (來自 update_videoGS_script.py) ---

def update_videogs_transforms(source_transforms_path, target_videoGS_path, output_path):
    """
    使用來源 transforms.json 更新 VideoGS 的 transforms.json。
    """
    print("--- 步驟 3: 開始同步姿態到 VideoGS transforms.json ---")
    source_data = load_json(source_transforms_path)
    
    source_poses_map = {}
    for frame_info in source_data.get("frames", []):
        if "colmap_im_id" in frame_info and "transform_matrix" in frame_info:
            key_filename = f"{frame_info['colmap_im_id']}.png"
            source_poses_map[key_filename] = frame_info["transform_matrix"]

    if not source_poses_map:
        print(f"錯誤：未能從來源檔案 '{source_transforms_path}' 載入任何有效的姿態。")
        return

    target_data = load_json(target_videoGS_path)
    
    updated_frames_count = 0
    not_found_frames_log = []
    
    for frame_in_target in target_data.get("frames", []):
        target_base_filename = os.path.basename(frame_in_target.get("file_path", ""))
        if target_base_filename in source_poses_map:
            frame_in_target["transform_matrix"] = source_poses_map[target_base_filename]
            updated_frames_count += 1
        else:
            not_found_frames_log.append(target_base_filename)

    if not_found_frames_log:
        print(f"警告：在來源檔案中找不到以下 {len(not_found_frames_log)} 個目標影像的對應姿態：{', '.join(not_found_frames_log)}")

    save_json(target_data, output_path)
    print(f"已成功更新 {updated_frames_count} 個影格。")
    print(f"✅ 更新後的 VideoGS transforms 檔案已儲存到: {output_path}\n")

# --- 輔助工具: 創建遮罩 (來自 create_masks.py) ---

def create_masks(project_path):
    """
    從 RGBA 影像中提取 Alpha 通道作為遮罩。
    """
    print("--- 輔助工具: 開始創建遮罩 ---")
    project_path = Path(project_path)
    image_dir = project_path / "images"
    mask_dir = project_path / "masks"

    if not image_dir.exists():
        print(f"錯誤：找不到圖片資料夾 '{image_dir}'")
        return

    mask_dir.mkdir(exist_ok=True)
    print(f"將從 '{image_dir}' 創建遮罩到 '{mask_dir}'...")

    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() == '.png'])
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                if img.mode == 'RGBA':
                    alpha_channel = img.split()[-1]
                    mask_save_path = mask_dir / image_file.name
                    alpha_channel.save(mask_save_path)
                    print(f"已為 '{image_file.name}' 生成遮罩")
                else:
                    print(f"警告：圖片 '{image_file.name}' 沒有 Alpha 通道，已跳過。")
        except Exception as e:
            print(f"處理檔案 '{image_file.name}' 時發生錯誤: {e}")

    print(f"✅ 遮罩生成完畢！\n")

# --- 輔助工具: 複製檔案 (來自 copy_transforms.py) ---

def distribute_file(source_file_path, target_parent_directory):
    """
    將指定檔案複製到目標父資料夾下的所有直接子資料夾中。
    """
    print("--- 輔助工具: 開始分發檔案 ---")
    if not os.path.isfile(source_file_path):
        print(f"錯誤：來源檔案 '{source_file_path}' 不存在。")
        return
    if not os.path.isdir(target_parent_directory):
        print(f"錯誤：目標資料夾 '{target_parent_directory}' 不存在。")
        return

    source_filename = os.path.basename(source_file_path)
    copied_count = 0
    
    print(f"準備將 '{source_filename}' 複製到 '{target_parent_directory}' 的子資料夾中...")
    
    for entry_name in os.listdir(target_parent_directory):
        potential_subdir_path = os.path.join(target_parent_directory, entry_name)
        if os.path.isdir(potential_subdir_path):
            destination_file_path = os.path.join(potential_subdir_path, source_filename)
            try:
                shutil.copy2(source_file_path, destination_file_path)
                print(f"  已複製到: {destination_file_path}")
                copied_count += 1
            except Exception as e:
                print(f"  複製到 '{destination_file_path}' 時發生錯誤: {e}")
                
    print(f"✅ 複製操作完成，共複製了 {copied_count} 個檔案。\n")


def main():
    parser = argparse.ArgumentParser(description="一個整合的工具，用於還原 Nerfstudio 相機姿態並更新 VideoGS 資料集。")
    subparsers = parser.add_subparsers(dest="command", required=True, help="可執行的指令")

    # 指令 1: restore
    parser_restore = subparsers.add_parser("restore", help="步驟 1: 從 Nerfstudio 訓練結果中還原原始相機姿態。")
    parser_restore.add_argument("--dataparser-transforms", required=True, help="dataparser_transforms.json 的路徑。")
    parser_restore.add_argument("--transforms-train", required=True, help="transforms_train.json 的路徑。")
    parser_restore.add_argument("--output", required=True, help="儲存還原後姿態的 JSON 檔案路徑。")

    # 指令 2: update-nerfstudio
    parser_update_ns = subparsers.add_parser("update-nerfstudio", help="步驟 2: 將還原後的姿態更新回 Nerfstudio 的 transforms.json。")
    parser_update_ns.add_argument("--restored-poses", required=True, help="步驟 1 產生的還原姿態 JSON 檔案路徑。")
    parser_update_ns.add_argument("--target-transforms", required=True, help="原始的 Nerfstudio transforms.json 檔案路徑。")
    parser_update_ns.add_argument("--output", required=True, help="儲存更新後 Nerfstudio transforms.json 的路徑。")

    # 指令 3: update-videogs
    parser_update_vgs = subparsers.add_parser("update-videogs", help="步驟 3: 將 Nerfstudio 的姿態同步到 VideoGS 的 transforms.json。")
    parser_update_vgs.add_argument("--source-transforms", required=True, help="步驟 2 產生的、已更新的 Nerfstudio transforms.json 路徑。")
    parser_update_vgs.add_argument("--target-transforms", required=True, help="目標 VideoGS transforms.json 檔案路徑。")
    parser_update_vgs.add_argument("--output", required=True, help="儲存更新後 VideoGS transforms.json 的路徑。")

    # 指令 4: create-masks
    parser_masks = subparsers.add_parser("create-masks", help="輔助工具: 從 RGBA 影像創建遮罩。")
    parser_masks.add_argument("--project-path", required=True, help="包含 'images' 資料夾的專案路徑。")

    # 指令 5: distribute
    parser_distribute = subparsers.add_parser("distribute", help="輔助工具: 將一個檔案複製到多個子目錄中。")
    parser_distribute.add_argument("--source-file", required=True, help="要複製的來源檔案路徑 (例如最終的 transforms.json)。")
    parser_distribute.add_argument("--target-dir", required=True, help="包含多個子資料夾的目標父資料夾。")
    
    args = parser.parse_args()

    try:
        if args.command == "restore":
            restore_extrinsics(args.dataparser_transforms, args.transforms_train, args.output)
        elif args.command == "update-nerfstudio":
            update_nerfstudio_transforms(args.restored_poses, args.target_transforms, args.output)
        elif args.command == "update-videogs":
            update_videogs_transforms(args.source_transforms, args.target_transforms, args.output)
        elif args.command == "create-masks":
            create_masks(args.project_path)
        elif args.command == "distribute":
            distribute_file(args.source_file, args.target_dir)
    except FileNotFoundError as e:
        print(f"\n錯誤：找不到檔案 '{e.filename}'。請檢查您的路徑是否正確。")
    except Exception as e:
        print(f"\n處理過程中發生未預期的錯誤: {e}")

if __name__ == "__main__":
    main()