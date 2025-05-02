import os
import argparse
import shutil
import pymeshlab
import open3d as o3d
import numpy as np
import subprocess
import re
import time
import shlex

# group_size = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default='')
    parser.add_argument('--end', type=int, default='')
    parser.add_argument('--cuda', type=int, default='')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--sh', type=str, default='0')
    parser.add_argument('--interval', type=str, default='')
    parser.add_argument('--group_size', type=str, default='')
    parser.add_argument('--resolution', type=int, default=2)
    parser.add_argument('--point3d', action="store_true", help='If use pcd as init')
    args = parser.parse_args()

    print(args.start, args.end)

    # os.system("conda activate torch")
    card_id = args.cuda
    data_root_path = args.data
    output_path = args.output
    sh = args.sh
    interval = int(args.interval)
    group_size = int(args.group_size)
    resolution_scale = int(args.resolution)
    max_retries = 3 # 設定最大重試次數
    retry_delay = 5

    # neus2_meshlab_filter_path = os.path.join(data_root_path, "luoxi_filter.mlx")

    neus2_output_path = os.path.join(output_path, "neus2_output")
    if not os.path.exists(neus2_output_path):
        os.makedirs(neus2_output_path)

    gaussian_output_path = os.path.join(output_path, "checkpoint")

    for i in range(args.start, args.end, group_size * interval):
        success = False
        while not success:
            group_start = i
            group_end = min(i + group_size * interval, args.end) - 1
            print(group_start, group_end)
            
            frame_path = os.path.join(data_root_path, str(i))
            if not os.path.exists(frame_path):
                os.makedirs(frame_path)
            frame_neus2_output_path = os.path.join(neus2_output_path, str(i))
            if not os.path.exists(frame_neus2_output_path):
                os.makedirs(frame_neus2_output_path)
            frame_neus2_ckpt_output_path = os.path.join(frame_neus2_output_path, "frame.msgpack")
            frame_neus2_mesh_output_path = os.path.join(frame_neus2_output_path, "points3d.obj")
            
            if not args.point3d:
                """NeuS2"""
                # neus2 command
                script_path = "scripts/run.py"
                neus2_command = f"cd external/NeuS2_K && CUDA_VISIBLE_DEVICES={card_id} python {script_path} --scene {frame_path} --name neus --mode nerf --save_snapshot {frame_neus2_ckpt_output_path} --save_mesh --save_mesh_path {frame_neus2_mesh_output_path} && cd ../.."
                os.system(neus2_command)
                delete_neus2_output_path = os.path.join(frame_path, "output")
                shutil.rmtree(delete_neus2_output_path)

                # revert axis
                mesh1 = o3d.io.read_triangle_mesh(frame_neus2_mesh_output_path)
                vertices = np.asarray(mesh1.vertices)
                vertices = vertices[:,[2,0,1]]
                mesh1.vertices = o3d.utility.Vector3dVector(vertices)
                o3d.io.write_triangle_mesh(frame_neus2_mesh_output_path, mesh1)

                # use pymeshlab to convert obj to point cloud
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(frame_neus2_mesh_output_path)
                # ms.load_filter_script(neus2_meshlab_filter_path)
                # ms.apply_filter_script()
                ms.generate_simplified_point_cloud(samplenum = 100000) 
                frame_points3d_output_path = os.path.join(frame_path, "points3d.ply")
                ms.save_current_mesh(frame_points3d_output_path, binary = True, save_vertex_normal = False)


            """ Gaussian """
            # generate output
            frame_model_path = os.path.join(gaussian_output_path, str(i))
            first_frame_iteration = 12000
            first_frame_save_iterations = first_frame_iteration
            first_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} python train.py -s {frame_path} -m {frame_model_path} --iterations {first_frame_iteration} --save_iterations {first_frame_save_iterations} --sh_degree {sh} -r {resolution_scale} --port 600{card_id}"
            os.system(first_gaussian_command)

            # prune
            prune_iterations = 4000
            prune_percentage = 0.1
            prune_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} python prune_gaussian.py -s {frame_path} -m {frame_model_path} --sh_degree {sh} -r {resolution_scale} --iterations {prune_iterations} --prune_percentage {prune_percentage}"
            os.system(prune_gaussian_command)

            # rest frame
            dynamic_command = f"CUDA_VISIBLE_DEVICES={card_id} python train_dynamic.py -s {data_root_path} -m {gaussian_output_path} --sh_degree {sh} -r {resolution_scale} --st {group_start} --ed {group_end} --interval {interval}"
            # os.system(dynamic_command)
            retries = 0
            while retries < max_retries:
                print(f"--- 第 {retries + 1}/{max_retries} 次嘗試 ---")
                print(f"執行命令: {dynamic_command}")

                process = None # 初始化 process 變數
                try:
                    # 推薦使用 shlex 分割命令
                    # cmd_list = shlex.split(dynamic_command)
                    # process = subprocess.run(cmd_list, capture_output=True, text=True, check=False)

                    # 或者使用 shell=True (注意安全風險)
                    process = subprocess.run(dynamic_command,
                                            shell=True,
                                            capture_output=True,
                                            text=True,
                                            check=False)

                    stdout = process.stdout
                    stderr = process.stderr
                    return_code = process.returncode

                    # 先印出每次嘗試的基本資訊 (可選，用於除錯)
                    print(f"命令返回碼: {return_code}")
                    print("--- STDOUT (本次嘗試預覽) ---")
                    print(stdout[:200] + "..." if len(stdout) > 200 else stdout if stdout else "<無標準輸出>") # 預覽部分輸出
                    print("--- STDERR (本次嘗試預覽) ---")
                    print(stderr[:200] + "..." if len(stderr) > 200 else stderr if stderr else "<無錯誤輸出>") # 預覽部分錯誤
                    print("-----------------------------")


                    if return_code != 0:
                        print(f"命令執行失敗，返回碼非零 ({return_code})。")

                    # 條件 1: 檢查是否包含完成字串
                    completion_string = f"Training frame {group_end} done"
                    found_completion_string = completion_string in stdout
                    print(f"檢查完成字串 '{completion_string}': {found_completion_string}")

                    # 條件 2: 檢查 *最後一個* PSNR 是否大於 30
                    psnr_value = None
                    psnr_ok = False
                    psnr_matches = re.findall(r"PSNR\s+(\d+\.?\d*)", stdout)

                    if psnr_matches:
                        last_psnr_str = psnr_matches[-1]
                        # (可選) 印出找到的 PSNR 資訊
                        # print(f"找到的所有 PSNR 值字串: {psnr_matches}")
                        print(f"使用最後一個 PSNR 值進行檢查: {last_psnr_str}")
                        try:
                            psnr_value = float(last_psnr_str)
                            print(f"轉換後的最後 PSNR 值: {psnr_value}")
                            if psnr_value > 30:
                                psnr_ok = True
                                print("最後 PSNR 檢查通過 (> 30)")
                            else:
                                print(f"最後 PSNR 檢查失敗 ({psnr_value} <= 30)")
                        except ValueError:
                            print(f"錯誤：無法將找到的最後 PSNR 值字串 '{last_psnr_str}' 轉換為浮點數。")
                    else:
                        print("在輸出中未找到任何 'PSNR xxx' 格式的字串。")

                    # 檢查所有條件是否滿足
                    if return_code == 0 and found_completion_string and psnr_ok:
                        print("\n========================================")
                        print("成功：命令執行完成且滿足所有條件。")
                        print("========================================")
                        success = True

                        # --- 輸出成功執行的完整結果 ---
                        print("\n--- 命令成功執行結果 ---")
                        print("\n--- STDOUT ---")
                        print(stdout if stdout else "<無標準輸出>")
                        # print("\n--- STDERR ---")
                        # print(stderr if stderr else "<無錯誤輸出>")
                        # print("\n------------------------")
                        # --- 結束輸出 ---

                        break # 成功，跳出 while 循環

                    else:
                        # 條件未滿足，準備重試
                        print("條件未滿足，準備重試...")
                        # (可以保留之前的詳細失敗原因輸出)
                        if return_code != 0: print("  - 原因：命令返回碼非零")
                        if not found_completion_string: print(f"  - 原因：未找到完成字串 '{completion_string}'")
                        if not psnr_ok:
                            if psnr_matches:
                                print(f"  - 原因：最後 PSNR 值 ({psnr_value}) 未 > 30")
                            else:
                                print("  - 原因：未找到 PSNR 值")


                except FileNotFoundError:
                    print(f"錯誤：命令 '{dynamic_command.split()[0]}' 不存在或無法執行。")
                    print("無法重試，請檢查命令。")
                    break
                except Exception as e:
                    print(f"執行命令時發生未預期的錯誤: {e}")
                    # 檢查 process 是否被賦值，如果命令執行出錯可能為 None
                    if process:
                        print("--- STDOUT (錯誤發生時) ---")
                        print(process.stdout if process.stdout else "<無標準輸出>")
                        print("--- STDERR (錯誤發生時) ---")
                        print(process.stderr if process.stderr else "<無錯誤輸出>")
                    print("準備重試...")

                # 重試前的準備
                retries += 1
                if retries < max_retries:
                    print(f"\n將在 {retry_delay} 秒後重試...")
                    time.sleep(retry_delay)
                    print("-" * 30) # 分隔線，讓輸出更清晰

            # 循環結束後的操作
            if not success: # 只有在失敗時才需要額外輸出這個訊息
                print(f"\n============================================================")
                print(f"最終結果：達到最大重試次數 ({max_retries})，命令未能成功或滿足條件。")
                print(f"============================================================")
                # 如果需要，可以在這裡印出最後一次嘗試的輸出 (如果 process 有被賦值的話)
                if process:
                    print("\n--- 最後一次嘗試的輸出 ---")
                    print("--- STDOUT ---")
                    print(process.stdout if process.stdout else "<無標準輸出>")
                    print("--- STDERR ---")
                    print(process.stderr if process.stderr else "<無錯誤輸出>")
                    print("-------------------------")


            print(f"Finish {group_start} to {group_end}")