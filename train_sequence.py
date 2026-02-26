import os
import sys
import argparse
import shutil
import json
import pymeshlab
import open3d as o3d
import numpy as np

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
    parser.add_argument('--eval', action='store_true', default=False)
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

    # neus2_meshlab_filter_path = os.path.join(data_root_path, "luoxi_filter.mlx")

    neus2_output_path = os.path.join(output_path, "neus2_output")
    if not os.path.exists(neus2_output_path):
        os.makedirs(neus2_output_path)

    gaussian_output_path = os.path.join(output_path, "checkpoint")

    import time
    time_dict = {}
    time_list = []

    for i in range(args.start, args.end, group_size * interval):
        group_start = i
        group_end = min(i + group_size * interval, args.end) - 1
        print(group_start, group_end)
        
        t0 = time.time()
        
        frame_path = os.path.join(data_root_path, str(i))
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        frame_neus2_output_path = os.path.join(neus2_output_path, str(i))
        if not os.path.exists(frame_neus2_output_path):
            os.makedirs(frame_neus2_output_path)
        frame_neus2_ckpt_output_path = os.path.join(frame_neus2_output_path, "frame.msgpack")
        frame_neus2_mesh_output_path = os.path.join(frame_neus2_output_path, "points3d.obj")
        
        """NeuS2"""
        # Data split for NeuS2 if eval is True
        transforms_path = os.path.join(frame_path, "transforms.json")
        transforms_full_path = os.path.join(frame_path, "transforms_full.json")
        
        if args.eval and os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                transforms_data = json.load(f)
            
            all_frames = transforms_data.get('frames', [])
            train_frames = [f for idx, f in enumerate(all_frames) if idx % 8 != 0]
            test_frames = [f for idx, f in enumerate(all_frames) if idx % 8 == 0]
            
            # Save Test split for manual inspection
            transforms_data['frames'] = test_frames
            with open(os.path.join(frame_path, "transforms_test.json"), 'w') as f:
                json.dump(transforms_data, f, indent=4)
                
            # Create Train split
            transforms_data['frames'] = train_frames
            
            # Backup full and swap
            shutil.move(transforms_path, transforms_full_path)
            with open(transforms_path, 'w') as f:
                json.dump(transforms_data, f, indent=4)
            print(f"NeuS2 Split: {len(train_frames)} train frames, {len(test_frames)} test frames.")

        try:
            # neus2 command
            script_path = "scripts/run.py"
            neus2_command = f"cd external/NeuS2_K && CUDA_VISIBLE_DEVICES={card_id} {sys.executable} {script_path} --scene {frame_path} --name neus --mode nerf --save_snapshot {frame_neus2_ckpt_output_path} --save_mesh --save_mesh_path {frame_neus2_mesh_output_path} && cd ../.."
            os.system(neus2_command)
            delete_neus2_output_path = os.path.join(frame_path, "output")
            if os.path.exists(delete_neus2_output_path):
                shutil.rmtree(delete_neus2_output_path)
        finally:
            # Restore transforms.json for subsequent steps
            if args.eval and os.path.exists(transforms_full_path):
                 if os.path.exists(transforms_path):
                     os.remove(transforms_path) # Remove the temp train file
                 shutil.move(transforms_full_path, transforms_path)
                 print("Restored full transforms.json")

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
        first_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} {sys.executable} train.py -s {frame_path} -m {frame_model_path} --iterations {first_frame_iteration} --save_iterations {first_frame_save_iterations} --sh_degree {sh} -r {resolution_scale} --port 600{card_id}"
        if args.eval:
            first_gaussian_command += " --eval"
        os.system(first_gaussian_command)

        # prune
        prune_iterations = 4000
        prune_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} {sys.executable} prune_gaussian.py -s {frame_path} -m {frame_model_path} --sh_degree {sh} -r {resolution_scale} --iterations {prune_iterations}"
        os.system(prune_gaussian_command)

        # rest frame
        dynamic_command = f"CUDA_VISIBLE_DEVICES={card_id} {sys.executable} train_dynamic.py -s {data_root_path} -m {gaussian_output_path} --sh_degree {sh} -r {resolution_scale} --st {group_start} --ed {group_end} --interval {interval}"
        if args.eval:
            dynamic_command += " --eval"
        os.system(dynamic_command)

        print(f"Finish {group_start} to {group_end}")
        
        t1 = time.time()
        group_time = t1 - t0
        time_dict[f"group_{group_start}_to_{group_end}"] = {
            "Time_seconds": group_time,
            "Time_seconds_per_frame": group_time / group_size
        }
        time_list.append(group_time)

    if time_dict:
        total_time = sum(time_list)
        total_frames = group_size * len(time_list)
        full_dict = {
            "Overall": {
                "Total_Time_seconds": total_time,
                "Average_Time_seconds_per_frame": total_time / total_frames if total_frames > 0 else 0
            },
            "Details": time_dict
        }
        output_json = os.path.join(output_path, "results_train_time.json")
        with open(output_json, 'w') as f:
            json.dump(full_dict, f, indent=4)
        print(f"Training time results saved to {output_json}")