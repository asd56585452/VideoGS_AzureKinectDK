import os
import time
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_start", type=int, default=95)
    parser.add_argument("--frame_end", type=int, default=115)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--qp", type=int, default=25)
    parser.add_argument("--sh_degree", type=int, default=3)

    args = parser.parse_args()
    
    group_size = args.group_size
    init_index = args.frame_start
    group_num = int((args.frame_end - args.frame_start) / group_size)
    
    # Calculate number of videos based on sh_degree
    sh_degree = args.sh_degree
    SH_N = (sh_degree + 1) * (sh_degree + 1)
    sh_number = SH_N * 3
    # Total attributes = 3(pos) + 3(dc) + sh_number(rest) + 1(op) + 3(scale) + 4(rot)
    # Total videos = 6 + (3 + sh_number + 1 + 3 + 4)
    # Actually, simpler: 6 (pos) + non_pos_attributes.
    # non_pos = 3 + sh_number + 1 + 3 + 4.
    num_video = 6 + 3 + sh_number + 1 + 3 + 4
    print(f"Compressing {num_video} video streams (SH degree {sh_degree})")

    # base_dir = f"/data/new_disk5/wangph1/output/xyz_smalliter_group40/feature_image"
    # save_dir = f"/data/new_disk5/wangph1/output/xyz_smalliter_group40/feature_video"
    base_dir = os.path.join(args.output_path, "feature_image")
    save_dir = os.path.join(args.output_path, "feature_video")
    os.makedirs(save_dir, exist_ok=True)
    # qps = [12, 14, 16, 18, 20, 23, 28]
    # qps = [0, 5, 10, 15, 20, 22, 25]
    # qps = [15]
    qps = [args.qp]
    # qps = [28, 30, 32, 35, 37, 40, 50]
    # qps = [43, 45, 47, 50]
    # qps = [43, 45, 47, 50, 0, 10, 15, 22, 26, 32, 37, 40]
    # qps = [15]
    import json
    for qp in qps:
        out_dir = os.path.join(save_dir, "png_all_" + str(qp))
        # out_dir = save_dir
        # if os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # video_path = os.path.join(out_dir, "video")
        # os.makedirs(video_path, exist_ok=True)

        size_dict = {}
        total_size = 0.0
        total_frames = 0

        for group in range(group_num):

            start_index = group * group_size + init_index
            # end_index = (group + 1) * group_size
            group_path = os.path.join(out_dir, "group" + str(group))
            os.makedirs(group_path, exist_ok=True)
            # group_video_path = os.path.join(group_path, "video")
            # os.makedirs(group_video_path, exist_ok=True)
            group_video_path = group_path

            input_group_path = os.path.join(base_dir, "group" + str(group))

            for i in range(0, num_video):
                if i in [1,3,5]:
                    os.system(f"ffmpeg -start_number {start_index} -i {input_group_path}/%d_{i}.png -vframes {group_size} -c:v libx264 -qp 0 -pix_fmt yuvj444p {group_video_path}/{i}.mp4")
                elif i in [9, 10, 11,  13, 14, 15,  16, 17, 18, 19] and qp > 22:
                    os.system(f"ffmpeg -start_number {start_index} -i {input_group_path}/%d_{i}.png -vframes {group_size} -c:v libx264 -qp 22 -pix_fmt yuvj444p {group_video_path}/{i}.mp4")
                else:
                    os.system(f"ffmpeg -start_number {start_index} -i {input_group_path}/%d_{i}.png -vframes {group_size} -c:v libx264 -qp {qp} -pix_fmt yuvj444p {group_video_path}/{i}.mp4")
            
            group_size_bytes = 0
            for i in range(num_video):
                video_file = os.path.join(group_video_path, f"{i}.mp4")
                if os.path.exists(video_file):
                    group_size_bytes += os.path.getsize(video_file)
            
            group_size_mb = group_size_bytes / (1024 * 1024)
            size_dict[f"group_{group}"] = {
                "Size_MB": group_size_mb,
                "Size_MB_per_frame": group_size_mb / group_size
            }
            total_size += group_size_mb
            total_frames += group_size

        if size_dict:
            full_dict = {
                "Overall": {
                    "Total_Size_MB": total_size,
                    "Average_Size_MB_per_frame": total_size / total_frames if total_frames > 0 else 0
                },
                "Details": size_dict
            }
            output_json = os.path.join(out_dir, "results_compression_size.json")
            with open(output_json, 'w') as f:
                json.dump(full_dict, f, indent=4)
            print(f"Compression size results saved to {output_json}")

    # copy the json
    # shutil.copy(os.path.join(base_dir, "min_max.json"), os.path.join(out_dir, "min_max.json"))
    shutil.copy(os.path.join(base_dir, "viewer_min_max.json"), os.path.join(out_dir, "viewer_min_max.json"))
    shutil.copy(os.path.join(base_dir, "group_info.json"), os.path.join(out_dir, "group_info.json"))


    print("finish")
