import torch
from scene import Scene, GaussianModel, DynamicScene
import os
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
import sys
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from scene.dataset_readers import sceneLoadTypeCallbacks
import time
import json

def evaluate(dataset, opt, pipe, ply_path, window_name,frame_id,group_size):
    # Initialize Gaussian Model
    t0 = time.time()
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Setup scene but don't load default gaussians if we are providing specific PLYs
    # We pass load_iteration=-1 to avoid auto-loading
    if frame_id%group_size==0:
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    else:
        scene = DynamicScene(dataset)
    
    # Use test cameras
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras found! Make sure to use --eval")
        return
    t1 = time.time()
    print(f"[Timer] Scene & Camera Loading: {t1 - t0:.3f}s")

    # Determine PLY loading strategy
    # If ply_path is a file, use it for all frames (static evaluation?) 
    # If ply_path is a folder, look for {frame_id}.ply
    
    is_folder = os.path.isdir(ply_path)
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_l1 = 0.0
    count = 0
    total_io_time = 0.0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"Evaluating on {len(test_cameras)} test images...")
    
    for idx, viewpoint in enumerate(tqdm(test_cameras)):
        
        # Determine PLY to load
        current_ply = None
        if is_folder:
            # Try frame_id first
            candidates = [
                os.path.join(ply_path, f"{frame_id}.ply"),
                os.path.join(ply_path, f"point_cloud.ply") 
            ]
            for c in candidates:
                if os.path.exists(c):
                    current_ply = c
                    break
        else:
            current_ply = ply_path

        if not current_ply or not os.path.exists(current_ply):
            print(f"PLY not found for frame {frame_id} at {current_ply}")
            continue

        # Load PLY
        t2 = time.time()
        # try:
        gaussians.load_ply(current_ply)
        # except Exception as e:
        #     # Fallback for SH mismatch: try re-init with SH=0
        #     if "dimension mismatch" in str(e) or "assert" in str(e) or "empty" in str(e):
        #         print(f"Warning: Failed to load PLY with SH={dataset.sh_degree}, trying SH=0. Error: {e}")
        #         gaussians = GaussianModel(0)
        #         gaussians.load_ply(current_ply)
        #     else:
        #         raise e
        t3 = time.time()
        # print(f"[Timer] Load PLY: {t3 - t2:.3f}s")
        
        # Render
        render_pkg = render(viewpoint, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint.original_image.cuda()
        
        # Calculate metrics
        # l1 = l1_loss(image, gt_image).mean().double()
        # p = psnr(image, gt_image).mean().double()
        # s = ssim(image, gt_image).mean().double()
        
        # avg_psnr += p.item()
        # avg_ssim += s.item()
        # avg_l1 += l1.item()
        # count += 1
        
        io_start = time.time()
        # Save render and GT
        import torchvision
        
        # Define base path structure according to metrics.py requirements
        # Structure: {model_path}/stage2/window_0/test/ours/
        base_output_dir = os.path.join(args.model_path, "decompressed_renders", window_name, "test")
        renders_dir = os.path.join(base_output_dir, "renders")
        gt_dir = os.path.join(base_output_dir, "gt")
        
        makedirs(renders_dir, exist_ok=True)
        makedirs(gt_dir, exist_ok=True)
        
        torchvision.utils.save_image(image, os.path.join(renders_dir, f"{viewpoint.image_name}.png"))
        torchvision.utils.save_image(gt_image, os.path.join(gt_dir, f"{viewpoint.image_name}.png"))
        io_end = time.time()
        total_io_time += (io_end - io_start)
        
        count += 1
    t4 = time.time()
    rendered_time = (t4 - t1) - total_io_time
    fps = count / rendered_time if rendered_time > 0 else 0
    print(f"[Timer] Rendering Loop (excl. IO): {rendered_time:.3f}s, FPS: {fps:.2f}")

    if count > 0:
        print(f"\nEvaluation Complete. Saved {count} pairs to {base_output_dir}")
        return fps
    else:
        print("No frames evaluated.")
        return 0

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ply_path', type=str, required=True, help="Path to decompressed PLY folder or file")
    
    # Sequence arguments
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=20)
    parser.add_argument("--group_size", type=int, default=20)
    args = parser.parse_args(sys.argv[1:])
    
    print("Evaluating " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    
    base_source_path = args.source_path
    
    fps_dict = {}
    fps_list = []
    
    # Iterate over groups
    for i in range(args.frame_start, args.frame_end):
        frame_source_path = os.path.join(base_source_path, str(i))
        if not os.path.exists(frame_source_path):
            print(f"Skipping group {i}: Path {frame_source_path} not found")
            continue
            
        print(f"\n=== Evaluating Group starting at frame {i} ===")
        print(f"Dataset path: {frame_source_path}")
        
        # Update source_path in args temporarily for extraction
        args.source_path = frame_source_path
        
        # Calculate window name
        frame_idx = i
        window_name = f"frame_{frame_idx:05d}"
        
        fps = evaluate(lp.extract(args), op.extract(args), pp.extract(args), args.ply_path, window_name,frame_idx,args.group_size)
        fps_dict[window_name] = {"FPS": fps}
        if fps > 0:
            fps_list.append(fps)

    if fps_dict:
        avg_fps = sum(fps_list) / len(fps_list) if len(fps_list) > 0 else 0
        full_dict = {
            "Overall": {"FPS": avg_fps},
            "Details": fps_dict
        }
        output_json = os.path.join(args.model_path, "results_fps.json")
        with open(output_json, 'w') as f:
            json.dump(full_dict, f, indent=4)
        print(f"FPS results saved to {output_json}")

    # evaluate(lp.extract(args), op.extract(args), pp.extract(args), args.ply_path)
