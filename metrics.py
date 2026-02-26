import os
import json
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

# Assume these are available in VideoGS root
from utils.loss_utils import ssim
from utils.image_utils import psnr
import lpips

def readImages(renders_dir, gt_dir):
    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)
    renders = []
    gts = []
    image_names = []
    
    if not renders_dir.exists() or not gt_dir.exists():
        return renders, gts, image_names
        
    for fname in sorted(os.listdir(renders_dir)):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        
        if not gt_path.exists():
            continue
            
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
        
    return renders, gts, image_names

def _run_average_metrics(renders, gts, lpips_fn):
    ssims = []
    psnrs = []
    lpipss = []

    for idx in range(len(renders)):
        render_rescaled = renders[idx] * 2.0 - 1.0
        gt_rescaled = gts[idx] * 2.0 - 1.0
        
        ssims.append(ssim(renders[idx], gts[idx]).item())
        psnrs.append(psnr(renders[idx], gts[idx]).item())
        lpipss.append(lpips_fn(render_rescaled, gt_rescaled).mean().item())

    return np.mean(ssims), np.mean(psnrs), np.mean(lpipss)

def evaluate(model_path, renders_folder="decompressed_renders"):
    full_dict = {}
    print(f"---Evaluating VideoGS Renders in: {renders_folder}---")
    
    stage_path = os.path.join(model_path, renders_folder)
    if not os.path.exists(stage_path):
        print(f"Error: Path {stage_path} does not exist.")
        return

    # In evaluate_compression.py, it's saved as "frame_00000" etc.
    window_dirs = sorted([d for d in os.listdir(stage_path) if d.startswith("frame_")])
    
    if window_dirs:
        print(f"Found {len(window_dirs)} Frames in {renders_folder}")
        all_ssims, all_psnrs, all_lpipss = [], [], []
        window_results = {}
        
        for win_dir in tqdm(window_dirs, desc="Processing Frame"):
            win_path = os.path.join(stage_path, win_dir)
            test_dir = os.path.join(win_path, "test")
            
            renders_dir = os.path.join(test_dir, "renders")
            gt_dir = os.path.join(test_dir, "gt")
            
            if not os.path.exists(renders_dir) or not os.path.exists(gt_dir):
                continue
                
            renders, gts, _ = readImages(renders_dir, gt_dir)
            if renders:
                w_ssim, w_psnr, w_lpips = _run_average_metrics(renders, gts, lpips_fn)
                window_results[win_dir] = {
                    "SSIM": w_ssim, "PSNR": w_psnr, "LPIPS": w_lpips
                }
                all_ssims.append(w_ssim)
                all_psnrs.append(w_psnr)
                all_lpipss.append(w_lpips)
                
        if all_ssims:
            avg_ssim = np.mean(all_ssims)
            avg_psnr = np.mean(all_psnrs)
            avg_lpips = np.mean(all_lpipss)
                        
            print(f"  Avg SSIM : {avg_ssim:>12.7f}")
            print(f"  Avg PSNR : {avg_psnr:>12.7f}")
            print(f"  Avg LPIPS: {avg_lpips:>12.7f}")
                        
            full_dict[f"{renders_folder}_Overall"] = {
                "SSIM": avg_ssim, "PSNR": avg_psnr, "LPIPS": avg_lpips
            }
            full_dict[f"{renders_folder}_Details"] = window_results
        else:
            print(f"  No valid images found in {renders_folder}.")
            
    # Save the output to results_decompressed_renders.json
    output_json = os.path.join(model_path, f"results_{renders_folder}.json")
    
    # Merge existing results if any
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r') as fp:
                existing_data = json.load(fp)
                if isinstance(existing_data, dict):
                    full_dict = {**existing_data, **full_dict}
        except Exception as e:
            print(f"Warning: Failed to load existing {output_json}, creating new. Error: {e}")

    with open(output_json, 'w') as fp:
        json.dump(full_dict, fp, indent=4)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--model_path', '-m', required=True, type=str, help="Path to the model directory (e.g. outputs/something)")
    parser.add_argument('--renders_folder', type=str, default="decompressed_renders", help="Folder inside model_path containing rendered frames")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.renders_folder)
