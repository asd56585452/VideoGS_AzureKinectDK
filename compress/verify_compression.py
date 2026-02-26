import argparse
import numpy as np
from plyfile import PlyData

def get_ply_matrix(file_path):
    plydata = PlyData.read(file_path)
    num_vertices = len(plydata['vertex'])
    num_attributes = len(plydata['vertex'].properties)
    data_matrix = np.zeros((num_vertices, num_attributes), dtype=float)
    attribute_names = plydata['vertex'].data.dtype.names
    
    # Ensuring consistent ordering, but assuming both PLYs have the same property order
    for i, name in enumerate(attribute_names):
        data_matrix[:, i] = plydata['vertex'].data[name]
        
    return data_matrix, attribute_names

def find_attribute_indices(attribute_names, prefixes):
    indices = []
    for prefix in prefixes:
        # Match exact names like 'x', 'y' or prefix names like 'f_dc_0'
        matched = [i for i, name in enumerate(attribute_names) if name == prefix or name.startswith(prefix + "_")]
        indices.extend(matched)
    return indices

def apply_forward_activations(data, attr_names, sh_degree=3):
    """
    Applies the same activation functions used in compress_ckpt_2_image_precompute.py
    so we can compare values in the pre-quantized space (0-1 or 16-bit space).
    """
    data_transformed = data.copy()
    
    # Finding attribute indices dynamically to be safe
    scale_indices = find_attribute_indices(attr_names, ['scale'])
    opacity_index = find_attribute_indices(attr_names, ['opacity'])[0]
    rot_indices = find_attribute_indices(attr_names, ['rot'])
    
    # 1. Rotation Normalization
    rotation_data = data_transformed[:, rot_indices]
    rotation_length = np.sqrt(np.sum(rotation_data ** 2, axis=1))
    data_transformed[:, rot_indices] = rotation_data / rotation_length[:, None]
    
    # 2. Scale Exp
    data_transformed[:, scale_indices] = np.exp(data_transformed[:, scale_indices])
    
    # 3. Opacity Sigmoid
    data_transformed[:, opacity_index] = 1 / (1 + np.exp(-data_transformed[:, opacity_index]))
    
    # 4. SH Rearrangement
    if sh_degree > 0:
        dc_indices = find_attribute_indices(attr_names, ['f_dc'])
        rest_indices = find_attribute_indices(attr_names, ['f_rest'])
        
        # In PLY, they are typically contiguous: dc_0, dc_1, dc_2, rest_0, ...
        # Let's assume they are contiguous starting at dc_0
        sh_start_idx = dc_indices[0] 
        SH_N = (sh_degree + 1) ** 2
        sh_number = SH_N * 3
        
        shs_data = data_transformed[:, sh_start_idx:sh_start_idx + sh_number].copy()
        data_transformed[:, sh_start_idx + 0] = shs_data[:, 0]
        data_transformed[:, sh_start_idx + 1] = shs_data[:, 1]
        data_transformed[:, sh_start_idx + 2] = shs_data[:, 2]
        
        for j in range(1, SH_N):
            data_transformed[:, sh_start_idx + j * 3 + 0] = shs_data[:, (j - 1) + 3]
            data_transformed[:, sh_start_idx + j * 3 + 1] = shs_data[:, (j - 1) + SH_N + 2]
            data_transformed[:, sh_start_idx + j * 3 + 2] = shs_data[:, (j - 1) + 2 * SH_N + 1]
            
    return data_transformed

def main():
    parser = argparse.ArgumentParser(description="Verify compression and decompression error")
    parser.add_argument("--original", type=str, required=True, help="Path to original PLY")
    parser.add_argument("--decompressed", type=str, required=True, help="Path to decompressed PLY")
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading original PLY: {args.original}")
    orig_data, orig_attrs = get_ply_matrix(args.original)
    
    print(f"Loading decompressed PLY: {args.decompressed}")
    decomp_data, decomp_attrs = get_ply_matrix(args.decompressed)

    assert orig_data.shape == decomp_data.shape, f"Shape mismatch: {orig_data.shape} vs {decomp_data.shape}"

    print("-" * 50)
    print("Applying forward activations to both PLYs for fair comparison in quantization space...")
    orig_transformed = apply_forward_activations(orig_data, orig_attrs, args.sh_degree)
    decomp_transformed = apply_forward_activations(decomp_data, decomp_attrs, args.sh_degree)

    print("-" * 50)
    print("Calculating Absolute Errors (Transformed Space):")
    
    # Position uses 16-bit quantization (65535 levels)
    # Others use 8-bit quantization (255 levels)
    
    pos_indices = find_attribute_indices(orig_attrs, ['x', 'y', 'z'])
    dc_indices = find_attribute_indices(orig_attrs, ['f_dc'])
    rest_indices = find_attribute_indices(orig_attrs, ['f_rest'])
    opacity_index = find_attribute_indices(orig_attrs, ['opacity'])
    scale_indices = find_attribute_indices(orig_attrs, ['scale'])
    rot_indices = find_attribute_indices(orig_attrs, ['rot'])
    
    groups = {
        'Position (16-bit)': pos_indices,
        'SH DC (8-bit)': dc_indices,
        'SH Rest (8-bit)': rest_indices,
        'Opacity (8-bit)': opacity_index,
        'Scale (8-bit)': scale_indices,
        'Rotation (8-bit)': rot_indices
    }

    total_max_error = 0
    
    for name, indices in groups.items():
        if not indices:
            continue
            
        orig_group = orig_transformed[:, indices]
        decomp_group = decomp_transformed[:, indices]
        
        diff = np.abs(orig_group - decomp_group)
        
        # Normalize error based on min-max range to estimate quantization step error
        range_val = np.max(orig_group, axis=0) - np.min(orig_group, axis=0)
        # Avoid division by zero
        range_val = np.maximum(range_val, 1e-6)
        
        # Quantization steps
        quant_levels = 65535.0 if '16-bit' in name else 255.0
        
        # Max theoretical quantization error in normalized space is 1.0 (sometimes slightly more due to rounding)
        # We calculate: Error / (Range / QuantLevels)
        quantization_steps_error = diff / (range_val / quant_levels)
        
        max_err_steps = np.max(quantization_steps_error)
        mean_err_steps = np.mean(quantization_steps_error)
        
        print(f"{name:20s}: Max Error = {max_err_steps:.3f} quantization steps, Mean Error = {mean_err_steps:.3f} steps")
        total_max_error = max(total_max_error, max_err_steps)

    print("-" * 50)
    if total_max_error < 3.0: # Allowing for some video compression artifacts (e.g. H.264 lossy)
        print("✅ SUCCESS: The decompression error is extremely small, typically within expected video encoding/quantization error bounds.")
    else:
        print(f"⚠️ WARNING: The maximum error is {total_max_error:.2f} quantization steps. Since video compression (H.264) is lossy natively, some deviation > 1 step is expected, but investigate if this number is extremely high (e.g., > 10).")

if __name__ == "__main__":
    main()
