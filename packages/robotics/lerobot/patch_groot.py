#!/usr/bin/env python3
"""Patch GR00T for compatibility with transformers >= 4.46."""

import os

print("=" * 60)
print("GR00T Compatibility Patches")
print("=" * 60)

# ============================================================================
# Patch 1: Fix Beta distribution initialization and sampling
# ============================================================================
file1 = '/opt/lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py'
if os.path.exists(file1):
    with open(file1, 'r') as f:
        content = f.read()

    patches_applied = []

    # Patch 1a: Fix old patch that used torch.tensor (creates CPU tensors)
    if 'torch.tensor(config.noise_beta_alpha)' in content:
        content = content.replace(
            'self.beta_dist = Beta(torch.tensor(config.noise_beta_alpha), torch.tensor(config.noise_beta_beta), validate_args=False)',
            'self.beta_dist = Beta(float(config.noise_beta_alpha), float(config.noise_beta_beta), validate_args=False)'
        )
        patches_applied.append("float params")

    # Patch 1b: Add validate_args=False if not present
    if 'validate_args=False' not in content:
        content = content.replace(
            'self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)',
            'self.beta_dist = Beta(float(config.noise_beta_alpha), float(config.noise_beta_beta), validate_args=False)'
        )
        patches_applied.append("validate_args")

    # Patch 1c: Fix sample_time - use uniform sampling instead of Beta to avoid BFloat16 issues
    if '# Use uniform sampling to avoid' not in content:
        content = content.replace(
            '''def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s''',
            '''def sample_time(self, batch_size, device, dtype):
        # Use uniform sampling to avoid Beta distribution BFloat16 issues on ROCm
        # Beta distribution has issues with torch._sample_dirichlet on mixed precision
        import torch
        # Sample from uniform [0, 1] as approximation for Beta(1, 1) = Uniform
        sample = torch.rand(batch_size, device=device, dtype=torch.bfloat16)
        noise_s = torch.tensor(self.config.noise_s, device=device, dtype=torch.bfloat16)
        return ((noise_s - sample) / noise_s).to(dtype=dtype)'''
        )
        patches_applied.append("uniform sampling")

    if patches_applied:
        with open(file1, 'w') as f:
            f.write(content)
        print(f"✓ Patched FlowmatchingActionHead: {', '.join(patches_applied)}")
    else:
        print("✓ FlowmatchingActionHead already patched")
else:
    print("✗ FlowmatchingActionHead not found")

# ============================================================================
# Patch 2: Fix GR00TN15 for transformers >= 4.46 compatibility
# ============================================================================
file2 = '/opt/lerobot/src/lerobot/policies/groot/groot_n1.py'
if os.path.exists(file2):
    with open(file2, 'r') as f:
        content = f.read()

    patches_applied = []

    # Check if already patched
    if 'all_tied_weights_keys' not in content or '_tied_weights_keys' not in content:
        lines = content.split('\n')

        # Find the GR00TN15 class definition
        for i, line in enumerate(lines):
            if 'class GR00TN15(PreTrainedModel):' in line:
                # Find the line after class docstring/supports_gradient_checkpointing
                for j in range(i + 1, min(len(lines), i + 30)):
                    stripped = lines[j].strip()
                    if not stripped or stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                    # Insert the _tied_weights_keys class attribute and all_tied_weights_keys property
                    property_code = [
                        '    # Class attribute for transformers >= 4.46 compatibility',
                        '    _tied_weights_keys = {}',
                        '',
                        '    @property',
                        '    def all_tied_weights_keys(self):',
                        '        """Compatibility property for transformers >= 4.46."""',
                        '        return self._tied_weights_keys',
                        '',
                    ]
                    for idx, prop_line in enumerate(property_code):
                        lines.insert(j + idx, prop_line)
                    patches_applied.append("all_tied_weights_keys property")
                    print("✓ Added _tied_weights_keys and all_tied_weights_keys to GR00TN15")
                    break
                break

        content = '\n'.join(lines)

    # Patch 3: Add self.post_init() at end of GR00TN15.__init__
    if 'self.post_init()' not in content:
        lines = content.split('\n')

        in_gr00tn15_init = False
        init_start = -1
        init_indent = ''

        for i, line in enumerate(lines):
            if 'class GR00TN15(PreTrainedModel):' in line:
                in_gr00tn15_init = False

            if 'def __init__(self, config' in line:
                context = '\n'.join(lines[max(0, i-50):i])
                if 'class GR00TN15' in context:
                    in_gr00tn15_init = True
                    init_start = i
                    init_indent = line[:line.index('def')]

            if in_gr00tn15_init and i > init_start:
                if line.strip().startswith('def ') and line.startswith(init_indent):
                    insert_pos = i
                    for k in range(i - 1, init_start, -1):
                        if lines[k].strip():
                            insert_pos = k + 1
                            break

                    post_init_code = [
                        '',
                        '        # Call post_init for transformers compatibility',
                        '        self.post_init()',
                    ]
                    for idx, code_line in enumerate(post_init_code):
                        lines.insert(insert_pos + idx, code_line)
                    patches_applied.append("post_init()")
                    print("✓ Added self.post_init() to GR00TN15.__init__")
                    break

        content = '\n'.join(lines)

    with open(file2, 'w') as f:
        f.write(content)

    if patches_applied:
        print(f"✓ GR00TN15 patches applied: {', '.join(patches_applied)}")
    else:
        print("✓ GR00TN15 already patched")
else:
    print("✗ GR00TN15 not found")

# ============================================================================
# Patch 4: Fix Eagle processor source file (gets copied to cache)
# ============================================================================
file3 = '/opt/lerobot/src/lerobot/policies/groot/eagle2_hg_model/processing_eagle2_5_vl.py'
if os.path.exists(file3):
    with open(file3, 'r') as f:
        content = f.read()

    if '_groot_get_num_tiles' not in content:
        # Add helper functions
        helper_code = '''

def _groot_get_num_tiles(pixel_values):
    """Get number of tiles from pixel_values, handling both list and tensor."""
    if isinstance(pixel_values, list):
        return len(pixel_values)
    return pixel_values.shape[0]

def _groot_ensure_tensor(v):
    """Convert list to tensor if needed."""
    import torch
    import numpy as np
    if isinstance(v, list):
        if len(v) == 0:
            return torch.tensor([])
        # Check if all elements are tensors
        if all(isinstance(t, torch.Tensor) for t in v):
            # If 3D (C, H, W), use stack to add batch dim
            # If 4D (B, C, H, W), use cat to merge batches
            if v[0].dim() == 3:
                return torch.stack(v, dim=0)
            else:
                return torch.cat(v, dim=0)
        # Check if all elements are numpy arrays
        if all(isinstance(t, np.ndarray) for t in v):
            if v[0].ndim == 3:
                return torch.from_numpy(np.stack(v, axis=0))
            else:
                return torch.from_numpy(np.concatenate(v, axis=0))
        # Handle list of lists (nested) - but NOT tuples
        if all(isinstance(t, list) for t in v):
            converted = [_groot_ensure_tensor(t) for t in v]
            if converted[0].dim() == 3:
                return torch.stack(converted, dim=0)
            else:
                return torch.cat(converted, dim=0)
        # Fallback: try to create tensor directly
        try:
            return torch.as_tensor(v)
        except (ValueError, TypeError):
            # Last resort: convert each element
            tensors = [torch.as_tensor(t) if not isinstance(t, torch.Tensor) else t for t in v]
            if tensors[0].dim() == 3:
                return torch.stack(tensors, dim=0)
            else:
                return torch.cat(tensors, dim=0)
    return v
'''

        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('class '):
                insert_idx = i
                break

        lines.insert(insert_idx, helper_code)
        content = '\n'.join(lines)

        # Apply all patches
        patches = [
            ('image_inputs["pixel_values"].shape[0]',
             '_groot_get_num_tiles(image_inputs["pixel_values"])'),
            ('video_inputs["pixel_values"].shape[0]',
             '_groot_get_num_tiles(video_inputs["pixel_values"])'),
            ('pixel_values = torch.cat([frame["pixel_values"] for frame in unified_frame_list])',
             'pixel_values = torch.cat([_groot_ensure_tensor(frame["pixel_values"]) for frame in unified_frame_list])'),
            ('image_sizes = torch.cat([frame["image_sizes"] for frame in unified_frame_list])',
             'image_sizes = torch.cat([_groot_ensure_tensor(frame["image_sizes"]) for frame in unified_frame_list])'),
        ]

        for old, new in patches:
            content = content.replace(old, new)

        with open(file3, 'w') as f:
            f.write(content)
        print("✓ Patched Eagle processor source file")
    else:
        print("✓ Eagle processor source file already patched")
else:
    print("✗ Eagle processor source file not found")

# ============================================================================
# Patch 5: Patch cached Eagle processor files (if they exist)
# ============================================================================
import glob

cache_patterns = [
    '/root/.cache/huggingface/modules/transformers_modules/*/processing_eagle2_5_vl.py',
    '/root/.cache/huggingface/modules/transformers_modules/*/*/processing_eagle2_5_vl.py',
]

patched_count = 0
for pattern in cache_patterns:
    for proc_file in glob.glob(pattern):
        try:
            with open(proc_file, 'r') as f:
                proc_content = f.read()

            if '_groot_get_num_tiles' in proc_content:
                print(f"✓ {proc_file} already patched")
                patched_count += 1
                continue

            lines = proc_content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('class '):
                    insert_idx = i
                    break

            # Use the same helper code as above
            helper_code = '''

def _groot_get_num_tiles(pixel_values):
    if isinstance(pixel_values, list):
        return len(pixel_values)
    return pixel_values.shape[0]

def _groot_ensure_tensor(v):
    """Convert list to tensor if needed."""
    import torch
    import numpy as np
    if isinstance(v, list):
        if len(v) == 0:
            return torch.tensor([])
        # Check if all elements are tensors
        if all(isinstance(t, torch.Tensor) for t in v):
            # If 3D (C, H, W), use stack to add batch dim
            # If 4D (B, C, H, W), use cat to merge batches
            if v[0].dim() == 3:
                return torch.stack(v, dim=0)
            else:
                return torch.cat(v, dim=0)
        # Check if all elements are numpy arrays
        if all(isinstance(t, np.ndarray) for t in v):
            if v[0].ndim == 3:
                return torch.from_numpy(np.stack(v, axis=0))
            else:
                return torch.from_numpy(np.concatenate(v, axis=0))
        # Handle list of lists (nested) - but NOT tuples (tuples are scalars)
        if all(isinstance(t, list) for t in v):
            converted = [_groot_ensure_tensor(t) for t in v]
            if converted[0].dim() == 3:
                return torch.stack(converted, dim=0)
            else:
                return torch.cat(converted, dim=0)
        # Try to create tensor directly
        try:
            return torch.as_tensor(v)
        except (ValueError, TypeError):
            # Last resort: convert each element
            tensors = [torch.as_tensor(t) if not isinstance(t, torch.Tensor) else t for t in v]
            if tensors[0].dim() == 3:
                return torch.stack(tensors, dim=0)
            else:
                return torch.cat(tensors, dim=0)
    return v
'''
            lines.insert(insert_idx, helper_code)
            proc_content = '\n'.join(lines)

            # Apply all patches
            cache_patches = [
                ('image_inputs["pixel_values"].shape[0]',
                 '_groot_get_num_tiles(image_inputs["pixel_values"])'),
                ('video_inputs["pixel_values"].shape[0]',
                 '_groot_get_num_tiles(video_inputs["pixel_values"])'),
                ('pixel_values = torch.cat([frame["pixel_values"] for frame in unified_frame_list])',
                 'pixel_values = torch.cat([_groot_ensure_tensor(frame["pixel_values"]) for frame in unified_frame_list])'),
                ('image_sizes = torch.cat([frame["image_sizes"] for frame in unified_frame_list])',
                 'image_sizes = torch.cat([_groot_ensure_tensor(frame["image_sizes"]) for frame in unified_frame_list])'),
            ]

            for old, new in cache_patches:
                proc_content = proc_content.replace(old, new)

            with open(proc_file, 'w') as f:
                f.write(proc_content)
            print(f"✓ Patched cached {proc_file}")
            patched_count += 1
        except Exception as e:
            print(f"✗ Failed to patch {proc_file}: {e}")

if patched_count == 0:
    print("Note: No cached Eagle processor files found (source already patched)")

print("=" * 60)
print("GR00T patches complete")
print("=" * 60)
