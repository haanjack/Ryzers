#!/usr/bin/env python3
"""Patch GR00T for compatibility with transformers >= 4.46."""

# Patch 1: Fix Beta distribution initialization
file1 = '/ryzers/lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py'
with open(file1, 'r') as f:
    content = f.read()

if 'validate_args=False' not in content:
    content = content.replace(
        'self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)',
        'self.beta_dist = Beta(torch.tensor(config.noise_beta_alpha), torch.tensor(config.noise_beta_beta), validate_args=False)'
    )
    with open(file1, 'w') as f:
        f.write(content)
    print("✓ Patched FlowmatchingActionHead (Beta distribution)")
else:
    print("✓ FlowmatchingActionHead already patched")

# Patch 2: Fix GR00TN15 for transformers >= 4.46 compatibility
file2 = '/ryzers/lerobot/src/lerobot/policies/groot/groot_n1.py'
with open(file2, 'r') as f:
    content = f.read()

patches_applied = []

# Check if already patched
if 'all_tied_weights_keys' in content and '_tied_weights_keys' in content:
    print("✓ GR00TN15 already has all_tied_weights_keys property")
else:
    lines = content.split('\n')

    # Find the GR00TN15 class definition
    for i, line in enumerate(lines):
        if 'class GR00TN15(PreTrainedModel):' in line:
            # Find the line after class docstring/supports_gradient_checkpointing
            # Look for the first actual class attribute or method
            for j in range(i + 1, min(len(lines), i + 30)):
                # Skip empty lines and comments
                stripped = lines[j].strip()
                if not stripped or stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                # Found first non-empty, non-comment line
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
if 'self.post_init()' in content:
    print("✓ GR00TN15.__init__ already has post_init() call")
else:
    lines = content.split('\n')

    # Find GR00TN15.__init__ method
    in_gr00tn15_init = False
    init_start = -1
    init_indent = ''

    for i, line in enumerate(lines):
        # Check if we're entering GR00TN15 class context
        if 'class GR00TN15(PreTrainedModel):' in line:
            in_gr00tn15_init = False  # Reset, we're looking for __init__ after this

        # Check if this is the __init__ method within GR00TN15
        if 'def __init__(self, config' in line:
            # Look back to confirm we're in GR00TN15 class
            context = '\n'.join(lines[max(0, i-50):i])
            if 'class GR00TN15' in context:
                in_gr00tn15_init = True
                init_start = i
                init_indent = line[:line.index('def')]

        # If we're in __init__, look for the end (next method at same or lower indentation)
        if in_gr00tn15_init and i > init_start:
            # Check if this is the start of a new method (not inside __init__)
            if line.strip().startswith('def ') and line.startswith(init_indent):
                # This is the start of a new method, insert post_init() before it
                # Find the last non-empty line before this method
                insert_pos = i
                for k in range(i - 1, init_start, -1):
                    if lines[k].strip():
                        insert_pos = k + 1
                        break

                # Insert post_init() call
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

# Write the patched content
with open(file2, 'w') as f:
    f.write(content)

if patches_applied:
    print(f"✓ All patches completed: {', '.join(patches_applied)}")
else:
    print("✓ No new patches needed (already patched)")
