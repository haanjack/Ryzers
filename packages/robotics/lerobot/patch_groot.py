#!/usr/bin/env python3
"""Patch GRoOT for compatibility."""

# Patch 1: Fix Beta distribution initialization
file1 = '/ryzers/lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py'
with open(file1, 'r') as f:
    content = f.read()
content = content.replace(
    'self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)',
    'self.beta_dist = Beta(torch.tensor(config.noise_beta_alpha), torch.tensor(config.noise_beta_beta), validate_args=False)'
)
with open(file1, 'w') as f:
    f.write(content)
print("✓ Patched FlowmatchingActionHead")

# Patch 2: Add all_tied_weights_keys property + post_init() to GR00TN15
file2 = '/ryzers/lerobot/src/lerobot/policies/groot/groot_n1.py'
with open(file2, 'r') as f:
    content = f.read()

# Add the property right after class definition
if 'class GR00TN15' in content and 'all_tied_weights_keys' not in content:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'class GR00TN15' in line:
            # Find the first method in this class
            for j in range(i + 1, min(len(lines), i + 50)):
                if lines[j].strip().startswith('def '):
                    # Insert property before first method
                    property_code = [
                        '',
                        '    @property',
                        '    def all_tied_weights_keys(self):',
                        '        """Compatibility for transformers >= 4.46"""',
                        '        return getattr(self, "_tied_weights_keys", {})',
                    ]
                    for idx, prop_line in enumerate(property_code):
                        lines.insert(j + idx, prop_line)
                    content = '\n'.join(lines)
                    print("✓ Added all_tied_weights_keys property to GR00TN15")
                    break
            break

# Patch 3: Add self.post_init() at end of __init__
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'def __init__(self, config' in line:
        # Check if in GR00TN15 context
        context = '\n'.join(lines[max(0, i-20):i])
        if 'class GR00TN15' in context:
            # Find end of __init__ (next method definition at class level)
            for j in range(i + 1, min(len(lines), i + 300)):
                if lines[j].startswith('    def ') and 'post_init' not in '\n'.join(lines[i:j]):
                    # Insert post_init before this next method
                    lines.insert(j, '        self.post_init()')
                    lines.insert(j, '        # Call post_init to initialize weights')
                    content = '\n'.join(lines)
                    print("✓ Added self.post_init() to GR00TN15.__init__")
                    break
            break

with open(file2, 'w') as f:
    f.write(content)

print("✓ All patches completed")

