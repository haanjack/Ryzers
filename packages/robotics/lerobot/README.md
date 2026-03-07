# LeRobot

Contains everything you need to build & run a ROCm-enabled LeRobot container with support for GR00T fine-tuning and multi-GPU distributed training.

## Build & Run the Container

Ryzers automatically detects your container runtime (Podman or Docker) and uses the appropriate one.

```bash
ryzers build lerobot
ryzers run
```

This will run a quick training test to verify the installation. By default, it uses the ACT policy on a single GPU.

### Policy Selection

Choose between ACT (faster, default) and GR00T policies:

```bash
# ACT policy (default)
ryzers run --policy act

# GR00T policy
ryzers run --policy groot
```

### Container Runtime

Ryzers automatically detects and uses:
- **Podman** (preferred if available)
- **Docker** (fallback)

No configuration needed - the runtime is detected at build and run time.

## Distributed Training

Scale training across multiple AMD GPUs using ROCm's RCCL backend.

### Single-GPU (Default)

```bash
ryzers run
```

### Multi-GPU Auto-Detect

Automatically uses all available GPUs:

```bash
ryzers run --distributed
```

### Force Specific GPU Count

```bash
# Use exactly 4 GPUs
ryzers run --distributed --nproc-per-node 4
```

### Force Single-GPU Mode

Disable distributed training even on multi-GPU systems:

```bash
ryzers run --no-distributed
```

### Multi-Node Training

For training across multiple nodes, run on each node with appropriate settings:

```bash
# Node 0 (master)
ryzers run --distributed --nnodes 2 --nproc-per-node 4 --node-rank 0 --master-addr 10.0.0.1

# Node 1 (worker)
ryzers run --distributed --nnodes 2 --nproc-per-node 4 --node-rank 1 --master-addr 10.0.0.1
```

### Distributed Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--distributed` | Enable distributed training | auto-detect |
| `--no-distributed` | Force single-GPU mode | - |
| `--nproc-per-node` | GPUs per node | all available |
| `--nnodes` | Number of nodes | 1 |
| `--node-rank` | Node rank (0=master) | 0 |
| `--master-addr` | Master node address | localhost |
| `--master-port` | Master node port | 29500 |
| `--policy` | Policy type (act/groot) | act |

### Environment Variables

The following environment variables can be set for distributed training:

| Variable | Description |
|----------|-------------|
| `DISTRIBUTED` | Set to `0` to force single-GPU |
| `NPROC_PER_NODE` | Number of GPUs to use |
| `POLICY_TYPE` | Policy type (act or groot) |

## Optimization Options

### Mixed Precision (AMP)

Enable automatic mixed precision for faster training with BFloat16:

```bash
ryzers run -- "--policy groot --training-steps 100 --use-amp"
```

**Note:** GR00T uses BFloat16 internally, so GradScaler is automatically disabled (BFloat16 has the same dynamic range as Float32).

### torch.compile

torch.compile can significantly speed up training, but has compatibility issues with GR00T's Eagle vision model. Use selective compilation to compile only the trainable action head:

```bash
# GR00T with selective compilation (recommended)
ryzers run -- "--policy groot --training-steps 100 --use-amp --use-compile --compile-disable-eagle"

# ACT with full compilation (works fine)
ryzers run -- "--policy act --training-steps 100 --use-amp --use-compile"
```

### Compile Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-compile` | Enable torch.compile optimization | off |
| `--compile-mode` | Compilation mode: `default`, `reduce-overhead`, `max-autotune` | `reduce-overhead` |
| `--compile-fullgraph` | Require full graph compilation | off |
| `--compile-backend` | Backend for torch.compile | `inductor` |
| `--compile-dynamic` | Enable dynamic shape support | off |
| `--compile-disable-eagle` | Compile only action head (GR00T only) | off |

### Why `--compile-disable-eagle` for GR00T?

GR00T's Eagle vision backbone has dynamic control flow that causes issues with torch.compile's graph tracing:
- `hidden_states` tuple access fails during Dynamo tracing
- Selective compilation avoids this by only compiling the DiT action head
- The frozen backbone remains uncompiled but still functional

## Training and Controlling Robot Arms

For this example we use the LeRobot [SO-101](https://huggingface.co/docs/lerobot/en/so101) leader and follower arms, however you can easily swap them with a different robot arm type in the following scripts.

### 1. Reference & Config
- **Guide:** Hugging Face "Imitation Learning on Real-World Robots"
  <https://huggingface.co/docs/lerobot/en/il_robots>
- **`config.yaml`:**
  - Pay attention to the TODO items - add your own `HF_TOKEN` from Hugging Face, and map your robot and video devices accordingly. Step 2. makes this simpler and more reproducible, but is optional.

**Important:** you will likely need read/write permissions enabled for the serial devices before you start the docker.

```bash
sudo chmod 666 /dev/ttyACM*
```

Once you've updated your config make sure to rebuild the lerobot docker.

```bash
ryzers build lerobot
```

After initial setup, steps 3-5 should be run inside an interactive shell of the docker container:
```
ryzers run bash
```

---

### 2. USB device mapping (optional)

Your serial and video devices may change indexes in `/dev` between sessions or when you re-plug them. To save the hassle of trying to figure out the device index every time we can map them to consistent named pointers by their serial IDs.

#### USB-serial mapping

1. **Record serial IDs**
   ```bash
   ls -l /dev/serial/by-id/
   ```
2. **Create or edit** `99-usb-serial.rules` with your favorite editor:
   ```
   sudo vim /etc/udev/rules.d/99-usb-serial.rules
   ```

   Add the following:
   ```ini
   SUBSYSTEM=="tty", ATTRS{serial}=="<leader-serial>",   SYMLINK+="ttyACM_leader"
   SUBSYSTEM=="tty", ATTRS{serial}=="<follower-serial>", SYMLINK+="ttyACM_follower"
   ```
   Replace `<leader-serial>` and `<follower-serial>` with the values from step 1.
3. **Reload rules & trigger udev**
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

#### USB-video mapping

1. **List webcam details**
   ```bash
   for dev in /dev/video*; do
       echo "=== $dev ==="
       udevadm info --query=all --name=$dev | grep -E "ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL|DEVPATH"
   done
   ```
2. **Create or edit** `99-usb-video.rules` with your favorite editor.
   ```
   sudo vim /etc/udev/rules.d/99-usb-video.rules
   ```
   You can use `ID_SERIAL_SHORT` from step 1. as the serial number for each device. Give the symlink any name that's meaningful to you.
   ```ini
   KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", ATTRS{serial}=="<cam1-serial-short>", SYMLINK+="webcam_top"
   KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", ATTRS{serial}=="<cam2-serial-short>", SYMLINK+="webcam_front"
   ```
3. **Install & update devices**
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

Your USB serial ports and cameras should mount exactly as specified. E.g. the robot and teleop ports will be available as `/dev/ttyACM_leader` and `/dev/ttyACM_follower`.

---

### 3. Collecting dataset

In order to train the policy we will need data for our specific embodiment. We use an SO-101 setup with two C270 USB webcams, however you can use more or less cameras.

#### Teleoperation (optional)

Before starting any data collection tasks you can make sure your setup works by running `lerobot-teleoperate`. We will re-use a lot of these parameters in `lerobot-record`.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_follower \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_leader \
    --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/webcam_top, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/webcam_front, width: 640, height: 480, fps: 30}}" \
    --display_data=true
```

#### Record a dataset

We record 30 episodes of manually placing a green cube into a mug. Make sure to set `robot.cameras` with the resolution and index according to your setup. Adjust dataset parameters like number of episodes or durations as needed for your task.

Set `dataset.push_to_hub=True` if you want to upload the dataset online to your HuggingFace hub.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_follower \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_leader \
    --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/webcam_top, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/webcam_front, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/cube_test_dataset \
    --dataset.num_episodes=30 \
    --dataset.single_task="place green cube in mug" \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=False \
    --play_sound=False
```

You can later visualize individual episodes in your collected dataset using `lerobot-dataset-viz`. You can do this with local cached dataset - no need to upload anything online.
```bash
lerobot-dataset-viz \
    --repo-id=${HF_USER}/cube_test_dataset \
    --episode-index=0
```

<img src="images/lerobot_record_dataset_so101.gif">

### 4. Train a policy

Using the collected dataset you can use it to train a policy like [ACT](https://github.com/tonyzhaozh/act) or [GR00T](https://huggingface.co/nvidia/GR00T-Vision-Language-Action). Depending on your dataset size you should be able to train a small policy like ACT within a couple hours on the Strix Halo iGPU. Adjust training parameters as required for your policy and dataset.

#### Single-GPU Training

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/cube_test_dataset \
    --policy.type=act \
    --output_dir=/ryzers/mounted/outputs/train/place_cube_act \
    --job_name=place_cube \
    --policy.device=cuda \
    --policy.repo_id=${HF_USER}/place_cube_act \
    --steps=20000 \
    --save_freq=2000
```

#### Multi-GPU Distributed Training

For larger datasets or faster training, use distributed training across multiple GPUs:

```bash
# Inside the container, or use ryzers run --distributed
torchrun --nproc_per_node=4 \
    -m lerobot.scripts.train \
    --dataset.repo_id=${HF_USER}/cube_test_dataset \
    --policy.type=act \
    --policy.device=cuda \
    --training.batch_size=64 \
    --steps=20000
```

Or use the convenience script:
```bash
/ryzers/distributed_train.sh --nproc-per-node 4 -- \
    --dataset.repo_id=${HF_USER}/cube_test_dataset \
    --policy.type=act \
    --policy.device=cuda
```

### 5. Run inference

To deploy the model we re-use the `lerobot-record` command omitting training settings and with a `policy.path` parameter set. **Note:** the `dataset.repo_id` parameter should start with the word `eval`.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_follower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/webcam_top, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/webcam_front, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/eval_place_cube_act \
    --dataset.single_task="place green cube in mug" \
    --policy.path=/ryzers/mounted/outputs/train/place_cube_act/checkpoints/last/pretrained_model/ \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=20 \
    --dataset.push_to_hub=False \
    --play_sound=False
```

Observe your arm doing its tasks autonomously!

<img src="images/evaluating_act_policy_so101.gif">

Now you are well equipped to run the LeRobot stack on your Strix Halo machine. Try tackling different tasks, collect more data or explore other policies - have fun!

## HPC Cluster Training

For training on AMD HPC clusters, use the provided SLURM job templates in `hpc_cluster_setup.sh`. This script creates several job templates:

- `job.train_pusht_example` - Single-GPU training
- `job.train_pusht_distributed_4gpu` - 4 GPUs on 1 node
- `job.train_pusht_multinode` - 8 GPUs across 2 nodes
- `job.train_groot_example` - Single-GPU GR00T training
- `job.train_groot_distributed_4gpu` - 4 GPUs GR00T training

## Troubleshooting

### Arms out of sync

If there's a big difference between movements of the leader and follower you can re-run calibration:
```bash
lerobot-calibrate  --teleop.type=so101_leader     --teleop.port=/dev/ttyACM_leader     --teleop.id=my_awesome_leader_arm
lerobot-calibrate  --robot.type=so101_follower    --robot.port=/dev/ttyACM_follower    --robot.id=my_awesome_follower_arm
```

### Timeouts

If you run into motor bus timeout issues, you may need to increase the number of communication retries, here's a oneliner to make that change from an interactive session:
```bash
find . -type f -name "*.py" -exec sed -i.bak 's/num_retry: int = 0/num_retry: int = 10/g' {} +
```

### Camera issues

If the 2nd camera doesn't connect and you see something like:
```bash
RuntimeError: OpenCVCamera(/dev/webcam_top) read failed (status=False).
```

It might be a USB controller bandwidth limitation - try connecting cameras to different usb controllers or reduce resolution/fps.

### Distributed training issues

If distributed training fails to start:
1. Ensure RCCL is installed (`librccl-dev` on Ubuntu)
2. Check that all GPUs are visible (`rocm-smi`)
3. Verify network connectivity between nodes for multi-node training
4. Check `MASTER_ADDR` and `MASTER_PORT` are accessible from all nodes
