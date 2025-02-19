# MLGYM: A New Framework and Benchmark for Advancing AI Research Agents

## Setup

1. Clone and install dependencies

    ```bash
    git clone git@github.com:fairinternal/MLGYM.git
    cd MLGYM
    conda create -n mlgym python=3.11
    pip install -e .
    ```

2. Create a `.env` file in the MLGYM directory (`MLGYM/.env`) to save all the environment variables including API keys.

    ```bash
    # Env variables
    MLGYM_CONFIG_ROOT="<path_to_MLGYM_root>/configs"
    MLGYM_TASK_CONFIG_DIR="<path_to_MLGYM_root>/configs/tasks"
    MLGYM_WORKSPACE_PATH="<path_to_MLGYM_root>/workspace"
    MLGYM_ENV_TIMEOUT=10000
    MLGYM_ACTION_SHORT_TIMEOUT=60
    MLGYM_ACTION_LONG_TIMEOUT=10000
    MLGYM_MODEL_MAX_RETRIES=3

    # API keys
    OPENAI_API_KEY=""
    ANTHROPIC_API_KEY=""
    ```

3. Follow the instructions [here](https://docs.docker.com/desktop/) to install docker. Choose the appropriate install command depending on your OS.

4. If you are working on a linux machine, please install the `nvidia-container-runtime`. This is required to start docker containers with GPU support.

    ```bash
    sudo dnf install -y nvidia-container-toolkit
    ```

5. **Please skip to step 9 if you don't want to use Podman**.
6. Follow the instructions [here](https://podman.io/get-started) to install Podman.
7. Start podman socket. The last command should return a running podman socket.

    ```bash
    systemctl --user enable podman.socket
    systemctl --user start podman.socket
    systemctl --user status podman.socket 
    ```

8. Redirect docker host to podman by exporting docker host env variable in bashrc or current session.

    ```bash
    export DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock
    ```

9. Pull the docker container

    ```bash
    docker pull aigym/mlgym-agent:latest
    ```

10. Test launching a docker/podman container with GPU support

    ```bash
    docker run -it --gpus all --name test aigym/mlgym-agent /bin/bash
    ls -la
    exit
    ```

11. Check that GPUs are available in the docker container using `nvidia-smi`.

12. Download data files form [this](https://drive.google.com/drive/folders/1jVnBRHbSinIpDbhrrVkcbu0akrbFFLrg?usp=drive_link) google drive link. Copy the contents of each folder from google drive to the corresponding data folder: `MLGYM/data/<TASK_NAME>/`.

### Troubleshooting

If you get Nvidia CDI spec errors on linux (eg. `Error: setting up CDI devices: unresolvable CDI devices nvidia.com/gpu=all`), run these additional commands.

```bash
sudo mkdir /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo touch /etc/containers/nodocker
```

## Run your first Agent

### Using Docker

```bash
python run.py \
  --container_type docker \
  --task_config_path tasks/battleOfSexes.yaml \
  --model litellm:claude-3-5-sonnet-20240620 \
  --per_instance_cost_limit 4.00 \
  --config_file configs/agents/default.yaml \
  --temp 1 \
  --gpus 0 \
  --gpus_per_agent 1 \
  --max_steps 50 \
  --aliases_file ./docker/aliases.sh
```

### Using Podman (on DevGPU)

```bash
python run.py \
  --container_type podman \
  --task_config_path tasks/battleOfSexes.yaml \
  --model litellm:claude-3-5-sonnet-20240620 \
  --per_instance_cost_limit 4.00 \
  --config_file configs/agents/default.yaml \
  --temp 1 \
  --gpus 0 \
  --gpus_per_agent 1 \
  --max_steps 50 \
  --aliases_file ./docker/aliases.sh
```

To see a full list of flags, please run `python run.py --help`.

## Trajectory Visualizer

MLGym provides a Web UI to inspect the agent trajectories.

```bash
streamlit run demo/trajectory_visualizer.py -- --trajectory_dir <absolute_path_to_trajectories>

# An example
streamlit run demo/trajectory_visualizer.py -- --trajectory_dir $HOME/Projects/MLGYM/trajectories/mlgym_bench_v0
```

To run the demo for MLGym, use the following command:

```bash
streamlit run demo/demo.py
```

## License

The majority of this code is licensed under CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license. However portions of the project are available under separate license terms: [SWE-Agent](https://github.com/SWE-agent/SWE-agent?tab=MIT-1-ov-file) and [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt?tab=MIT-1-ov-file) are released under MIT license; [Gymnax](https://github.com/RobertTLange/gymnax?tab=Apache-2.0-1-ov-file) and [Gymnax-blines](https://github.com/RobertTLange/gymnax-blines?tab=Apache-2.0-1-ov-file) are released under Apache 2.0 License.
