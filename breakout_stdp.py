import os
import torch

from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax

# ================== CẤU HÌNH ==================
CHECKPOINT_PATH = "breakout_checkpoint.pth"
CHECKPOINT_INTERVAL = 100  # lưu mỗi 100 episode
# ==============================================

# ----------------- BUILD NETWORK -----------------
network = Network(dt=1.0)

# Layers
inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)
middle = IzhikevichNodes(n=100, traces=True)
out = IzhikevichNodes(n=4, rfrace=0, traces=True)

# Connections
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add to network
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Environment (nếu chạy SSH/không GUI thì để render_mode=None)
environment = GymEnvironment("BreakoutDeterministic-v4", render_mode="rgb_array")
environment.reset()

environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=None,
    render_interval=None,
)


# ----------------- CHECKPOINT -----------------
def save_checkpoint(network, episode, path=CHECKPOINT_PATH):
    state = {
        "episode": episode,
        "network_state": network.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint_if_exists(network, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return 0

    state = torch.load(path, map_location="cpu")
    state_dict = state.get("network_state", {})

    # Bỏ các key trạng thái nội bộ có .s (dễ lệch shape giữa các version)
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if ".s" in k:
            continue
        filtered_state_dict[k] = v

    # strict=False để bỏ qua missing/unexpected keys
    network.load_state_dict(filtered_state_dict, strict=False)

    return state.get("episode", 0)


# ----------------- TRAIN VÔ HẠN -----------------
environment_pipeline.network.learning = True

episode = load_checkpoint_if_exists(environment_pipeline.network, CHECKPOINT_PATH)

while True:
    total_reward = 0.0
    environment_pipeline.reset_state_variables()
    done = False

    while not done:
        result = environment_pipeline.env_step()
        environment_pipeline.step(result)

        reward = result[1]
        done = result[2]

        total_reward += reward

    # Chỉ in episode + reward
    print(f"Episode {episode}  reward {total_reward}")

    # Lưu checkpoint mỗi 100 episode
    if (episode + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(environment_pipeline.network, episode + 1, CHECKPOINT_PATH)

    episode += 1
