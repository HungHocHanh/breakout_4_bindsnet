import os
import torch
import time

from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_first_spike

# ----------------- CẤU HÌNH -----------------
CHECKPOINT_PATH = "Izhikevich_exc_inh4.pth"
REWARD_LOG_FILE = "reward_log_izhikevich_ex_inh4.txt" 
TRAIN_EPISODES = 100
TEST_EPISODES = 100
CHECKPOINT_INTERVAL = 50 

N_HIDDEN = 100      
LEARNING_RATE = 5e-2

# ----------------- KHỞI TẠO MẠNG -----------------
network = Network(dt=1.0)

# SỬ DỤNG 5 CHIỀU THEO YÊU CẦU: [Batch, Time, Channel, Height, Width]
inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)

# Lớp Reservoir: Hồ chứa neuron Izhikevich
middle = IzhikevichNodes(n=N_HIDDEN, excitatory=0.9, traces=True)

# Lớp Output: 4 hành động
out = IzhikevichNodes(n=4, excitatory=0.9, traces=True)

# Kết nối
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDPET,
    nu=LEARNING_RATE,
    norm=0.5 * middle.n,
)

network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# ----------------- THIẾT LẬP MÔI TRƯỜNG -----------------
environment = GymEnvironment("BreakoutDeterministic-v4", render_mode="human")
environment.reset()

environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_first_spike,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=10**9,
    render_interval=10**9,
)

# ----------------- HÀM LOAD/SAVE FIX LỖI 5D/6D -----------------
def save_checkpoint(network, episode, path=CHECKPOINT_PATH):
    state = {
        "episode": episode,
        "network_state": network.state_dict(),
    }
    torch.save(state, path)
    print(f"--- Đã lưu checkpoint tại Episode {episode} ---")

def load_checkpoint_if_exists(network, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return 0
    
    state = torch.load(path, map_location="cpu")
    network_state = state.get("network_state", {})
    model_state = network.state_dict()
    
    updated_state = {}
    for key in network_state:
        if key in model_state:
            # Nếu tensor trong checkpoint (5D) khác với model hiện tại (6D)
            if network_state[key].shape != model_state[key].shape:
                print(f"Fixing {key}: {network_state[key].shape} -> {model_state[key].shape}")
                # Ép kiểu dữ liệu cũ về đúng định dạng model mong muốn
                updated_state[key] = network_state[key].view(model_state[key].shape)
            else:
                updated_state[key] = network_state[key]
    
    network.load_state_dict(updated_state, strict=False)
    return state.get("episode", 0)

# ----------------- VÒNG LẶP CHÍNH -----------------
all_test_rewards_sum = 0.0
total_test_episodes_count = 0

episode = load_checkpoint_if_exists(environment_pipeline.network, CHECKPOINT_PATH)

try:
    while True:
        # TRAINING
        environment_pipeline.network.learning = True
        for _ in range(TRAIN_EPISODES):
            total_reward = 0.0
            environment_pipeline.reset_state_variables()
            done = False
            while not done:
                result = environment_pipeline.env_step()
                environment_pipeline.step(result)
                total_reward += result[1]
                done = result[2]
            
            print(f"Train Ep {episode} | Reward: {total_reward}")
            with open(REWARD_LOG_FILE, "a") as f:
                f.write(f"{episode},{total_reward},train,0\n")
            
            if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(environment_pipeline.network, episode + 1, CHECKPOINT_PATH)
            episode += 1

        # TESTING
        environment_pipeline.network.learning = False
        for t_ep in range(TEST_EPISODES):
            total_reward = 0.0
            environment_pipeline.reset_state_variables()
            done = False
            while not done:
                result = environment_pipeline.env_step()
                environment_pipeline.step(result)
                total_reward += result[1]
                done = result[2]
            
            all_test_rewards_sum += total_reward
            total_test_episodes_count += 1
            cumulative_avg = all_test_rewards_sum / total_test_episodes_count
            print(f"Test | Reward: {total_reward} | Avg: {cumulative_avg:.3f}")

except KeyboardInterrupt:
    print("Dừng chương trình.")
