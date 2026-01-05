import os
import torch
import time

from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax

# ================== CẤU HÌNH (PERSONALIZED FOR HƯNG) ==================
CHECKPOINT_PATH = "Izhikevich_exc_inh.pth"
REWARD_LOG_FILE = "reward_log_izhikevich_ex_inh.txt"
TRAIN_EPISODES = 100
TEST_EPISODES = 100
CHECKPOINT_INTERVAL = 50 # Lưu mỗi 50 episode để tránh làm chậm máy

# Các thông số đã tinh chỉnh để AI học ổn định hơn
N_HIDDEN = 500          # Tăng lên 500 giúp "hồ chứa" ghi nhớ tốt hơn
LEARNING_RATE = 5e-4    # Giảm nu để tránh bão hòa trọng số
# ======================================================================

# ----------------- BUILD NETWORK (RESERVOIR COMPUTING) -----------------
network = Network(dt=1.0)

# Lớp đầu vào: Nhận hình ảnh 80x80 từ game
inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)

# Lớp Reservoir: Hồ chứa 500 neuron Izhikevich
middle = IzhikevichNodes(n=N_HIDDEN, excitatory=0.8, traces=True)

# Lớp Output: 4 hành động (Trái, Phải, Đứng yên, Bắn)
out = IzhikevichNodes(n=4, excitatory=0.8, traces=True)

# Kết nối Input -> Middle: Cố định (đặc trưng của Reservoir Computing)
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)

# Kết nối Middle -> Out: Lớp duy nhất thực hiện huấn luyện MSTDP
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=LEARNING_RATE,
    norm=0.5 * middle.n,
)

network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Thiết lập môi trường game
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
    plot_interval=10**9,
    render_interval=10**9,
)

# ----------------- HÀM TIỆN ÍCH (UTILITIES) -----------------
def save_checkpoint(network, episode, path=CHECKPOINT_PATH):
    state = {
        "episode": episode,
        "network_state": network.state_dict(),
    }
    torch.save(state, path)
    print(f"--- Đã lưu checkpoint tại Episode {episode} ---")

def load_checkpoint_if_exists(network, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        print("Không tìm thấy checkpoint. Bắt đầu train mới.")
        return 0
    
    print(f"Đang tải checkpoint từ {path}...")
    try:
        # Load về CPU để tránh lỗi device
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("network_state", {})
        
        # --- BẮT ĐẦU PATCH SỬA LỖI DIMENSION ---
        keys_to_fix = [
            "Input Layer.s", 
            "Input Layer_to_Hidden Layer.source.s"
        ]
        
        for key in keys_to_fix:
            if key in state_dict:
                tensor = state_dict[key]
                # Nếu kích thước là 5 chiều [1, 1, 1, 80, 80], thêm 1 chiều vào để khớp model mới
                if len(tensor.shape) == 5:
                    print(f"-> Đang sửa kích thước cho tensor: {key}...")
                    state_dict[key] = tensor.unsqueeze(3)
        # --- KẾT THÚC PATCH ---

        network.load_state_dict(state_dict, strict=False)
        ep = checkpoint.get("episode", 0)
        print(f"Tải thành công! Tiếp tục từ Episode {ep}")
        return ep
        
    except Exception as e:
        print(f"Lỗi khi tải checkpoint: {e}")
        print("Sẽ bắt đầu train lại từ đầu.")
        return 0

# ----------------- CHU KỲ TRAIN & TEST (CUMULATIVE AVG) -----------------
all_test_rewards_sum = 0.0
total_test_episodes_count = 0

episode = load_checkpoint_if_exists(environment_pipeline.network, CHECKPOINT_PATH)

print("Hệ thống đã sẵn sàng. Bắt đầu chu kỳ huấn luyện...")

while True:
    # --- GIAI ĐOẠN 1: TRAINING (Bật Learning) ---
    print(f"\n>>> BẮT ĐẦU TRAINING {TRAIN_EPISODES} EPISODES...")
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
            f.write(f"{episode},{total_reward},train,0\n") # 0 là placeholder cho avg
            
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(environment_pipeline.network, episode + 1, CHECKPOINT_PATH)
        episode += 1

    # --- GIAI ĐOẠN 2: TESTING (Tắt Learning & Tính Trung bình cộng dồn) ---
    print(f"\n>>> BẮT ĐẦU TESTING {TEST_EPISODES} EPISODES...")
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
        
        # Công thức trung bình cộng dồn
        all_test_rewards_sum += total_reward
        total_test_episodes_count += 1
        cumulative_avg = all_test_rewards_sum / total_test_episodes_count
        
        print(f"Test {t_ep+1}/{TEST_EPISODES} | Reward: {total_reward} | Cumul. Avg: {cumulative_avg:.3f}")
        
        with open(REWARD_LOG_FILE, "a") as f:
            f.write(f"{episode}_test_{t_ep},{total_reward},test,{cumulative_avg:.3f}\n")

    print(f"\n[BÁO CÁO] Trung bình cộng dồn sau {total_test_episodes_count} trận test: {cumulative_avg:.3f}")
