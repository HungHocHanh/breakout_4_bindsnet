import os
import torch
import numpy as np
from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_highest
# ----------------- CẤU HÌNH -----------------
CHECKPOINT_PATH = "Izhikevich_breakout_basic.pth"
REWARD_LOG_FILE = "full_reward_history.txt" 
TRAIN_EPISODES = 100
TEST_EPISODES = 100
CHECKPOINT_INTERVAL = 50 

N_HIDDEN = 100      
LEARNING_RATE = 5e-2

# ----------------- KHỞI TẠO MẠNG (BASIC SNN) -----------------
network = Network(dt=1.0)

# Input Layer: 5D theo yêu cầu [Batch, Time, Channel, Height, Width]
inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)

# Lớp Hidden (Izhikevich)
middle = IzhikevichNodes(n=N_HIDDEN, excitatory=0.9, traces=True)

# Lớp Output: 4 hành động (Trái, Phải, Đứng yên, Bắn)
out = IzhikevichNodes(n=4, excitatory=0.9, traces=True)

# Kết nối Input -> Middle: Cố định (Learning = False)
inpt_middle = Connection(
    source=inpt, target=middle, 
    wmin=0, wmax=1e-1,
    update_rule=None 
)

# Kết nối Middle -> Out: Lớp duy nhất thực hiện huấn luyện MSTDPET
middle_out = Connection(
    source=middle, target=out,
    wmin=0, wmax=1,
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
environment = GymEnvironment("BreakoutDeterministic-v4", render_mode="rgb_array")
environment.reset()

environment_pipeline = EnvironmentPipeline(
    network, environment,
    encoding=bernoulli,
    action_function=select_highest,
    output="Output Layer",
    time=100, history_length=1, delta=1,
    plot_interval=10**9, render_interval=10**9,
)

# ----------------- HÀM TIỆN ÍCH -----------------
def save_checkpoint(network, episode, path=CHECKPOINT_PATH):
    state = {"episode": episode, "network_state": network.state_dict()}
    torch.save(state, path)
    print(f"\n[HỆ THỐNG] Đã lưu checkpoint tại Episode {episode}")

def load_checkpoint_if_exists(network, path=CHECKPOINT_PATH):
    if not os.path.exists(path): return 0
    state = torch.load(path, map_location="cpu")
    net_state = state.get("network_state", {})
    model_state = network.state_dict()
    updated_state = {}
    for k in net_state:
        if k in model_state:
            if net_state[k].shape != model_state[k].shape:
                updated_state[k] = net_state[k].view(model_state[k].shape)
            else:
                updated_state[k] = net_state[k]
    network.load_state_dict(updated_state, strict=False)
    return state.get("episode", 0)

# Khởi tạo file log và viết tiêu đề nếu file mới
if not os.path.exists(REWARD_LOG_FILE):
    with open(REWARD_LOG_FILE, "w") as f:
        f.write("Episode,Mode,Reward,Cumulative_Avg,Avg_100,Improvement_Rate_Pct\n")

# ----------------- BIẾN THỐNG KÊ -----------------
all_rewards = []
reward_window = [] 
prev_avg_100 = 0.0

episode = load_checkpoint_if_exists(environment_pipeline.network)

# ----------------- VÒNG LẶP CHÍNH -----------------
try:
    while True:
        # --- GIAI ĐOẠN TRAINING ---
        print(f"\n>>> ĐANG TRAIN - BẮT ĐẦU TỪ EPISODE {episode}")
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
            
            # Cập nhật dữ liệu thống kê
            all_rewards.append(total_reward)
            reward_window.append(total_reward)
            if len(reward_window) > 100: reward_window.pop(0)
            
            cumul_avg = np.mean(all_rewards)
            avg_100 = np.mean(reward_window)
            
            # Tính tỉ lệ học (cải thiện) mỗi 100 episode
            improve_rate = 0.0
            if episode > 0 and episode % 100 == 0:
                if prev_avg_100 > 0:
                    improve_rate = ((avg_100 - prev_avg_100) / prev_avg_100) * 100
                prev_avg_100 = avg_100

            # Hiển thị tất cả kết quả ra màn hình
            print(f"Ep {episode} | Thưởng: {total_reward:.1f} | TB Cộng dồn: {cumul_avg:.2f} | TB 100 trận: {avg_100:.2f} | Cải thiện: {improve_rate:.1f}%")
            
            # Ghi vào file txt
            with open(REWARD_LOG_FILE, "a") as f:
                f.write(f"{episode},train,{total_reward},{cumul_avg:.3f},{avg_100:.3f},{improve_rate:.2f}\n")
            
            # Lưu checkpoint
            if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(environment_pipeline.network, episode + 1)
            
            episode += 1

        # --- GIAI ĐOẠN TESTING ---
        print(f"\n>>> ĐANG TEST {TEST_EPISODES} TRẬN (TẮT LEARNING)...")
        environment_pipeline.network.learning = False
        test_rewards = []
        
        for t_ep in range(TEST_EPISODES):
            total_reward = 0.0
            environment_pipeline.reset_state_variables()
            done = False
            while not done:
                result = environment_pipeline.env_step()
                environment_pipeline.step(result)
                total_reward += result[1]
                done = result[2]
            
            test_rewards.append(total_reward)
            print(f"Test trận {t_ep+1} | Thưởng: {total_reward}")
            
            with open(REWARD_LOG_FILE, "a") as f:
                f.write(f"{episode}_test,test,{total_reward},0,0,0\n")

        print(f"[*] Kết quả Test trung bình: {np.mean(test_rewards):.2f}")

except KeyboardInterrupt:
    print("\n[THÔNG BÁO] Đã dừng chương trình. Đang lưu trạng thái cuối cùng...")
    save_checkpoint(environment_pipeline.network, episode)
