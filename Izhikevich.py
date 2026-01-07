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
CHECKPOINT_PATH = "Izhikevich_breakout_final.pth"
REWARD_LOG_FILE = "test_performance_history.txt" # FILE LƯU THƯỞNG
TRAIN_EPISODES = 100
TEST_EPISODES = 100
CHECKPOINT_INTERVAL = 50 

N_HIDDEN = 100      
LEARNING_RATE = 5e-2

# ----------------- KHỞI TẠO MẠNG -----------------
network = Network(dt=1.0)
# Giữ nguyên 5 chiều: [Batch, Time, Channel, Height, Width]
inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)
middle = IzhikevichNodes(n=N_HIDDEN, excitatory=0.9, traces=True)
out = IzhikevichNodes(n=4, excitatory=0.9, traces=True)

inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1, update_rule=None)
middle_out = Connection(
    source=middle, target=out, wmin=0, wmax=1,
    update_rule=MSTDPET, nu=LEARNING_RATE, norm=0.5 * middle.n,
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
    network, environment,
    encoding=bernoulli,
    action_function=select_highest, # Đánh giá chính xác bằng cách đếm xung
    output="Output Layer",
    time=100, history_length=1, delta=1,
    plot_interval=10**9, render_interval=10**9,
)

# ----------------- HÀM TIỆN ÍCH (LƯU/LOAD) -----------------
def save_checkpoint(network, episode, path=CHECKPOINT_PATH):
    state = {"episode": episode, "network_state": network.state_dict()}
    torch.save(state, path)
    print(f"\n[HỆ THỐNG] Đã lưu Checkpoint tại Ep {episode}")

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

# Khởi tạo file thưởng và ghi tiêu đề
if not os.path.exists(REWARD_LOG_FILE):
    with open(REWARD_LOG_FILE, "w") as f:
        f.write("Cycle,Test_Total_Ep,Reward,Cumul_Avg_Test,Avg_100_Test,Improve_Rate_Pct\n")

# ----------------- BIẾN THỐNG KÊ (DÀNH RIÊNG CHO TEST) -----------------
test_rewards_history = []
test_window_100 = [] 
prev_avg_100 = 0.0
cycle_count = 0

episode = load_checkpoint_if_exists(environment_pipeline.network)

# ----------------- CHU KỲ CHẠY CHÍNH -----------------
try:
    while True:
        cycle_count += 1
        
        # --- 1. GIAI ĐOẠN TRAIN (HỌC) ---
        print(f"\n>>> CHU KỲ {cycle_count}: ĐANG HUẤN LUYỆN {TRAIN_EPISODES} TRẬN...")
        environment_pipeline.network.learning = True
        
        for _ in range(TRAIN_EPISODES):
            environment_pipeline.reset_state_variables()
            done = False
            while not done:
                result = environment_pipeline.env_step()
                environment_pipeline.step(result)
                done = result[2]
            
            # Lưu checkpoint mỗi 50 trận train
            if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(environment_pipeline.network, episode + 1)
            episode += 1

        # --- 2. GIAI ĐOẠN TEST (ĐÁNH GIÁ & GHI FILE) ---
        print(f"\n>>> CHU KỲ {cycle_count}: ĐANG TEST ĐÁNH GIÁ {TEST_EPISODES} TRẬN...")
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
            
            # CẬP NHẬT THỐNG KÊ TEST
            test_rewards_history.append(total_reward)
            test_window_100.append(total_reward)
            if len(test_window_100) > 100: test_window_100.pop(0)
            
            cumul_avg = np.mean(test_rewards_history)
            avg_100 = np.mean(test_window_100)
            
            # Tính tỉ lệ cải thiện mỗi khi đủ 100 trận test
            improve_rate = 0.0
            if len(test_rewards_history) % 100 == 0:
                if prev_avg_100 > 0:
                    improve_rate = ((avg_100 - prev_avg_100) / prev_avg_100) * 100
                prev_avg_100 = avg_100

            # Hiển thị kết quả Test
            test_count = len(test_rewards_history)
            print(f"Test Ep {test_count} | Thưởng: {total_reward:.1f} | TB Test: {cumul_avg:.2f} | TB 100: {avg_100:.2f} | Học: {improve_rate:.1f}%")
            
            # GHI VÀO FILE THƯỞNG (QUAN TRỌNG)
            with open(REWARD_LOG_FILE, "a") as f:
                f.write(f"{cycle_count},{test_count},{total_reward},{cumul_avg:.3f},{avg_100:.3f},{improve_rate:.2f}\n")

except KeyboardInterrupt:
    print("\n[DỪNG] Đã lưu toàn bộ lịch sử thưởng vào file. Đang lưu checkpoint cuối...")
    save_checkpoint(environment_pipeline.network, episode)
