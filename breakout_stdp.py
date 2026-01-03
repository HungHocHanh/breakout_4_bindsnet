# ================== CHU KỲ TRAIN & TEST (NÂNG CẤP) ==================
TRAIN_EPISODES = 100
TEST_EPISODES = 100

# Khởi tạo các biến tính trung bình cộng dồn
all_test_rewards_sum = 0.0
total_test_episodes_count = 0

episode = load_checkpoint_if_exists(environment_pipeline.network, CHECKPOINT_PATH)

while True:
    # --- GIAI ĐOẠN 1: TRAINING (100 Episodes) ---
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
        
        print(f"Train Episode {episode} | Reward: {total_reward}")
        with open(REWARD_LOG_FILE, "a") as f:
            f.write(f"{episode},{total_reward},train\n")
        
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(environment_pipeline.network, episode + 1, CHECKPOINT_PATH)
        episode += 1

    # --- GIAI ĐOẠN 2: TESTING (100 Episodes) ---
    print(f"\n>>> BẮT ĐẦU TESTING {TEST_EPISODES} EPISODES (TẮT LEARNING)...")
    environment_pipeline.network.learning = False
    
    current_cycle_test_rewards = []

    for t_ep in range(TEST_EPISODES):
        total_reward = 0.0
        environment_pipeline.reset_state_variables()
        done = False
        while not done:
            result = environment_pipeline.env_step()
            environment_pipeline.step(result)
            total_reward += result[1]
            done = result[2]
        
        current_cycle_test_rewards.append(total_reward)
        
        # Cập nhật trung bình cộng dồn (Cumulative Moving Average)
        all_test_rewards_sum += total_reward
        total_test_episodes_count += 1
        cumulative_avg = all_test_rewards_sum / total_test_episodes_count
        
        print(f"Test Ep {t_ep+1}/{TEST_EPISODES} | Reward: {total_reward} | Cumulative Avg: {cumulative_avg:.3f}")
        
        with open(REWARD_LOG_FILE, "a") as f:
            # Lưu thêm cả giá trị trung bình cộng dồn vào log để Hưng dễ vẽ biểu đồ
            f.write(f"{episode}_test_{t_ep},{total_reward},test,{cumulative_avg:.3f}\n")

    # Kết quả cuối mỗi đợt
    avg_this_cycle = sum(current_cycle_test_rewards) / TEST_EPISODES
    print(f"\n[KẾT QUẢ CHU KỲ]")
    print(f"- Trung bình chu kỳ này: {avg_this_cycle:.3f}")
    print(f"- TRUNG BÌNH CỘNG DỒN TẤT CẢ CÁC ĐỢT TEST: {all_test_rewards_sum / total_test_episodes_count:.3f}")
