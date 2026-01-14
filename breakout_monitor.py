"""
Breakout + Izhikevich (BindsNet) with:
- Monitors
- Per-episode CSV logging (train + test)
- Per-cycle summaries (train avg + test avg)
- .pth weight save/load

Schedule:
- 3 cycles
- each cycle: 100 episodes TRAIN + 100 episodes TEST
"""

import os
import csv
import time
import sys
import torch

# Nếu bạn dùng BindsNet local ở F:\code\bindsnet\
# sys.path.insert(0, r"F:\code")  # parent directory that contains "bindsnet/"

from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax


# =========================
# Config
# =========================
CYCLES = 3
TRAIN_EPISODES_PER_CYCLE = 100
TEST_EPISODES_PER_CYCLE = 100
SAVE_EVERY_EPISODES = 25   # during train, checkpoint every N episodes (optional)

# Optional: debug prints (not required for logging)
PRINT_EVERY_STEPS = 0      # set 0 to disable prints

RUN_DIR = "runs_breakout_izh"
os.makedirs(RUN_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RUN_DIR, "results.csv")
WEIGHTS_PTH = os.path.join(RUN_DIR, "weights.pth")


# =========================
# Save/Load utilities
# =========================
def save_weights(path: str, inpt_middle: Connection, middle_out: Connection):
    payload = {
        "inpt_middle_w": inpt_middle.w.detach().cpu(),
        "middle_out_w": middle_out.w.detach().cpu(),
    }
    torch.save(payload, path)


def load_weights(path: str, inpt_middle: Connection, middle_out: Connection, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    inpt_middle.w.data = payload["inpt_middle_w"].to(inpt_middle.w.device)
    middle_out.w.data = payload["middle_out_w"].to(middle_out.w.device)


def init_results_csv(csv_path: str):
    """
    Log requirements:
    - per episode: train & test: total_reward, steps, v_out_mean
    - per cycle summary: avg (reward, steps, v_out_mean) for train cycle and for test cycle
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "cycle",
                "mode",                # 'train'|'test'|'train_cycle_avg'|'test_cycle_avg'
                "episode_in_mode",     # 0..N-1 for per-episode rows; -1 for summary rows
                "total_reward",
                "steps",
                "v_out_mean",          # average of output layer membrane potential over the whole episode
            ])


def append_row(csv_path: str, cycle: int, mode: str, episode_in_mode: int,
               total_reward: float, steps: int, v_out_mean: float):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            int(time.time()),
            cycle,
            mode,
            int(episode_in_mode),
            float(total_reward),
            int(steps),
            float(v_out_mean),
        ])


# =========================
# Build network
# =========================
def build_network():
    network = Network(dt=1.0)

    # NOTE: This assumes your environment/pipeline produces 80x80 observations (after preprocessing).
    inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)

    middle = IzhikevichNodes(n=100, excitatory=0.8, traces=True)

    out = IzhikevichNodes(n=4, refrac=0, traces=True)

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

    network.add_layer(inpt, name="Input Layer")
    network.add_layer(middle, name="Hidden Layer")
    network.add_layer(out, name="Output Layer")
    network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
    network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

    # Monitors
    m_hidden = Monitor(obj=middle, state_vars=("s", "v"))
    m_out = Monitor(obj=out, state_vars=("s", "v"))
    m_w = Monitor(obj=middle_out, state_vars=("w",))
    network.add_monitor(monitor=m_hidden, name="HiddenMon")
    network.add_monitor(monitor=m_out, name="OutMon")
    network.add_monitor(monitor=m_w, name="W_HO_Mon")

    return network, inpt_middle, middle_out


# =========================
# Build env + pipeline
# =========================
def build_pipeline(network):
    environment = GymEnvironment("BreakoutDeterministic-v4", render_mode="rgb_array")
    environment.reset()

    pipeline = EnvironmentPipeline(
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
    return pipeline


# =========================
# Episode runner
# =========================
def _tensor_mean_safe(x):
    try:
        return float(x.float().mean().item())
    except Exception:
        try:
            return float(x.mean())
        except Exception:
            return float("nan")


def run_episodes_and_log(pipeline: EnvironmentPipeline,
                         mode: str,
                         cycle: int,
                         episodes: int,
                         inpt_middle: Connection,
                         middle_out: Connection):
    """
    Logs:
      - per episode row: total_reward, steps, v_out_mean
      - returns lists for summary averages
    """
    rewards = []
    steps_list = []
    vmeans = []

    for ep in range(episodes):
        total_reward = 0.0
        pipeline.reset_state_variables()
        done = False
        steps = 0

        while not done:
            obs, reward, done, info = pipeline.env_step()
            pipeline.step((obs, reward, done, info))

            total_reward += reward
            steps += 1

            if PRINT_EVERY_STEPS and (steps % PRINT_EVERY_STEPS == 0):
                s_out = pipeline.network.monitors["OutMon"].get("s")
                v_out = pipeline.network.monitors["OutMon"].get("v")
                spike_sum = float(s_out.sum())
                v_mean_debug = _tensor_mean_safe(v_out)
                print(f"[{mode} | cycle {cycle} | ep {ep} | step {steps}] "
                      f"reward_so_far={total_reward:.2f}, out_spikes_sum={spike_sum:.1f}, out_v_mean={v_mean_debug:.3f}")

        # Compute episode v_out_mean after episode end
        v_out = pipeline.network.monitors["OutMon"].get("v")
        v_out_mean = _tensor_mean_safe(v_out)

        # Per-episode log row
        append_row(
            RESULTS_CSV,
            cycle=cycle,
            mode=mode,
            episode_in_mode=ep,
            total_reward=total_reward,
            steps=steps,
            v_out_mean=v_out_mean,
        )

        print(f"[{mode.upper()}] cycle {cycle} ep {ep} | reward={total_reward:.2f} | steps={steps} | v_out_mean={v_out_mean:.4f}")

        rewards.append(total_reward)
        steps_list.append(steps)
        vmeans.append(v_out_mean)

        # Optional mid-cycle checkpoints during training
        if mode == "train" and SAVE_EVERY_EPISODES and ((ep + 1) % SAVE_EVERY_EPISODES == 0):
            save_weights(WEIGHTS_PTH, inpt_middle, middle_out)
            print(f"✅ checkpoint saved -> {WEIGHTS_PTH}")

    return rewards, steps_list, vmeans


def log_cycle_average(cycle: int, mode: str, rewards, steps_list, vmeans):
    """
    mode is 'train_cycle_avg' or 'test_cycle_avg'
    """
    if len(rewards) == 0:
        avg_reward = float("nan")
        avg_steps = -1
        avg_v = float("nan")
    else:
        avg_reward = sum(rewards) / len(rewards)
        avg_steps = int(round(sum(steps_list) / len(steps_list)))
        avg_v = sum(vmeans) / len(vmeans)

    append_row(
        RESULTS_CSV,
        cycle=cycle,
        mode=mode,
        episode_in_mode=-1,  # summary row marker
        total_reward=avg_reward,
        steps=avg_steps,
        v_out_mean=avg_v,
    )
    print(f"📊 [{mode}] cycle {cycle} | avg_reward={avg_reward:.2f} | avg_steps~={avg_steps} | avg_v_out_mean={avg_v:.4f}")


# =========================
# Main schedule: 3 cycles
# =========================
def main():
    init_results_csv(RESULTS_CSV)
    print(f"📄 Results CSV: {os.path.abspath(RESULTS_CSV)}")

    network, inpt_middle, middle_out = build_network()
    pipeline = build_pipeline(network)

    # Load existing weights if present
    if os.path.exists(WEIGHTS_PTH):
        load_weights(WEIGHTS_PTH, inpt_middle, middle_out)
        print(f"✅ loaded weights <- {os.path.abspath(WEIGHTS_PTH)}")
    else:
        print("ℹ️ no weights.pth found; start from scratch.")

    for cycle in range(1, CYCLES + 1):
        print(f"\n====================\nCYCLE {cycle}/{CYCLES}\n====================")

        # TRAIN
        pipeline.network.learning = True
        print(f"--- TRAIN {TRAIN_EPISODES_PER_CYCLE} episodes ---")
        train_rewards, train_steps, train_vmeans = run_episodes_and_log(
            pipeline, "train", cycle, TRAIN_EPISODES_PER_CYCLE, inpt_middle, middle_out
        )
        log_cycle_average(cycle, "train_cycle_avg", train_rewards, train_steps, train_vmeans)

        # Save after each train cycle
        save_weights(WEIGHTS_PTH, inpt_middle, middle_out)
        print(f"✅ saved weights after train cycle -> {os.path.abspath(WEIGHTS_PTH)}")

        # TEST
        pipeline.network.learning = False
        print(f"--- TEST {TEST_EPISODES_PER_CYCLE} episodes ---")
        test_rewards, test_steps, test_vmeans = run_episodes_and_log(
            pipeline, "test", cycle, TEST_EPISODES_PER_CYCLE, inpt_middle, middle_out
        )
        log_cycle_average(cycle, "test_cycle_avg", test_rewards, test_steps, test_vmeans)

    # Final save
    save_weights(WEIGHTS_PTH, inpt_middle, middle_out)
    print(f"\n✅ DONE. Final weights -> {os.path.abspath(WEIGHTS_PTH)}")
    print(f"📄 Final results CSV -> {os.path.abspath(RESULTS_CSV)}")


if __name__ == "__main__":
    main()
