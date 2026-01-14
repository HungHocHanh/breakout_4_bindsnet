"""
TEST-ONLY Breakout + Izhikevich (BindsNet)
- Load weights.pth
- Run N test episodes
- Write separate CSV: runs_breakout_izh/test_results.csv
"""

import os
import csv
import time
import torch

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
TEST_EPISODES = 100
RUN_DIR = "runs_breakout_izh"
os.makedirs(RUN_DIR, exist_ok=True)

WEIGHTS_PTH = os.path.join(RUN_DIR, "weights.pth")
TEST_RESULTS_CSV = os.path.join(RUN_DIR, "test_results.csv")

PRINT_EVERY_EPISODE = 1  # 0 to disable


# =========================
# Save/Load utilities
# =========================
def load_weights(path: str, inpt_middle: Connection, middle_out: Connection, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    inpt_middle.w.data = payload["inpt_middle_w"].to(inpt_middle.w.device)
    middle_out.w.data = payload["middle_out_w"].to(middle_out.w.device)


def init_test_csv(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "episode",
                "total_reward",
                "steps",
                "v_out_mean",
            ])


def append_test_row(csv_path: str, episode: int, total_reward: float, steps: int, v_out_mean: float):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            int(time.time()),
            int(episode),
            float(total_reward),
            int(steps),
            float(v_out_mean),
        ])


# =========================
# Build network
# =========================
def build_network():
    network = Network(dt=1.0)

    inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)
    middle = IzhikevichNodes(n=100, excitatory=0.8, traces=True)
    out = IzhikevichNodes(n=4, refrac=0, traces=True)

    inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)

    middle_out = Connection(
        source=middle,
        target=out,
        wmin=0,
        wmax=1,
        update_rule=MSTDP,   # update_rule still ok, but we'll disable learning for test
        nu=1e-1,
        norm=0.5 * middle.n,
    )

    network.add_layer(inpt, name="Input Layer")
    network.add_layer(middle, name="Hidden Layer")
    network.add_layer(out, name="Output Layer")
    network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
    network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

    # Monitors
    m_out = Monitor(obj=out, state_vars=("s", "v"))
    network.add_monitor(monitor=m_out, name="OutMon")

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


def _tensor_mean_safe(x):
    try:
        return float(x.float().mean().item())
    except Exception:
        try:
            return float(x.mean())
        except Exception:
            return float("nan")


# =========================
# Test runner
# =========================
def run_test(pipeline: EnvironmentPipeline, episodes: int, csv_path: str):
    rewards = []
    steps_list = []
    vmeans = []

    # IMPORTANT: disable learning for test
    pipeline.network.learning = False

    for ep in range(episodes):
        total_reward = 0.0
        steps = 0
        done = False

        pipeline.reset_state_variables()

        while not done:
            obs, reward, done, info = pipeline.env_step()
            pipeline.step((obs, reward, done, info))
            total_reward += reward
            steps += 1

        v_out = pipeline.network.monitors["OutMon"].get("v")
        v_out_mean = _tensor_mean_safe(v_out)

        append_test_row(csv_path, ep, total_reward, steps, v_out_mean)

        rewards.append(total_reward)
        steps_list.append(steps)
        vmeans.append(v_out_mean)

        if PRINT_EVERY_EPISODE:
            print(f"[TEST] ep {ep} | reward={total_reward:.2f} | steps={steps} | v_out_mean={v_out_mean:.4f}")

    # Summary in console
    if rewards:
        avg_r = sum(rewards) / len(rewards)
        avg_s = sum(steps_list) / len(steps_list)
        avg_v = sum(vmeans) / len(vmeans)
        print("\n==== TEST SUMMARY ====")
        print(f"episodes: {episodes}")
        print(f"avg_reward: {avg_r:.3f}")
        print(f"avg_steps:  {avg_s:.2f}")
        print(f"avg_v_out_mean: {avg_v:.6f}")


def main():
    if not os.path.exists(WEIGHTS_PTH):
        raise FileNotFoundError(f"Không thấy weights file: {os.path.abspath(WEIGHTS_PTH)}")

    init_test_csv(TEST_RESULTS_CSV)
    print(f"📄 Test CSV: {os.path.abspath(TEST_RESULTS_CSV)}")

    network, inpt_middle, middle_out = build_network()
    pipeline = build_pipeline(network)

    load_weights(WEIGHTS_PTH, inpt_middle, middle_out)
    print(f"✅ Loaded weights from: {os.path.abspath(WEIGHTS_PTH)}")

    run_test(pipeline, TEST_EPISODES, TEST_RESULTS_CSV)

    print(f"\n✅ DONE. Test results -> {os.path.abspath(TEST_RESULTS_CSV)}")


if __name__ == "__main__":
    main()
