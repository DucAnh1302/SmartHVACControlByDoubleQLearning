"""
Maximization Bias Toy Experiment (Sutton & Barto, Example 6.7).
Demonstrates the theoretical overestimation bias of Q-Learning
and how Double Q-Learning resolves it on a 2-state MDP.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def run_maximization_bias_toy(n_runs: int = 2000, n_episodes: int = 300, output_dir: Path = None):
    """
    Simulates Sutton & Barto Example 6.7:
    State A has 2 actions:
      - 0: 'right' -> transitions to terminal with reward 0.
      - 1: 'left' -> transitions to State B with reward 0.
    State B has 10 actions:
      - Each transitions to terminal with reward N(-0.1, 1.0).
    Expected return of any action from B is -0.1 < 0.
    Optimal choice at A is 'right' (return 0).
    """
    alpha = 0.1
    epsilon = 0.1
    gamma = 1.0

    ql_left_counts = np.zeros(n_episodes, dtype=np.float64)
    dql_left_counts = np.zeros(n_episodes, dtype=np.float64)

    # 1. Q-Learning Runs
    for run in range(n_runs):
        rng = np.random.default_rng(run)
        q_a = np.zeros(2)   # State A
        q_b = np.zeros(10)  # State B

        for ep in range(n_episodes):
            # Select action at A
            if rng.random() < epsilon:
                action_a = int(rng.integers(0, 2))
            else:
                max_q = np.max(q_a)
                best = np.where(np.isclose(q_a, max_q))[0]
                action_a = int(rng.choice(best))

            if action_a == 1:  # Left -> B
                ql_left_counts[ep] += 1
                if rng.random() < epsilon:
                    action_b = int(rng.integers(0, 10))
                else:
                    max_b = np.max(q_b)
                    best_b = np.where(np.isclose(q_b, max_b))[0]
                    action_b = int(rng.choice(best_b))

                r_b = float(rng.normal(-0.1, 1.0))
                # Update Q(B, a)
                q_b[action_b] += alpha * (r_b + 0.0 - q_b[action_b])
                # Update Q(A, left) with max_a Q(B, a)
                q_a[1] += alpha * (0.0 + gamma * np.max(q_b) - q_a[1])
            else:  # Right -> Terminal
                q_a[0] += alpha * (0.0 - q_a[0])

    # 2. Double Q-Learning Runs
    for run in range(n_runs):
        rng = np.random.default_rng(run + 50000)
        qa_a = np.zeros(2); qb_a = np.zeros(2)
        qa_b = np.zeros(10); qb_b = np.zeros(10)

        for ep in range(n_episodes):
            q_sum_a = (qa_a + qb_a) / 2.0
            if rng.random() < epsilon:
                action_a = int(rng.integers(0, 2))
            else:
                max_q = np.max(q_sum_a)
                best = np.where(np.isclose(q_sum_a, max_q))[0]
                action_a = int(rng.choice(best))

            if action_a == 1:  # Left -> B
                dql_left_counts[ep] += 1
                q_sum_b = (qa_b + qb_b) / 2.0
                if rng.random() < epsilon:
                    action_b = int(rng.integers(0, 10))
                else:
                    max_b = np.max(q_sum_b)
                    best_b = np.where(np.isclose(q_sum_b, max_b))[0]
                    action_b = int(rng.choice(best_b))

                r_b = float(rng.normal(-0.1, 1.0))

                if rng.random() < 0.5:
                    # Update QA
                    qa_b[action_b] += alpha * (r_b + 0.0 - qa_b[action_b])
                    best_b_star = int(np.argmax(qa_b))
                    qa_a[1] += alpha * (0.0 + gamma * qb_b[best_b_star] - qa_a[1])
                else:
                    # Update QB
                    qb_b[action_b] += alpha * (r_b + 0.0 - qb_b[action_b])
                    best_a_star = int(np.argmax(qb_b))
                    qb_a[1] += alpha * (0.0 + gamma * qa_b[best_a_star] - qb_a[1])
            else:  # Right -> Terminal
                if rng.random() < 0.5:
                    qa_a[0] += alpha * (0.0 - qa_a[0])
                else:
                    qb_a[0] += alpha * (0.0 - qb_a[0])

    ql_pct = (ql_left_counts / n_runs) * 100.0
    dql_pct = (dql_left_counts / n_runs) * 100.0

    ql_mean = float(np.mean(ql_pct))
    dql_mean = float(np.mean(dql_pct))

    print(f"[Toy MaxBias] Q-Learning left %: {ql_mean:.2f}% | Double Q-Learning left %: {dql_mean:.2f}%")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(range(1, n_episodes + 1), ql_pct, label="Single Q-Learning", color="#D9534F", lw=2)
        ax.plot(range(1, n_episodes + 1), dql_pct, label="Double Q-Learning", color="#0B6E70", lw=2)
        ax.axhline(y=5.0, color="gray", linestyle="--", label="Optimal Exploration Level (ε/2 = 5%)")

        ax.set_title("Maximization Bias in Sutton & Barto Example 6.7 (Toy MDP)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episodes", fontsize=11)
        ax.set_ylabel("% Actions 'Left' (Sub-optimal) from State A", fontsize=11)
        ax.set_ylim(-2, 100)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="white", loc="upper right")

        fig_path = output_dir / "maxbias_toy.png"
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"[Toy MaxBias] Saved figure to: {fig_path}")

    return {
        "ql_mean_bias": ql_mean,
        "dql_mean_bias": dql_mean,
        "ql_left_pct_curve": ql_pct.tolist(),
        "dql_left_pct_curve": dql_pct.tolist()
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "result" / "final"
    run_maximization_bias_toy(n_runs=2000, n_episodes=300, output_dir=out_dir)
