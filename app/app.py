"""
SmartHVACControlByDoubleQLearning - Streamlit Demonstration Application.
An interactive web application demonstrating intelligent HVAC control with Double Q-Learning.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.environment.hvac_env import SmartHVACEnv, ACTION_POWER_MAP, ACTION_NAMES
from src.agent.q_learning_agent import QLearningAgent
from src.agent.double_q_learning_agent import DoubleQLearningAgent
from src.agent.rule_based_agent import RuleBasedAgent
from src.data.preprocess import index_to_state
from src.reward.reward_function import HVACRewardFunction


# Page configuration
st.set_page_config(
    page_title="Smart HVAC Control | Double Q-Learning",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_processed_data():
    """Load preprocessed CU-BEMS hourly dataset."""
    data_path = project_root / "data" / "processed" / "cubems_zone1_hourly.csv"
    if not data_path.exists():
        st.error(f"Processed dataset not found at: {data_path}")
        return None
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_resource
def load_trained_models():
    """Load serialized models from model directory."""
    model_dir = project_root / "model"
    ql_path = model_dir / "q_learning_model.pkl"
    dql_path = model_dir / "double_q_learning_model.pkl"

    models = {}
    if ql_path.exists():
        models["Q-Learning"] = QLearningAgent.load(str(ql_path))
    if dql_path.exists():
        models["Double Q-Learning"] = DoubleQLearningAgent.load(str(dql_path))

    # Add Rule-based baseline
    models["Rule-Based"] = RuleBasedAgent(n_states=45, n_actions=4)
    return models


# Load data and models
df_data = load_processed_data()
models_dict = load_trained_models()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/air-conditioner.png", width=72)
st.sidebar.title("Smart HVAC Control")
st.sidebar.markdown("**Thuật toán:** Double Q-Learning  \n**Bộ dữ liệu:** CU-BEMS (Chulalongkorn Univ)")

page = st.sidebar.radio(
    "Chọn chức năng:",
    [
        "🎮 Trình Giả lập Điều khiển Thông minh",
        "📊 Khám phá Dữ liệu CU-BEMS",
        "⚖️ So sánh Hiệu năng (Benchmark)",
        "🧠 Trực quan hóa Bảng Q-Table & Policy"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Cấu hình Tham số Môi trường")
initial_temp_slider = st.sidebar.slider("Nhiệt độ phòng ban đầu (°C):", 20.0, 32.0, 27.0, step=0.5)
sim_hours = st.sidebar.slider("Số giờ mô phỏng (Hours):", 12, 72, 24, step=12)
t_min_comfort = st.sidebar.slider("Ngưỡng tiện nghi tối thiểu (°C):", 20.0, 23.0, 22.0, step=0.5)
t_max_comfort = st.sidebar.slider("Ngưỡng tiện nghi tối đa (°C):", 24.0, 27.0, 24.5, step=0.5)


# -------------------------------------------------------------
# PAGE 1: INTERACTIVE SIMULATION PLAYGROUND
# -------------------------------------------------------------
if page == "🎮 Trình Giả lập Điều khiển Thông minh":
    st.markdown('<div class="main-title">🎮 Trình Giả lập Điều khiển Thông minh HVAC</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Mô phỏng quyết định điều khiển HVAC theo thời gian thực: Cân bằng giữa tiện nghi nhiệt và tiết kiệm năng lượng.</div>', unsafe_allow_html=True)

    if df_data is None:
        st.warning("Vui lòng chạy script tiền xử lý dữ liệu trước!")
        st.stop()

    col_ctrl, col_scn = st.columns([2, 3])
    with col_ctrl:
        selected_agent_name = st.selectbox(
            "Chọn thuật toán điều khiển:",
            ["Tất cả thuật toán (So sánh đồng thời)", "Double Q-Learning (Proposed)", "Q-Learning (Baseline)", "Rule-Based (Thermostat)"]
        )

    with col_scn:
        scenario_idx = st.slider("Chọn mốc thời gian bắt đầu trong tập dữ liệu:", 0, len(df_data) - sim_hours - 1, 120, step=24)
        sample_time = df_data.iloc[scenario_idx]["datetime"]
        st.caption(f"📅 Thời điểm mô phỏng: **{sample_time.strftime('%Y-%m-%d %H:%M')}**")

    # Setup Custom Reward Function with UI bounds
    reward_fn = HVACRewardFunction(
        t_comfort_min=t_min_comfort,
        t_comfort_max=t_max_comfort,
        t_target=(t_min_comfort + t_max_comfort) / 2.0
    )

    env = SmartHVACEnv(
        data_df=df_data,
        episode_length=sim_hours,
        reward_fn=reward_fn
    )

    if st.button("🚀 Chạy Mô phỏng Điều khiển", type="primary", use_container_width=True):
        agents_to_run = {}
        if selected_agent_name == "Tất cả thuật toán (So sánh đồng thời)":
            agents_to_run = {
                "Rule-Based": models_dict["Rule-Based"],
                "Q-Learning": models_dict["Q-Learning"],
                "Double Q-Learning": models_dict["Double Q-Learning"]
            }
        elif "Double Q-Learning" in selected_agent_name:
            agents_to_run = {"Double Q-Learning": models_dict["Double Q-Learning"]}
        elif "Q-Learning" in selected_agent_name:
            agents_to_run = {"Q-Learning": models_dict["Q-Learning"]}
        else:
            agents_to_run = {"Rule-Based": models_dict["Rule-Based"]}

        sim_trajectories = {}
        summary_cards = {}

        for name, agent in agents_to_run.items():
            state, info = env.reset(start_idx=scenario_idx, initial_temp=initial_temp_slider)
            records = []
            total_energy = 0.0
            total_reward = 0.0
            violations = 0

            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, trunc, step_info = env.step(action)

                total_reward += reward
                total_energy += step_info["hvac_kw"] * 1.0
                temp = step_info["temp_c"]
                if temp < t_min_comfort or temp > t_max_comfort:
                    violations += 1

                records.append({
                    "Step": step_info["step"],
                    "Hour": step_info["hour"],
                    "Indoor Temp (°C)": temp,
                    "HVAC Power (kW)": step_info["hvac_kw"],
                    "Action": step_info["action_name"],
                    "Ambient Temp (°C)": step_info["ambient_temp"],
                    "Internal Heat (kW)": step_info["internal_heat_kw"],
                    "Reward": reward
                })
                state = next_state

            sim_trajectories[name] = pd.DataFrame(records)
            summary_cards[name] = {
                "Total Reward": total_reward,
                "Total Energy (kWh)": total_energy,
                "Violation Rate (%)": (violations / sim_hours) * 100.0,
                "Avg Temp (°C)": np.mean([r["Indoor Temp (°C)"] for r in records])
            }

        # Metric Scorecards
        st.markdown("### 📈 Kết quả Chỉ số Hiệu năng (KPIs)")
        cols_metric = st.columns(len(agents_to_run))
        for idx, (name, metrics) in enumerate(summary_cards.items()):
            with cols_metric[idx]:
                st.markdown(f"#### **{name}**")
                st.metric("Tổng Phần thưởng (Reward)", f"{metrics['Total Reward']:.1f}")
                st.metric("Điện năng tiêu thụ", f"{metrics['Total Energy (kWh)']:.1f} kWh")
                st.metric("Tỷ lệ vi phạm tiện nghi", f"{metrics['Violation Rate (%)']:.1f}%")
                st.metric("Nhiệt độ trung bình", f"{metrics['Avg Temp (°C)']:.2f} °C")

        # Visualization
        st.markdown("### 📉 Quỹ đạo Điều khiển & Nhiệt độ Phòng")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        colors = {"Rule-Based": "#6c757d", "Q-Learning": "#0275d8", "Double Q-Learning": "#28a745"}

        # Panel 1: Temperature Trajectory
        ax1.axhspan(t_min_comfort, t_max_comfort, color="#5cb85c", alpha=0.22, label=f"Vùng Tiện nghi ({t_min_comfort}°C - {t_max_comfort}°C)")
        
        # Plot ambient temperature as dashed line
        sample_traj = next(iter(sim_trajectories.values()))
        ax1.plot(sample_traj["Step"], sample_traj["Ambient Temp (°C)"], "k--", alpha=0.45, label="Nhiệt độ ngoài trời (Ambient)")

        for name, df_traj in sim_trajectories.items():
            c = colors.get(name, "#333333")
            lw = 2.5 if "Double" in name else 1.8
            ax1.plot(df_traj["Step"], df_traj["Indoor Temp (°C)"], label=f"Nhiệt độ ({name})", color=c, lw=lw)

        ax1.set_ylabel("Nhiệt độ (°C)", fontsize=11, fontweight="bold")
        ax1.set_title(f"Diễn biến Nhiệt độ Phòng qua {sim_hours} Giờ", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Panel 2: HVAC Power Output
        for name, df_traj in sim_trajectories.items():
            c = colors.get(name, "#333333")
            lw = 2.2 if "Double" in name else 1.5
            ax2.step(df_traj["Step"], df_traj["HVAC Power (kW)"], where="post", label=f"Công suất ({name})", color=c, lw=lw)

        ax2.set_xlabel("Thời gian mô phỏng (Giờ)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Công suất HVAC (kW)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Detailed step data expander
        with st.expander("🔍 Xem dữ liệu chi tiết từng bước mô phỏng (Logs)"):
            for name, df_traj in sim_trajectories.items():
                st.markdown(f"**Bảng theo dõi: {name}**")
                st.dataframe(df_traj, use_container_width=True)


# -------------------------------------------------------------
# PAGE 2: DATASET EXPLORER
# -------------------------------------------------------------
elif page == "📊 Khám phá Dữ liệu CU-BEMS":
    st.markdown('<div class="main-title">📊 Khám phá Bộ Dữ liệu Tòa nhà CU-BEMS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Phân tích đa cảm biến chất lượng không khí (IAQ) và phụ tải điện năng của Tòa nhà Đại học Chulalongkorn.</div>', unsafe_allow_html=True)

    if df_data is None:
        st.warning("Vui lòng hoàn tất bước tiền xử lý dữ liệu!")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tổng số giờ quan trắc", f"{len(df_data):,} h")
    with c2:
        st.metric("Nhiệt độ TB phòng", f"{df_data['temp_c'].mean():.2f} °C")
    with c3:
        st.metric("Độ ẩm TB phòng", f"{df_data['humidity_rh'].mean():.1f} %")
    with c4:
        st.metric("Công suất HVAC cực đại", f"{df_data['hvac_kw'].max():.1f} kW")

    tab1, tab2, tab3 = st.tabs(["📈 Chuỗi Thời gian Đa biến", "🔥 Ma trận Tương quan", "⏰ Chu kỳ 24 Giờ (Diurnal)"])

    with tab1:
        st.markdown("#### Diễn biến 1 Tuần Cảm biến Môi trường & Năng lượng")
        fig_ts = project_root / "result" / "figure" / "eda_time_series.png"
        if fig_ts.exists():
            st.image(str(fig_ts), use_container_width=True)
        else:
            st.info("Chưa có ảnh biểu đồ thời gian. Hãy chạy run_preprocess.py.")

    with tab2:
        st.markdown("#### Ma trận Tương quan Pearson giữa các Đại lượng Đo")
        fig_corr = project_root / "result" / "figure" / "eda_correlation_matrix.png"
        if fig_corr.exists():
            st.image(str(fig_corr), use_container_width=True)

    with tab3:
        st.markdown("#### Chu kỳ 24 Giờ: Ngày làm việc (Có người) vs Cuối tuần (Trống phòng)")
        fig_diurnal = project_root / "result" / "figure" / "eda_diurnal_patterns.png"
        if fig_diurnal.exists():
            st.image(str(fig_diurnal), use_container_width=True)


# -------------------------------------------------------------
# PAGE 3: BENCHMARKING & COMPARISON
# -------------------------------------------------------------
elif page == "⚖️ So sánh Hiệu năng (Benchmark)":
    st.markdown('<div class="main-title">⚖️ So sánh Hiệu năng: Double Q-Learning vs Baselines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Đánh giá khách quan trên tập kiểm thử chưa từng thấy (Unseen Test Dataset).</div>', unsafe_allow_html=True)

    summary_csv = project_root / "result" / "log" / "evaluation_summary.csv"
    if summary_csv.exists():
        df_eval = pd.read_csv(summary_csv)
        st.markdown("### 📋 Bảng Tổng hợp Chỉ số Đo lường")
        st.dataframe(df_eval.style.highlight_max(subset=["Total Reward"], color="#d4edda")
                                  .highlight_min(subset=["Comfort Violation (%)"], color="#d4edda"),
                     use_container_width=True)

        st.markdown("### 📊 Biểu đồ So sánh Trực quan")
        eval_fig = project_root / "result" / "figure" / "evaluation_comparison.png"
        if eval_fig.exists():
            st.image(str(eval_fig), use_container_width=True)

        conv_fig = project_root / "result" / "figure" / "training_convergence.png"
        if conv_fig.exists():
            st.markdown("### 🎯 Phân tích Sự Hội tụ & Hiện tượng Ước lượng Thừa (Overestimation Bias)")
            st.image(str(conv_fig), use_container_width=True)
    else:
        st.info("Chưa tìm thấy kết quả đánh giá. Hãy chạy `python evaluate.py` để tạo báo cáo!")


# -------------------------------------------------------------
# PAGE 4: Q-TABLE & POLICY INSPECTION
# -------------------------------------------------------------
elif page == "🧠 Trực quan hóa Bảng Q-Table & Policy":
    st.markdown('<div class="main-title">🧠 Trực quan hóa Bảng Q-Table & Chính sách Hành động</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quan sát trực tiếp giá trị hành động Q(s, a) mà Agent đã học được sau quá trình huấn luyện.</div>', unsafe_allow_html=True)

    agent_choice = st.selectbox("Chọn mô hình để kiểm tra:", ["Double Q-Learning", "Q-Learning"])
    target_agent = models_dict.get(agent_choice)

    if target_agent is not None:
        if agent_choice == "Double Q-Learning":
            q_matrix = target_agent.combined_q_table
        else:
            q_matrix = target_agent.q_table

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            q_matrix,
            cmap="viridis",
            cbar_kws={"label": "Giá trị Q(s, a)"},
            ax=ax
        )
        ax.set_title(f"Bảng Q-Table của {agent_choice} (45 States × 4 Actions)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Action (0: OFF, 1: LOW, 2: MED, 3: HIGH)", fontsize=11)
        ax.set_ylabel("Mã Trạng thái Rời rạc (State Index 0 - 44)", fontsize=11)
        st.pyplot(fig)

        # Policy interpretation
        st.markdown("### 🧭 Chính sách Tối ưu $\\pi^*(s) = \\arg\\max_a Q(s, a)$")
        best_actions = np.argmax(q_matrix, axis=1)
        policy_df = pd.DataFrame({
            "State Index": range(len(best_actions)),
            "Optimal Action": [ACTION_NAMES[a] for a in best_actions],
            "Max Q Value": np.max(q_matrix, axis=1).round(2)
        })
        st.dataframe(policy_df, use_container_width=True)
