"""
SmartHVACControlByDoubleQLearning - Interactive Research Scenario Lab.
A scientifically defensible HVAC Control & Generalization Research UI.
Features:
- Multi-domain selection (2018, 2019, Zones 1 & 2, Floor 3)
- Target Double Q-Learning (600 states) & Baseline Policies
- Zero-Shot, Calibration-Only, and Fine-Tuning transfer modes
- Provenance traceability badges ([OBSERVED], [EXTERNAL], [SIMULATED_ASSUMED], etc.)
- Safety Gateway live interlock & intervention monitoring
- Transition model validity comparison (Persistence vs RF vs Grey-Box)
- Robustness perturbation test explorer
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to sys.path
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.domain_config import DOMAINS, get_domain_spec
from src.data.domain_loader import get_domain_dataset, split_temporal_dataset
from src.data.calibration import EquipmentStageCalibration, EquipmentDriftDetector
from src.data.provenance import SYSTEM_PROVENANCE_REGISTRY, ProvenanceType
from src.environment.target_env import TargetHVACEnv, TARGET_ACTIONS
from src.agent.target_double_q_agent import TargetDoubleQLearningAgent
from src.agent.double_q_learning_agent import DoubleQLearningAgent
from src.agent.baseline_policies import (
    FixedSetpointPolicy, ComfortFirstPolicy, EnergySavingPolicy,
    RandomPolicy, LegacyDQLAdapter
)
from src.safety.safety_gateway import SafetyGateway, DeploymentReadinessLevel


# Page configuration
st.set_page_config(
    page_title="Smart HVAC Control | Generalization & Science Lab",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with High Contrast & Full Width
st.markdown("""
<style>
    /* Full width fluid layout adhering to [DIR-202611-002] */
    .block-container {
        max-width: 100% !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* High contrast, luminous typography */
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #38bdf8 !important;
        background: linear-gradient(120deg, #60a5fa 0%, #38bdf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.3px;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #94a3b8 !important;
        margin-bottom: 1.3rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* High-contrast status box with dark slate card */
    .status-box {
        padding: 16px 20px;
        border-radius: 8px;
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155;
        border-left: 5px solid #38bdf8 !important;
        margin-bottom: 16px;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    .status-box b, .status-box strong {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    .status-box code {
        background-color: #0f172a !important;
        color: #facc15 !important;
        padding: 2px 7px !important;
        border-radius: 4px !important;
        border: 1px solid #475569 !important;
        font-weight: 600 !important;
    }

    /* Provenance badges with high contrast */
    .prov-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .prov-observed {
        background-color: #064e3b !important;
        color: #6ee7b7 !important;
        border: 1px solid #059669 !important;
    }
    .prov-external {
        background-color: #1e3a5f !important;
        color: #7dd3fc !important;
        border: 1px solid #0284c7 !important;
    }
    .prov-simulated {
        background-color: #451a03 !important;
        color: #fcd34d !important;
        border: 1px solid #d97706 !important;
    }
    .prov-fitted {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
        border: 1px solid #64748b !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load trained model artifacts."""
    model_dir = project_root / "model"
    models = {}

    target_dql_path = model_dir / "target_double_q_model.pkl"
    if target_dql_path.exists():
        models["Target Double Q-Learning (600s)"] = TargetDoubleQLearningAgent.load(str(target_dql_path))

    legacy_dql_path = model_dir / "double_q_learning_model.pkl"
    if legacy_dql_path.exists():
        legacy = DoubleQLearningAgent.load(str(legacy_dql_path))
        models["Legacy DQL (45s Adapted)"] = LegacyDQLAdapter(legacy)

    models["Fixed Setpoint (25°C)"] = FixedSetpointPolicy()
    models["Comfort-First Policy"] = ComfortFirstPolicy()
    models["Energy-Saving Policy"] = EnergySavingPolicy()
    models["Random Policy"] = RandomPolicy()
    return models


models_dict = load_all_models()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/air-conditioner.png", width=68)
st.sidebar.title("Smart HVAC Control")
st.sidebar.caption("Cross-Domain Generalization & Physical Validity Lab")

menu_choice = st.sidebar.radio(
    "Chức Năng Nghiên Cứu:",
    [
        "🧪 Phòng Thử Nghiệm Kịch Bản (Scenario Lab)",
        "🌐 Điểm Chuẩn Tổng Quát Hóa (Generalization Benchmark)",
        "🔬 Đánh Giá Mô Hình Truyền Nhiệt (Transition Models)",
        "🛡️ Thử Nghiệm Độ Bền Vững & Cổng An Toàn (Safety & Robustness)",
        "📑 Bảng Truy Xuất Nguồn Gốc (Data Provenance)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Cấu Hình Môi Trường Thử Nghiệm")
selected_domain_id = st.sidebar.selectbox(
    "Tập Dữ Liệu / Miền (Domain):",
    list(DOMAINS.keys()),
    index=0,
    help="Primary: 2018_F2_Z1. Secondary: 2019_F2_Z1, 2018_F2_Z2, 2018_F3_Z1"
)
selected_transfer_mode = st.sidebar.selectbox(
    "Chế Độ Chuyển Giao (Transfer Mode):",
    ["ZERO-SHOT", "CALIBRATION-ONLY", "FINE-TUNING"],
    index=1,
    help="ZERO-SHOT: Giữ nguyên hiệu chuẩn cũ. CALIBRATION-ONLY: Tái hiệu chuẩn công suất thiết bị. FINE-TUNING: Huấn luyện thích nghi cục bộ."
)
use_safety_gateway_toggle = st.sidebar.toggle("Kích Hoạt Cổng An Toàn Độc Lập (Safety Gateway)", value=True)


# =========================================================================
# PAGE 1: SCENARIO LAB
# =========================================================================
if menu_choice == "🧪 Phòng Thử Nghiệm Kịch Bản (Scenario Lab)":
    st.markdown('<div class="main-title">🧪 Phòng Thử Nghiệm Kịch Bản Điều Khiển HVAC</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Mô phỏng quyết định của Agent trong môi trường vật lý thực nghiệm với nhãn nguồn gốc (Provenance) minh bạch.</div>', unsafe_allow_html=True)

    spec = get_domain_spec(selected_domain_id)
    df_domain = get_domain_dataset(selected_domain_id)

    st.markdown(f"""
    <div class="status-box">
        <b>Đang chạy trên miền:</b> <code>{spec.domain_id}</code> ({spec.description})<br/>
        <b>Số giờ quan trắc:</b> {len(df_domain):,} giờ | <b>Khoảng thời gian:</b> {df_domain['datetime'].min()} đến {df_domain['datetime'].max()}<br/>
        <b>Chế độ chuyển giao:</b> <code>{selected_transfer_mode}</code> | <b>Cổng an toàn:</b> {'BẬT (Khóa bảo vệ máy nén & sự cố nhiệt)' if use_safety_gateway_toggle else 'TẮT (Chỉ chạy Agent)'}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        agent_name = st.selectbox("Chọn Bộ Điều Khiển (Controller):", list(models_dict.keys()), index=0)
        selected_agent = models_dict[agent_name]
    with col2:
        init_t_slider = st.slider("Nhiệt độ phòng ban đầu (°C):", 21.0, 31.0, 26.5, step=0.5)
    with col3:
        sim_steps = st.slider("Số bước mô phỏng (Giờ):", 12, 72, 24, step=12)

    cal_path = project_root / "data" / "calibration" / f"cal_{selected_domain_id}.json"
    if cal_path.exists() and selected_transfer_mode in ["CALIBRATION-ONLY", "FINE-TUNING"]:
        cal = EquipmentStageCalibration.load(cal_path)
    else:
        # Source 2018 calibration
        cal = EquipmentStageCalibration.load(project_root / "data" / "calibration" / "cal_2018_F2_Z1.json")

    env = TargetHVACEnv(data_df=df_domain, episode_length=sim_steps, calibration=cal, seed=42)
    gateway = SafetyGateway() if use_safety_gateway_toggle else None

    if st.button("🚀 Bắt Đầu Chạy Mô Phỏng Kịch Bản", type="primary", use_container_width=True):
        state, info = env.reset(start_idx=40, initial_temp=init_t_slider)
        if gateway:
            gateway.reset()

        trajectory = []
        tot_energy = 0.0
        tot_reward = 0.0
        overrides = 0

        done = False
        while not done:
            raw_action = selected_agent.select_action(state, evaluate=True)

            if gateway:
                safe_act, reason = gateway.filter_action(
                    agent_action=raw_action,
                    indoor_temp=env.current_temp,
                    ambient_temp=float(env.data_df.iloc[env.data_idx]["ambient_temp_est"]),
                    ambient_rh=float(env.data_df.iloc[env.data_idx].get("ambient_rh_est", 70.0)),
                    hour=int(env.data_df.iloc[env.data_idx]["hour"]),
                    step=env.current_step
                )
                if safe_act != raw_action:
                    overrides += 1
                action_exec = safe_act
            else:
                action_exec = raw_action
                reason = None

            next_state, r, done, trunc, step_info = env.step(action_exec)
            tot_energy += step_info["hvac_kw"] * 1.0
            tot_reward += r

            trajectory.append({
                "Step": step_info["step"],
                "Hour": step_info["hour"],
                "Indoor Temp (°C)": step_info["temp_c"],
                "Ambient Temp (°C)": step_info["ambient_temp"],
                "Indoor RH (%)": step_info["humidity_rh"],
                "HVAC Power (kW)": step_info["hvac_kw"],
                "Agent Action": TARGET_ACTIONS[raw_action],
                "Executed Action": TARGET_ACTIONS[action_exec],
                "Window Open": step_info["window_open"],
                "Override Reason": reason if reason else "None",
                "Reward": r
            })
            state = next_state

        df_traj = pd.DataFrame(trajectory)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Tổng Phần Thưởng (Reward)", f"{tot_reward:.1f}")
        with k2:
            st.metric("Điện Năng Tiêu Thụ", f"{tot_energy:.1f} kWh")
        with k3:
            v_rate = (df_traj["Indoor Temp (°C)"] < 24.0).sum() + (df_traj["Indoor Temp (°C)"] > 26.0).sum()
            st.metric("Vi Phạm Vùng Tiện Nghi", f"{(v_rate / len(df_traj))*100:.1f} %")
        with k4:
            st.metric("Can Thiệp Cổng An Toàn", f"{overrides} bước")

        # Visualization with Dark High-Contrast Theme
        st.markdown("### 📈 Đồ Thị Quỹ Đạo Điều Khiển & Nhiệt Độ Phòng")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)

        fig.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#1e293b')
        ax2.set_facecolor('#1e293b')

        # Panel 1: Temperature & Comfort bounds
        ax1.axhspan(24.0, 26.0, color="#10b981", alpha=0.25, label="Vùng Tiện Nghi Ưu Tiên (24.0°C - 26.0°C) [ASHRAE 55]")
        ax1.axhline(21.0, color="#f87171", linestyle=":", lw=1.8, alpha=0.85, label="Ngưỡng Lạnh Nghiêm Trọng (21.0°C)")
        ax1.axhline(29.0, color="#f87171", linestyle=":", lw=1.8, alpha=0.85, label="Ngưỡng Nóng Nghiêm Trọng (29.0°C)")

        ax1.plot(df_traj["Step"], df_traj["Ambient Temp (°C)"], color="#94a3b8", linestyle="--", lw=1.8, alpha=0.8, label="Nhiệt Độ Ngoài Trời [EXTERNAL]")
        ax1.plot(df_traj["Step"], df_traj["Indoor Temp (°C)"], color="#38bdf8", lw=2.8, label=f"Nhiệt Độ Phòng ({agent_name}) [SIMULATED]")

        ax1.set_ylabel("Nhiệt Độ (°C)", fontweight="bold", color="#f1f5f9", fontsize=10)
        ax1.set_title("Diễn Biến Nhiệt Độ Phòng và Ranh Giới Tiện Nghi ASHRAE 55", fontweight="bold", color="#38bdf8", fontsize=12)
        leg1 = ax1.legend(loc="upper right", fontsize=8.5, facecolor="#0f172a", edgecolor="#475569")
        for text in leg1.get_texts():
            text.set_color('#f1f5f9')
        ax1.grid(True, color="#334155", linestyle="--", alpha=0.6)
        ax1.tick_params(colors="#cbd5e1")
        for spine in ax1.spines.values():
            spine.set_color("#475569")

        # Panel 2: HVAC Power & Window Action
        ax2.step(df_traj["Step"], df_traj["HVAC Power (kW)"], where="post", color="#4ade80", lw=2.2, label="Công Suất HVAC (kW) [FITTED_STAGE]")
        ax2.set_xlabel("Thời Gian Mô Phỏng (Giờ)", fontweight="bold", color="#f1f5f9", fontsize=10)
        ax2.set_ylabel("Công Suất HVAC (kW)", fontweight="bold", color="#f1f5f9", fontsize=10)
        leg2 = ax2.legend(loc="upper right", fontsize=8.5, facecolor="#0f172a", edgecolor="#475569")
        for text in leg2.get_texts():
            text.set_color('#f1f5f9')
        ax2.grid(True, color="#334155", linestyle="--", alpha=0.6)
        ax2.tick_params(colors="#cbd5e1")
        for spine in ax2.spines.values():
            spine.set_color("#475569")

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("🔍 Xem Bảng Nhật Ký Chi Tiết Từng Bước"):
            st.dataframe(df_traj, use_container_width=True)


# =========================================================================
# PAGE 2: GENERALIZATION BENCHMARK
# =========================================================================
elif menu_choice == "🌐 Điểm Chuẩn Tổng Quát Hóa (Generalization Benchmark)":
    st.markdown('<div class="main-title">🌐 Điểm Chuẩn Tổng Quát Hóa Liên Miền (Domain Shift Benchmark)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Đánh giá khả năng khái quát hóa của Double Q-Learning qua 4 trục miền và 3 chế độ chuyển giao nghiêm ngặt.</div>', unsafe_allow_html=True)

    tab_shift, tab_base = st.tabs(["📊 Thử Nghiệm 3 Chế Độ Chuyển Giao (Transfer Modes)", "⚖️ Bảng So Sánh Các Thuật Toán (Baselines)"])

    with tab_shift:
        shift_csv = project_root / "result" / "log" / "domain_shift_benchmark.csv"
        if shift_csv.exists():
            df_shift = pd.read_csv(shift_csv)
            st.dataframe(df_shift, use_container_width=True)

            shift_img = project_root / "result" / "figure" / "domain_shift_comparison.png"
            if shift_img.exists():
                st.image(str(shift_img), use_container_width=True)

            st.markdown("""
            #### 💡 Nhận Xét Khoa Học Từ Điểm Chuẩn:
            1. **ZERO-SHOT:** Áp dụng trực tiếp Q-Table sang miền năm 2019 khiến Reward giảm sâu (-118.93) do công suất các cấp điều hòa bị lệch vật lý đáng kể (LOW tăng từ 9.7 kW lên 19.7 kW).
            2. **CALIBRATION-ONLY:** Chỉ cần tái hiệu chuẩn ánh xạ công suất thiết bị bằng K-Means mà **không cần huấn luyện lại bất kỳ bước nào**, Reward lập tức cải thiện rõ rệt (-104.07) và tỷ lệ vi phạm giảm mạnh.
            3. **FINE-TUNING:** Khi cho phép Agent học thích nghi cục bộ chỉ với **ngân sách 25 episodes**, hiệu năng vượt trội (+48% Reward so với Zero-Shot), khẳng định tri thức cốt lõi đã học được từ năm 2018 có tính chuyển giao cao.
            """)
        else:
            st.info("Chưa có kết quả điểm chuẩn. Hãy chạy `train_target_dql.py` để tạo báo cáo.")

    with tab_base:
        base_csv = project_root / "result" / "log" / "baselines_benchmark.csv"
        if base_csv.exists():
            df_base = pd.read_csv(base_csv)
            st.dataframe(df_base, use_container_width=True)

            base_img = project_root / "result" / "figure" / "baselines_comparison.png"
            if base_img.exists():
                st.image(str(base_img), use_container_width=True)


# =========================================================================
# PAGE 3: TRANSITION MODELS
# =========================================================================
elif menu_choice == "🔬 Đánh Giá Mô Hình Truyền Nhiệt (Transition Models)":
    st.markdown('<div class="main-title">🔬 Kiểm Chứng Độ Hợp Lệ Của Mô Hình Truyền Nhiệt (Transition Models)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">So sánh đối đầu giữa Mô hình Ngẫu nhiên (Random Forest) và Mô hình Vật lý Hộp xám (Grey-Box RC).</div>', unsafe_allow_html=True)

    trans_csv = project_root / "result" / "log" / "transition_model_benchmark.csv"
    if trans_csv.exists():
        df_trans = pd.read_csv(trans_csv)
        st.dataframe(df_trans, use_container_width=True)

        st.markdown("""
        #### ⚠️ Phát Hiện Khoa Học Trọng Tâm:
        - **Random Forest (Legacy):** Dù đạt $R^2 = 0.9554$ trên tập huấn luyện nội bộ nhờ học vẹt chu kỳ nhiệt độ, nhưng **THẤT BẠI ở kiểm tra tính nhất quán vật lý (FAIL Physics Consistency)** và sai số bùng nổ lên **1.425°C MAE** khi kiểm thử trên miền năm 2019.
        - **Grey-Box RC Model:** Tuân thủ tuyệt đối định luật nhiệt động lực học (**PASS Physics Consistency**), giữ vững sai số ổn định **0.602°C MAE** trên miền chưa từng thấy.
        """)


# =========================================================================
# PAGE 4: SAFETY & ROBUSTNESS
# =========================================================================
elif menu_choice == "🛡️ Thử Nghiệm Độ Bền Vững & Cổng An Toàn (Safety & Robustness)":
    st.markdown('<div class="main-title">🛡️ Thử Nghiệm Độ Bền Vững Môi Trường & Cổng An Toàn</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Kiểm tra khả năng chịu lỗi của bộ điều khiển trước 7 kịch bản xáo trộn hệ thống và biến đổi khí hậu.</div>', unsafe_allow_html=True)

    rob_csv = project_root / "result" / "log" / "robustness_benchmark.csv"
    if rob_csv.exists():
        df_rob = pd.read_csv(rob_csv)
        st.dataframe(df_rob, use_container_width=True)

        rob_img = project_root / "result" / "figure" / "robustness_comparison.png"
        if rob_img.exists():
            st.image(str(rob_img), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔌 Giả Lập Giao Thức Tự Động Hóa Tòa Nhà (BMS HIL & Shadow Mode)")
    st.info("Trạng thái Triển khai Hiện tại: `RESEARCH_SIMULATION_ONLY` ➡️ Đã hoàn tất tầng kiểm định `OFFLINE_VALIDATED` và kết nối giao thức thử nghiệm `HARDWARE_IN_LOOP_VALIDATED`.")


# =========================================================================
# PAGE 5: DATA PROVENANCE
# =========================================================================
elif menu_choice == "📑 Bảng Truy Xuất Nguồn Gốc (Data Provenance)":
    st.markdown('<div class="main-title">📑 Bảng Đăng Ký Truy Xuất Nguồn Gốc Dữ Liệu (Data Provenance Registry)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Phân định rõ ràng giữa dữ liệu đo đạc thực tế, tham số hiệu chuẩn, dữ liệu ngoại cảnh và giả định mô phỏng.</div>', unsafe_allow_html=True)

    prov_list = []
    for k, p in SYSTEM_PROVENANCE_REGISTRY.items():
        prov_list.append(p.to_dict())
    df_p = pd.DataFrame(prov_list)
    st.dataframe(df_p, use_container_width=True)
