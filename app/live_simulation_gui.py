"""
HVAC Agent Live Simulation — Industrial Control Room & Digital Twin Dashboard
=============================================================================
A scientifically rigorous, real-time and step-by-step Digital Twin observation
environment for Multi-Device HVAC Control powered by Double Q-Learning.

Complies strictly with:
- [DIR-202611-002][web-layout] (Fluid 100% full-width responsive layout)
- [ERR-202611-003][gui-combobox-width] (Dedicated wide rows >= 600px with popdown width & dynamic descriptive labels)
- High-fidelity Dark Industrial Control Room Aesthetic (#0B1220 / #111C2E / #22D3EE)
- Research simulation only declaration (RESEARCH_SIMULATION_ONLY)
- Separation of Wall-clock time and Simulation chronology
- Scientific Data Provenance badges (SIMULATED, SIMULATED_ASSUMED, DERIVED, OBSERVED)
- Step Forward strictly advancing exactly one discrete simulation step
- Online Learning safety: disabled for baseline controllers, enabled for learning agents
- No redundant RNG sampling for exploration / exploitation detection
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd

# Matplotlib embedded in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add project root to sys.path
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.environment.multi_device_env import (
    MultiDeviceHVACEnv, RoomGeometry, ROOM_PRESETS,
    MULTI_DEVICE_ACTION_TEMPLATES, index_to_state_tuple_720
)
from src.environment.action_descriptions_vi import (
    ACTION_META_VI, CONTROLLER_META_VI,
    format_action_short_vi, format_action_devices_vi
)
from src.reward.multi_device_reward import MultiDeviceRewardFunction
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyGateway
from src.agent.multi_device_double_q_agent import MultiDeviceDoubleQAgent, SingleQMultiDeviceAgent
from src.agent.multi_device_baselines import (
    FixedSetpointMultiDevice, ConventionalThermostatMultiDevice,
    RuleBasedSmartHVACMultiDevice, RandomMultiDevicePolicy
)
from src.data.domain_loader import get_domain_dataset


class LiveSimulationApp:
    """Master Industrial Control Dashboard for HVAC Digital Twin Live Simulation."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart HVAC Digital Twin | Phòng Điều Khiển Mô Phỏng Đa Thiết Bị")
        self.root.geometry("1560x980")
        self.root.minsize(1280, 780)

        # High-Fidelity Industrial Dark Theme Palette
        self.c_bg = "#0b1220"             # Deep Navy background
        self.c_card = "#111c2e"           # Main panel / container
        self.c_card_inner = "#172438"     # Secondary panel / inner card
        self.c_border = "#26364d"         # Crisp border accent
        self.c_accent = "#22d3ee"         # Primary Cyan accent
        self.c_success = "#34d399"        # Emerald green (healthy/normal)
        self.c_warn = "#fbbf24"           # Amber warning
        self.c_danger = "#f87171"         # Coral red (alert/hazard)
        self.c_purple = "#a78bfa"         # Purple (learning/neural)
        self.c_text = "#e5e7eb"           # Primary light text
        self.c_text_muted = "#94a3b8"     # Secondary muted slate text

        self.root.configure(bg=self.c_bg)
        self._configure_ttk_styles()

        # Simulation playback state
        self.is_running = False
        self.is_paused = False
        self.delay_ms = 500
        self.timer_job = None
        self.online_learning_enabled = False
        self.q_updates_count = 0

        # Core Digital Twin components
        self.cubems_df: Optional[pd.DataFrame] = None
        self.env: Optional[MultiDeviceHVACEnv] = None
        self.reward_fn = MultiDeviceRewardFunction()
        self.safety_gateway = MultiDeviceSafetyGateway()
        self.current_agent = None
        self.agent_name = ""
        self.agents_dict: Dict[str, Any] = {}

        # Trajectory history for live telemetry subplots
        self.history: Dict[str, List[Any]] = {
            "step": [],
            "sim_hour": [],
            "temp_c": [],
            "amb_temp": [],
            "rh": [],
            "amb_rh": [],
            "co2": [],
            "pm25": [],
            "total_kw": [],
            "cooling_kw": [],
            "vent_kw": [],
            "filtration_kw": [],
            "cumul_kwh": [],
            "reward": [],
            "safe_override": []
        }
        self.step_diagnostics_log: List[Dict[str, Any]] = []

        # Initialize Models & Default Environment
        self._load_models_and_environment()

        # Build Full-Width Responsive UI Layout
        self._build_ui()

        # Reset Environment to Step 0 Initial Conditions
        self.reset_simulation()

        # Start Wall-clock ticker
        self._tick_wall_clock()

    # -----------------------------------------------------------------------
    # TTK Styles Configuration
    # -----------------------------------------------------------------------
    def _configure_ttk_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Combobox style matching dark theme & [ERR-202611-003]
        style.configure(
            "Wide.TCombobox",
            fieldbackground=self.c_card_inner,
            background=self.c_card,
            foreground=self.c_text,
            arrowcolor=self.c_accent,
            bordercolor=self.c_border,
            darkcolor=self.c_card,
            lightcolor=self.c_card,
            padding=6,
            font=("Segoe UI", 10, "bold")
        )
        style.map(
            "Wide.TCombobox",
            fieldbackground=[("readonly", self.c_card_inner)],
            selectbackground=[("readonly", self.c_card_inner)],
            selectforeground=[("readonly", self.c_text)]
        )

        # Treeview styling
        style.configure(
            "Dark.Treeview",
            background=self.c_card_inner,
            foreground=self.c_text,
            fieldbackground=self.c_card_inner,
            rowheight=24,
            font=("Consolas", 8),
            bordercolor=self.c_border
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=self.c_card,
            foreground=self.c_accent,
            font=("Segoe UI", 9, "bold"),
            relief="flat"
        )
        style.map("Dark.Treeview", background=[("selected", "#0369a1")])

    # -----------------------------------------------------------------------
    # Model Loading & Environment Instantiation
    # -----------------------------------------------------------------------
    def _load_models_and_environment(self):
        model_dir = project_root / "model"
        dql_path = model_dir / "final_dql_seed0.pkl"
        if not dql_path.exists():
            dql_path = model_dir / "multi_device_double_q_model.pkl"
        sql_path = model_dir / "final_ql_seed0.pkl"
        if not sql_path.exists():
            sql_path = model_dir / "multi_device_single_q_model.pkl"

        self.is_dql_checkpoint_loaded = False
        self.is_sql_checkpoint_loaded = False

        if dql_path.exists():
            try:
                loaded_dql = MultiDeviceDoubleQAgent.load_checkpoint(dql_path)
                loaded_dql.is_learning_agent = True
                self.agents_dict["Double Q-Learning (Học tăng cường kép)"] = loaded_dql
                self.is_dql_checkpoint_loaded = True
            except Exception as e:
                print(f"Warning: loading Double Q model failed ({e}), creating fresh agent.")
                self.agents_dict["Double Q-Learning (Học tăng cường kép)"] = MultiDeviceDoubleQAgent(seed=42)
        else:
            self.agents_dict["Double Q-Learning (Học tăng cường kép)"] = MultiDeviceDoubleQAgent(seed=42)

        if sql_path.exists():
            try:
                import pickle
                with open(sql_path, "rb") as f:
                    loaded_sql = pickle.load(f)
                    loaded_sql.is_learning_agent = True
                    self.agents_dict["Q-Learning đơn (Học tăng cường đơn)"] = loaded_sql
                    self.is_sql_checkpoint_loaded = True
            except Exception:
                self.agents_dict["Q-Learning đơn (Học tăng cường đơn)"] = SingleQMultiDeviceAgent(seed=42)
        else:
            self.agents_dict["Q-Learning đơn (Học tăng cường đơn)"] = SingleQMultiDeviceAgent(seed=42)

        self.agents_dict["Bộ luật chuyên gia IF-THEN (Theo ngưỡng)"] = RuleBasedSmartHVACMultiDevice()
        self.agents_dict["Rơ-le nhiệt truyền thống (Bật/Tắt đơn thuần)"] = ConventionalThermostatMultiDevice()
        self.agents_dict["Đặt cố định 25°C (Làm mát & thông gió nền)"] = FixedSetpointMultiDevice()
        self.agents_dict["Chính sách ngẫu nhiên (Chọn bừa 1/12 hành động)"] = RandomMultiDevicePolicy(seed=42)

        # Load processed CU-BEMS dataset if present
        try:
            self.cubems_df = get_domain_dataset("2018_F2_Z1")
        except Exception as e:
            print(f"Notice: CU-BEMS dataset fallback ({e}). Using synthetic scenario.")
            self.cubems_df = None

        self.env = MultiDeviceHVACEnv(data_df=self.cubems_df, room=ROOM_PRESETS["medium"], episode_length=48, seed=42)
        self.agent_name = "Double Q-Learning (Học tăng cường kép)"
        self.current_agent = self.agents_dict[self.agent_name]

    # -----------------------------------------------------------------------
    # Master UI Layout Construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.main_container = tk.Frame(self.root, bg=self.c_bg)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # A. Top Header
        self._build_top_header(self.main_container)

        # B. KPI Overview Card Bar
        self._build_kpi_overview_bar(self.main_container)

        # C. Master PanedWindow: Left Observation & Controls (630px) | Right Telemetry Charts & Table (Fluid)
        self.paned = tk.PanedWindow(self.main_container, orient=tk.HORIZONTAL, bg=self.c_bg, sashwidth=5, bd=0)
        self.paned.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Left Scrollable Panel
        left_outer = tk.Frame(self.paned, bg=self.c_bg, width=630)
        right_outer = tk.Frame(self.paned, bg=self.c_bg)
        self.paned.add(left_outer, minsize=610)
        self.paned.add(right_outer, minsize=720)

        left_canvas = tk.Canvas(left_outer, bg=self.c_bg, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        self.scroll_frame = tk.Frame(left_canvas, bg=self.c_bg)

        self.scroll_frame.bind(
            "<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        self.canvas_win = left_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Ensure scroll frame matches canvas width on resize
        left_canvas.bind("<Configure>", lambda e: left_canvas.itemconfig(self.canvas_win, width=e.width))

        # Build Left Panel Sections
        self._build_control_panel(self.scroll_frame)
        self._build_environmental_telemetry_section(self.scroll_frame)
        self._build_equipment_section(self.scroll_frame)
        self._build_agent_decision_trace_section(self.scroll_frame)

        # Build Right Panel: Charts + History Table
        self._build_charts_and_history_section(right_outer)

    # -----------------------------------------------------------------------
    # Section A: Top Header
    # -----------------------------------------------------------------------
    def _build_top_header(self, parent):
        header = tk.Frame(parent, bg=self.c_card, highlightbackground=self.c_border, highlightthickness=1)
        header.pack(fill=tk.X, pady=(0, 6))

        left_sub = tk.Frame(header, bg=self.c_card)
        left_sub.pack(side=tk.LEFT, padx=12, pady=6)

        title_lbl = tk.Label(
            left_sub, text="⚡ HỆ THỐNG MÔ PHỎNG SMART HVAC (DIGITAL TWIN)",
            font=("Segoe UI", 12, "bold"), fg=self.c_accent, bg=self.c_card
        )
        title_lbl.pack(side=tk.LEFT, padx=(0, 8))

        sub_lbl = tk.Label(
            left_sub, text="| Điều khiển 7 nhóm thiết bị bằng Double Q-Learning",
            font=("Segoe UI", 9), fg=self.c_text_muted, bg=self.c_card
        )
        sub_lbl.pack(side=tk.LEFT)

        right_sub = tk.Frame(header, bg=self.c_card)
        right_sub.pack(side=tk.RIGHT, padx=12, pady=6)

        self.badge_status = tk.Label(
            right_sub, text="⏹ ĐÃ DỪNG", font=("Segoe UI", 9, "bold"),
            fg="#94a3b8", bg=self.c_card_inner, padx=8, pady=2,
            highlightbackground=self.c_border, highlightthickness=1
        )
        self.badge_status.pack(side=tk.LEFT, padx=4)

        self.badge_mode = tk.Label(
            right_sub, text="🔒 CHẾ ĐỘ ĐÁNH GIÁ (Đóng băng Q-Table)", font=("Segoe UI", 9, "bold"),
            fg=self.c_accent, bg=self.c_card_inner, padx=8, pady=2,
            highlightbackground=self.c_border, highlightthickness=1
        )
        self.badge_mode.pack(side=tk.LEFT, padx=4)
        self._update_badge_status()

        tag_lbl = tk.Label(
            right_sub, text="[MÔ PHỎNG NGHIÊN CỨU]",
            font=("Segoe UI", 8, "italic"), fg=self.c_warn, bg=self.c_card
        )
        tag_lbl.pack(side=tk.LEFT, padx=(6, 0))

    # -----------------------------------------------------------------------
    # Section B: KPI Overview Cards Bar
    # -----------------------------------------------------------------------
    def _build_kpi_overview_bar(self, parent):
        kpi_bar = tk.Frame(parent, bg=self.c_bg)
        kpi_bar.pack(fill=tk.X, pady=(0, 6))

        for col in range(8):
            kpi_bar.columnconfigure(col, weight=1)

        self.kpi_temp = self._make_kpi_card(kpi_bar, 0, "NHIỆT ĐỘ PHÒNG", "25.50 °C", "Mục tiêu [24.0, 26.0]°C", self.c_accent)
        self.kpi_rh = self._make_kpi_card(kpi_bar, 1, "ĐỘ ẨM PHÒNG", "55.0 %RH", "Thoải mái [40, 65]%", self.c_accent)
        self.kpi_co2 = self._make_kpi_card(kpi_bar, 2, "NỒNG ĐỘ CO2", "650 ppm", "< 800 ppm Trong lành", self.c_success)
        self.kpi_pm25 = self._make_kpi_card(kpi_bar, 3, "BỤI MỊN PM2.5", "18.0 µg/m³", "<= 35 µg/m³ An toàn", self.c_success)
        self.kpi_power = self._make_kpi_card(kpi_bar, 4, "CÔNG SUẤT TỨC THỜI", "0.00 kW", "Tổng tải 7 thiết bị", self.c_warn)
        self.kpi_energy = self._make_kpi_card(kpi_bar, 5, "ĐIỆN NĂNG TÍCH LŨY", "0.0 kWh", "Cộng dồn từ đầu phiên", self.c_warn)
        self.kpi_time = self._make_kpi_card(kpi_bar, 6, "GIỜ MÔ PHỎNG", "12:00", "Bước 0 / 48", self.c_purple)
        self.kpi_updates = self._make_kpi_card(kpi_bar, 7, "CẬP NHẬT Q-TABLE", "#0", "Đóng băng (Eval)", self.c_text_muted)

    def _make_kpi_card(self, parent, col, title, initial_val, subtext, color):
        frame = tk.Frame(parent, bg=self.c_card, highlightbackground=self.c_border, highlightthickness=1, padx=6, pady=4)
        frame.grid(row=0, column=col, padx=2, sticky="nsew")

        tk.Label(frame, text=title.upper(), font=("Segoe UI", 7, "bold"), fg=self.c_text_muted, bg=self.c_card).pack(anchor="w")
        val_lbl = tk.Label(frame, text=initial_val, font=("Consolas", 11, "bold"), fg=color, bg=self.c_card)
        val_lbl.pack(anchor="w")
        sub_lbl = tk.Label(frame, text=subtext, font=("Segoe UI", 7), fg=self.c_text_muted, bg=self.c_card)
        sub_lbl.pack(anchor="w")
        return {"val": val_lbl, "sub": sub_lbl}

    # -----------------------------------------------------------------------
    # Section C: Simulation Control Panel
    # -----------------------------------------------------------------------
    def _build_control_panel(self, parent):
        card = tk.LabelFrame(
            parent, text=" 🕹️ BẢNG ĐIỀU KHIỂN KỊCH BẢN & TÁC TỬ AI ",
            font=("Segoe UI", 9, "bold"), fg=self.c_accent, bg=self.c_card,
            padx=8, pady=4, highlightbackground=self.c_border, highlightthickness=1
        )
        card.pack(fill=tk.X, pady=(0, 5))

        # Adhere to [ERR-202611-003]: Dedicated wide row for Scenario Combobox >= 600px
        tk.Label(
            card, text="Mô Hình Phòng & Kịch Bản Môi Trường:",
            font=("Segoe UI", 9, "bold"), fg=self.c_text, bg=self.c_card
        ).pack(anchor="w")

        self.scenario_var = tk.StringVar(value="Phòng Văn phòng Vừa 50m² — Dữ liệu thực tế CU-BEMS 2018")
        self.scenario_cb = ttk.Combobox(
            card, textvariable=self.scenario_var, state="readonly", style="Wide.TCombobox",
            values=[
                "Phòng Văn phòng Vừa 50m² — Dữ liệu thực tế CU-BEMS 2018",
                "Phòng Nghiên cứu Nhỏ 20m² — Mô phỏng nắng nóng cực đại",
                "Hội trường / Phòng họp Lớn 120m² — Đông người cao điểm",
                "Kịch bản Ô nhiễm Bụi mịn Ngoại cảnh — PM2.5 & Ẩm cao"
            ]
        )
        self.scenario_cb.pack(fill=tk.X, pady=(2, 2))
        self.scenario_cb.bind("<<ComboboxSelected>>", self._on_scenario_changed)
        self.scenario_cb.config(postcommand=lambda: self.scenario_cb.configure(width=72))

        self.scenario_desc_lbl = tk.Label(
            card, text="ℹ️ Diện tích 50m², cách nhiệt U=0.55 W/m²K, tải 3-4 người. Nguồn dữ liệu: CU-BEMS 2018.",
            font=("Segoe UI", 8, "italic"), fg=self.c_text_muted, bg=self.c_card
        )
        self.scenario_desc_lbl.pack(anchor="w", pady=(0, 6))

        # Dedicated wide row for Agent Controller Combobox >= 600px
        tk.Label(
            card, text="Bộ Điều Khiển (Thuật toán AI / Đường cơ sở):",
            font=("Segoe UI", 9, "bold"), fg=self.c_text, bg=self.c_card
        ).pack(anchor="w")

        self.agent_var = tk.StringVar(value=self.agent_name)
        self.agent_cb = ttk.Combobox(
            card, textvariable=self.agent_var, state="readonly", style="Wide.TCombobox",
            values=list(self.agents_dict.keys())
        )
        self.agent_cb.pack(fill=tk.X, pady=(2, 2))
        self.agent_cb.bind("<<ComboboxSelected>>", self._on_agent_changed)
        self.agent_cb.config(postcommand=lambda: self.agent_cb.configure(width=72))

        self.agent_desc_lbl = tk.Label(
            card, text="ℹ️ Double Q-Learning: 720 trạng thái x 12 hành động, dùng 2 bảng QA/QB khử thiên lệch cực đại.",
            font=("Segoe UI", 8, "italic"), fg=self.c_text_muted, bg=self.c_card
        )
        self.agent_desc_lbl.pack(anchor="w", pady=(0, 6))

        # Speed and Online Learning Toggles
        opts_frame = tk.Frame(card, bg=self.c_card)
        opts_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Label(opts_frame, text="Tốc Độ:", font=("Segoe UI", 9, "bold"), fg=self.c_text, bg=self.c_card).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0x (500ms)")
        speed_cb = ttk.Combobox(
            opts_frame, textvariable=self.speed_var, state="readonly", width=13,
            values=["0.25x (2000ms)", "0.5x (1000ms)", "1.0x (500ms)", "2.0x (250ms)", "5.0x (100ms)"]
        )
        speed_cb.pack(side=tk.LEFT, padx=6)
        speed_cb.bind("<<ComboboxSelected>>", self._on_speed_changed)

        self.online_learn_var = tk.BooleanVar(value=False)
        self.online_learn_chk = tk.Checkbutton(
            opts_frame, text="Bật Học Trực Tuyến (Online Learning)",
            variable=self.online_learn_var, command=self._on_online_learning_toggle,
            font=("Segoe UI", 9, "bold"), fg=self.c_purple, bg=self.c_card,
            selectcolor=self.c_card_inner, activebackground=self.c_card, activeforeground=self.c_purple
        )
        self.online_learn_chk.pack(side=tk.RIGHT)

        # Control Buttons
        btn_frame = tk.Frame(card, bg=self.c_card)
        btn_frame.pack(fill=tk.X, pady=2)

        self.btn_start = tk.Button(
            btn_frame, text="▶ BẮT ĐẦU", font=("Segoe UI", 9, "bold"),
            bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
            width=11, relief=tk.FLAT, cursor="hand2", command=self.start_simulation
        )
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_pause = tk.Button(
            btn_frame, text="⏸ TẠM DỪNG", font=("Segoe UI", 9, "bold"),
            bg="#d97706", fg="#ffffff", activebackground="#b45309", activeforeground="#ffffff",
            width=11, relief=tk.FLAT, cursor="hand2", state=tk.DISABLED, command=self.pause_simulation
        )
        self.btn_pause.pack(side=tk.LEFT, padx=2)

        self.btn_step = tk.Button(
            btn_frame, text="⏭ CHẠY 1 BƯỚC", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857", activeforeground="#ffffff",
            width=15, relief=tk.FLAT, cursor="hand2", command=self.step_forward_simulation
        )
        self.btn_step.pack(side=tk.LEFT, padx=4)

        self.btn_stop = tk.Button(
            btn_frame, text="⏹ DỪNG", font=("Segoe UI", 9, "bold"),
            bg="#dc2626", fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff",
            width=8, relief=tk.FLAT, cursor="hand2", command=self.stop_simulation
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.btn_reset = tk.Button(
            btn_frame, text="🔄 ĐẶT LẠI", font=("Segoe UI", 9, "bold"),
            bg="#475569", fg="#ffffff", activebackground="#334155", activeforeground="#ffffff",
            width=9, relief=tk.FLAT, cursor="hand2", command=self.reset_simulation
        )
        self.btn_reset.pack(side=tk.RIGHT, padx=2)

    # -----------------------------------------------------------------------
    # Section E: Environmental Telemetry (Indoor, Outdoor, Room Physics)
    # -----------------------------------------------------------------------
    def _build_environmental_telemetry_section(self, parent):
        card = tk.LabelFrame(
            parent, text=" 🌡️ THÔNG SỐ VẬT LÝ MÔI TRƯỜNG & NGUỒN DỮ LIỆU ",
            font=("Segoe UI", 9, "bold"), fg=self.c_accent, bg=self.c_card,
            padx=8, pady=4, highlightbackground=self.c_border, highlightthickness=1
        )
        card.pack(fill=tk.X, pady=(0, 5))

        cols = tk.Frame(card, bg=self.c_card)
        cols.pack(fill=tk.X)

        # Col 1: Indoor Environment
        col_in = tk.Frame(cols, bg=self.c_card_inner, padx=8, pady=6, highlightbackground=self.c_border, highlightthickness=1)
        col_in.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        tk.Label(col_in, text="TRONG PHÒNG [MÔ PHỎNG]", font=("Segoe UI", 8, "bold"), fg=self.c_accent, bg=self.c_card_inner).pack(anchor="w")
        self.lbl_in_temp = tk.Label(col_in, text="Nhiệt độ: 25.50 °C", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_in_temp.pack(anchor="w")
        self.lbl_in_rh = tk.Label(col_in, text="Độ ẩm: 55.0 %RH", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_in_rh.pack(anchor="w")
        self.lbl_in_co2 = tk.Label(col_in, text="CO2: 650 ppm", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_in_co2.pack(anchor="w")
        self.lbl_in_pm25 = tk.Label(col_in, text="Bụi PM2.5: 18.0 µg/m³", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_in_pm25.pack(anchor="w")
        self.lbl_in_occ = tk.Label(col_in, text="Số người: 3 người [SUY LUẬN]", font=("Segoe UI", 8, "italic"), fg=self.c_text_muted, bg=self.c_card_inner)
        self.lbl_in_occ.pack(anchor="w")

        # Col 2: Outdoor Environment
        col_out = tk.Frame(cols, bg=self.c_card_inner, padx=8, pady=6, highlightbackground=self.c_border, highlightthickness=1)
        col_out.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        tk.Label(col_out, text="NGOÀI TRỜI [GIẢ LẬP]", font=("Segoe UI", 8, "bold"), fg=self.c_warn, bg=self.c_card_inner).pack(anchor="w")
        self.lbl_out_temp = tk.Label(col_out, text="Nhiệt độ: 32.00 °C", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_out_temp.pack(anchor="w")
        self.lbl_out_rh = tk.Label(col_out, text="Độ ẩm: 65.0 %RH", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_out_rh.pack(anchor="w")
        self.lbl_out_co2 = tk.Label(col_out, text="CO2 nền: 415 ppm [CƠ SỞ]", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_out_co2.pack(anchor="w")
        self.lbl_out_pm25 = tk.Label(col_out, text="Bụi PM2.5: 25.0 µg/m³", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_out_pm25.pack(anchor="w")
        self.lbl_out_cond = tk.Label(col_out, text="Thời tiết: Chu kỳ ngày đêm", font=("Segoe UI", 8, "italic"), fg=self.c_text_muted, bg=self.c_card_inner)
        self.lbl_out_cond.pack(anchor="w")

        # Col 3: Room Physics
        col_room = tk.Frame(cols, bg=self.c_card_inner, padx=8, pady=6, highlightbackground=self.c_border, highlightthickness=1)
        col_room.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        tk.Label(col_room, text="KẾT CẤU PHÒNG [VẬT LÝ]", font=("Segoe UI", 8, "bold"), fg=self.c_success, bg=self.c_card_inner).pack(anchor="w")
        self.lbl_room_geom = tk.Label(col_room, text="Sàn: 50 m² | Thể tích: 140 m³", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_room_geom.pack(anchor="w")
        self.lbl_room_cap = tk.Label(col_room, text="Nhiệt dung C_th: 11,000 kJ/K", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_room_cap.pack(anchor="w")
        self.lbl_room_u = tk.Label(col_room, text="Hệ số vỏ U: 0.55 W/m²K", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_room_u.pack(anchor="w")
        self.lbl_room_p_int = tk.Label(col_room, text="Tải nội sinh: 2.0 kW [CU-BEMS]", font=("Segoe UI", 8), fg=self.c_text, bg=self.c_card_inner)
        self.lbl_room_p_int.pack(anchor="w")
        self.lbl_room_ach = tk.Label(col_room, text="Thẩm thấu nền: 0.20 ACH", font=("Segoe UI", 8, "italic"), fg=self.c_text_muted, bg=self.c_card_inner)
        self.lbl_room_ach.pack(anchor="w")

    # -----------------------------------------------------------------------
    # Section F: HVAC Device Control Panel (7 Groups)
    # -----------------------------------------------------------------------
    def _build_equipment_section(self, parent):
        card = tk.LabelFrame(
            parent, text=" ⚙️ TRẠNG THÁI 7 NHÓM THIẾT BỊ & CỬA SỔ (kW) ",
            font=("Segoe UI", 9, "bold"), fg=self.c_accent, bg=self.c_card,
            padx=8, pady=4, highlightbackground=self.c_border, highlightthickness=1
        )
        card.pack(fill=tk.X, pady=(0, 5))

        grid = tk.Frame(card, bg=self.c_card)
        grid.pack(fill=tk.X)

        self.eq_ac = self._make_eq_badge(grid, 0, 0, "Máy lạnh (AC)", "TẮT\n0.00 kW", self.c_accent)
        self.eq_heat = self._make_eq_badge(grid, 0, 1, "Máy sưởi", "TẮT\n0.00 kW", self.c_danger)
        self.eq_vent = self._make_eq_badge(grid, 0, 2, "Quạt gió tươi", "0 m³/h\n0.00 kW", self.c_success)
        self.eq_win = self._make_eq_badge(grid, 0, 3, "Cửa sổ / Van gió", "ĐANG ĐÓNG\n0 m³/h", "#a3e635")

        self.eq_humid = self._make_eq_badge(grid, 1, 0, "Máy tạo ẩm", "0.00 kW\n+0.00 kg/h", "#67e8f9")
        self.eq_dehumid = self._make_eq_badge(grid, 1, 1, "Máy hút ẩm", "0.00 kW\n-0.00 kg/h", "#c084fc")
        self.eq_filter = self._make_eq_badge(grid, 1, 2, "Lọc bụi HEPA", "0 m³/h\n0.00 kW", self.c_warn)
        self.eq_total = self._make_eq_badge(grid, 1, 3, "TỔNG ĐIỆN NĂNG", "0.00 kW\n0.0 kWh", "#f43f5e")

    def _make_eq_badge(self, parent, row, col, name, initial_val, color):
        frame = tk.Frame(parent, bg=self.c_card_inner, padx=6, pady=4, highlightbackground=self.c_border, highlightthickness=1)
        frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1, uniform="eq_col")

        name_lbl = tk.Label(frame, text=name, font=("Segoe UI", 8, "bold"), fg=color, bg=self.c_card_inner, anchor="w")
        name_lbl.pack(anchor="w")
        val_lbl = tk.Label(frame, text=initial_val, font=("Consolas", 8, "bold"), fg=self.c_text, bg=self.c_card_inner, justify="left", anchor="w")
        val_lbl.pack(anchor="w")
        return val_lbl

    # -----------------------------------------------------------------------
    # Section G: Agent Decision Trace
    # -----------------------------------------------------------------------
    def _build_agent_decision_trace_section(self, parent):
        card = tk.LabelFrame(
            parent, text=" 🧠 DẤU VẾT QUYẾT ĐỊNH CỦA AI & CẤU HÌNH THIẾT BỊ ",
            font=("Segoe UI", 9, "bold"), fg=self.c_accent, bg=self.c_card,
            padx=8, pady=4, highlightbackground=self.c_border, highlightthickness=1
        )
        card.pack(fill=tk.X, pady=(0, 4))

        # Row 1: State Observation
        self.lbl_agent_state = tk.Label(
            card, text="Trạng thái s: [Nhiệt:3, Ẩm:1, CO2:0, Bụi:0, Tải:0, Giờ:0] -> Ô 0 / 720",
            font=("Consolas", 8, "bold"), fg=self.c_accent, bg=self.c_card_inner, padx=6, pady=3,
            anchor="w", justify="left", wraplength=575,
            highlightbackground=self.c_border, highlightthickness=1
        )
        self.lbl_agent_state.pack(fill=tk.X, pady=(0, 4))

        # Row 2: Exploration / Exploitation & Q Update
        row2 = tk.Frame(card, bg=self.c_card)
        row2.pack(fill=tk.X, pady=2)
        self.lbl_agent_mode = tk.Label(
            row2, text="Chế độ: KHAI THÁC (Chọn max Q)",
            font=("Segoe UI", 8, "bold"), fg=self.c_success, bg=self.c_card
        )
        self.lbl_agent_mode.pack(side=tk.LEFT)

        self.lbl_q_update = tk.Label(
            row2, text="Cập nhật Q-Table: KHÔNG (Đóng băng đánh giá)",
            font=("Segoe UI", 8, "bold"), fg=self.c_text_muted, bg=self.c_card
        )
        self.lbl_q_update.pack(side=tk.RIGHT)

        # Row 3: Agent Action & Value estimates
        self.lbl_agent_choice = tk.Label(
            card, text="AI Đề xuất: Chưa chạy bước nào (Nhấn CHẠY 1 BƯỚC hoặc BẮT ĐẦU).",
            font=("Segoe UI", 8, "bold"), fg="#fcd34d", bg=self.c_card,
            anchor="w", justify="left", wraplength=575
        )
        self.lbl_agent_choice.pack(fill=tk.X, pady=2)

        # Row 4: Concrete Device & Window Setup for the Executed Action
        self.lbl_device_config = tk.Label(
            card, text="Thiết lập thiết bị: Chưa kích hoạt (Tất cả đang ở trạng thái chờ).",
            font=("Consolas", 8, "bold"), fg="#38bdf8", bg=self.c_card_inner, padx=6, pady=3,
            anchor="w", justify="left", wraplength=575,
            highlightbackground=self.c_border, highlightthickness=1
        )
        self.lbl_device_config.pack(fill=tk.X, pady=2)

        # Row 5: Safety Gateway Verdict
        self.lbl_safety_verdict = tk.Label(
            card, text="Cổng An toàn: Đang trực (Chờ kiểm tra bước tiếp theo).",
            font=("Segoe UI", 8, "bold"), fg=self.c_text_muted, bg=self.c_card,
            anchor="w", justify="left", wraplength=575
        )
        self.lbl_safety_verdict.pack(fill=tk.X, pady=2)

        # Row 6: Decomposed Multi-Objective Reward Breakdown
        self.lbl_reward_breakdown = tk.Label(
            card,
            text="Điểm thưởng: Tổng: +0.00 | Nhiệt: +0.00 | Ẩm: +0.00 | CO2: +0.00 | Bụi: +0.00 | Điện: -0.00 | Đổi lệnh: 0.00 | An toàn: 0.00",
            font=("Consolas", 8), fg="#cbd5e1", bg=self.c_card_inner, padx=6, pady=4,
            anchor="w", justify="left", wraplength=575,
            highlightbackground=self.c_border, highlightthickness=1
        )
        self.lbl_reward_breakdown.pack(fill=tk.X, pady=(4, 0))

    # -----------------------------------------------------------------------
    # Section D & H: Charts & Step Decision History (Right Panel)
    # -----------------------------------------------------------------------
    def _build_charts_and_history_section(self, parent):
        # Vertical PanedWindow so both live telemetry charts and the enlarged Step Table scale proportionally
        right_paned = tk.PanedWindow(parent, orient=tk.VERTICAL, bg=self.c_bg, sashwidth=6, bd=0)
        right_paned.pack(fill=tk.BOTH, expand=True)

        top_half = tk.Frame(right_paned, bg=self.c_bg)
        bottom_half = tk.Frame(right_paned, bg=self.c_bg)
        right_paned.add(top_half, minsize=340, stretch="always")
        right_paned.add(bottom_half, minsize=290, stretch="always")

        # Matplotlib 4 Subplots in Modern Dark Theme
        self.fig = Figure(figsize=(7.5, 4.0), dpi=100, facecolor=self.c_bg)
        self.ax_temp = self.fig.add_subplot(221)
        self.ax_rh = self.fig.add_subplot(222)
        self.ax_iaq = self.fig.add_subplot(223)
        self.ax_energy = self.fig.add_subplot(224)

        for ax in [self.ax_temp, self.ax_rh, self.ax_iaq, self.ax_energy]:
            ax.set_facecolor(self.c_card_inner)
            ax.tick_params(colors=self.c_text_muted, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_color(self.c_border)
            ax.grid(True, color=self.c_card, linestyle="--", linewidth=0.5)

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_half)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom Half: Enlarged Step Decision History Table
        hist_header = tk.Frame(bottom_half, bg=self.c_bg)
        hist_header.pack(fill=tk.X, pady=(4, 4))

        tk.Label(
            hist_header, text="📋 LỊCH SỬ QUYẾT ĐỊNH TỪNG BƯỚC & CẤU HÌNH THIẾT BỊ",
            font=("Segoe UI", 10, "bold"), fg=self.c_accent, bg=self.c_bg
        ).pack(side=tk.LEFT)

        tk.Label(
            hist_header, text="(Nhấp vào 1 dòng để xem lại chi tiết trạng thái Q-Table & Thiết lập)",
            font=("Segoe UI", 8, "italic"), fg=self.c_text_muted, bg=self.c_bg
        ).pack(side=tk.RIGHT)

        tree_container = tk.Frame(bottom_half, bg=self.c_bg)
        tree_container.pack(fill=tk.BOTH, expand=True)

        cols = (
            "step", "hour", "temp", "rh", "co2", "pm25",
            "agent_act", "exec_act", "device_cfg", "power_kw", "reward", "override"
        )
        self.tree = ttk.Treeview(tree_container, columns=cols, show="headings", height=11, style="Dark.Treeview")

        self.tree.heading("step", text="Bước")
        self.tree.heading("hour", text="Giờ")
        self.tree.heading("temp", text="Nhiệt (°C)")
        self.tree.heading("rh", text="Ẩm (%)")
        self.tree.heading("co2", text="CO2")
        self.tree.heading("pm25", text="PM2.5")
        self.tree.heading("agent_act", text="AI Đề xuất")
        self.tree.heading("exec_act", text="Lệnh Thực thi")
        self.tree.heading("device_cfg", text="Thiết lập Thiết bị & Cửa sổ")
        self.tree.heading("power_kw", text="Điện (kW)")
        self.tree.heading("reward", text="Điểm")
        self.tree.heading("override", text="An toàn")

        self.tree.column("step", width=40, minwidth=36, anchor="center", stretch=False)
        self.tree.column("hour", width=42, minwidth=38, anchor="center", stretch=False)
        self.tree.column("temp", width=52, minwidth=46, anchor="center", stretch=False)
        self.tree.column("rh", width=46, minwidth=42, anchor="center", stretch=False)
        self.tree.column("co2", width=46, minwidth=42, anchor="center", stretch=False)
        self.tree.column("pm25", width=44, minwidth=40, anchor="center", stretch=False)
        self.tree.column("agent_act", width=135, minwidth=110, anchor="w")
        self.tree.column("exec_act", width=135, minwidth=110, anchor="w")
        self.tree.column("device_cfg", width=215, minwidth=165, anchor="w")
        self.tree.column("power_kw", width=54, minwidth=46, anchor="center", stretch=False)
        self.tree.column("reward", width=52, minwidth=44, anchor="center", stretch=False)
        self.tree.column("override", width=70, minwidth=60, anchor="w")

        tree_vscroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        tree_hscroll = ttk.Scrollbar(tree_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_vscroll.set, xscrollcommand=tree_hscroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_vscroll.grid(row=0, column=1, sticky="ns")
        tree_hscroll.grid(row=1, column=0, sticky="ew")
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select_row)

    # -----------------------------------------------------------------------
    # Simulation Playback Engine
    # -----------------------------------------------------------------------
    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.badge_status.config(text="▶ ĐANG CHẠY", fg=self.c_success)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_step.config(state=tk.DISABLED)
            self._schedule_next_step()

    def pause_simulation(self):
        if self.is_running and not self.is_paused:
            self.is_paused = True
            self.is_running = False
            if self.timer_job:
                self.root.after_cancel(self.timer_job)
                self.timer_job = None
            self.badge_status.config(text="⏸ TẠM DỪNG", fg=self.c_warn)
            self.btn_start.config(state=tk.NORMAL, text="▶ TIẾP TỤC")
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_step.config(state=tk.NORMAL)

    def stop_simulation(self):
        self.is_running = False
        self.is_paused = False
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.badge_status.config(text="⏹ ĐÃ DỪNG", fg="#94a3b8")
        self.btn_start.config(state=tk.NORMAL, text="▶ BẮT ĐẦU")
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_step.config(state=tk.NORMAL)

    def step_forward_simulation(self):
        """Execute precisely ONE environment step and halt."""
        if not self.is_running:
            self._execute_single_step()

    def _schedule_next_step(self):
        if self.is_running and not self.is_paused:
            self._execute_single_step()
            if self.is_running:
                self.timer_job = self.root.after(self.delay_ms, self._schedule_next_step)

    def reset_simulation(self):
        """Reset environment to initial conditions without cross-episode leakage."""
        self.stop_simulation()
        self.q_updates_count = 0
        state_idx, info = self.env.reset(start_idx=10, initial_temp=26.5, initial_rh=60.0)
        self.safety_gateway.reset()

        for k in self.history:
            self.history[k].clear()
        self.step_diagnostics_log.clear()

        for item in self.tree.get_children():
            self.tree.delete(item)

        self._update_telemetry_ui(info, raw_action=None, exec_action=None, reason=None, reward=0.0, r_dict={})
        self._update_charts()

    def _execute_single_step(self):
        """Perform exactly one transition: agent selection -> safety -> env step -> reward -> optional update."""
        current_state = self.env.current_state_idx
        is_eval = not self.online_learning_enabled

        # 1. Agent Decision (select action and capture exploration status without double RNG sampling)
        raw_action = self.current_agent.select_action(current_state, evaluate=is_eval)
        is_exploring = getattr(self.current_agent, "last_action_is_exploring", False) if not is_eval else False

        # Read Q-values for decision trace
        q1_val, q2_val, combined_q = 0.0, 0.0, 0.0
        if isinstance(self.current_agent, MultiDeviceDoubleQAgent):
            q1_val = float(self.current_agent.q_table_a[current_state, raw_action])
            q2_val = float(self.current_agent.q_table_b[current_state, raw_action])
            combined_q = float(self.current_agent.combined_q_table[current_state, raw_action])
        elif isinstance(self.current_agent, SingleQMultiDeviceAgent):
            q1_val = float(self.current_agent.q_table[current_state, raw_action])
            combined_q = q1_val

        # 2. Downstream Safety Gateway Evaluation
        row = self.env.data_df.iloc[self.env.data_idx]
        safe_action, override_reason = self.safety_gateway.filter_action(
            agent_action=raw_action,
            indoor_temp=self.env.current_temp,
            indoor_rh=self.env.current_rh,
            indoor_co2=self.env.current_co2,
            indoor_pm25=self.env.current_pm25,
            ambient_temp=float(row.get("ambient_temp_est", 32.0)),
            ambient_rh=float(row.get("ambient_rh_est", 65.0)),
            ambient_pm25=float(row.get("ambient_pm25_est", 25.0)),
            hour=int(row.get("hour", 12)),
            step=self.env.current_step
        )
        action_to_exec = safe_action

        # 3. Step Environment forward
        next_state, _, done, trunc, step_info = self.env.step(action_to_exec)

        # 4. Compute Multi-Objective Decomposed Reward with explicit is_workhour
        reward, r_dict = self.reward_fn.calculate_reward(
            temp_c=step_info["temp_c"],
            humidity_rh=step_info["humidity_rh"],
            co2_ppm=step_info["co2_ppm"],
            pm25_ug_m3=step_info["pm25_ug_m3"],
            total_power_kw=step_info["powers_kw"]["total_kw"],
            hour=step_info["hour"],
            dayofweek=step_info["dayofweek"],
            is_workhour=step_info.get("is_workhour"),
            action_changed=step_info["action_changed"],
            safety_violation=(override_reason is not None)
        )

        # 5. Optional Online Learning Update (Strictly only when enabled AND agent is learning agent)
        can_learn = getattr(self.current_agent, "is_learning_agent", False)
        if self.online_learning_enabled and can_learn:
            self.current_agent.update(current_state, action_to_exec, reward, next_state, done)
            self.q_updates_count += 1
            if hasattr(self.current_agent, "decay_epsilon"):
                self.current_agent.decay_epsilon()

        # 6. Record history for charts
        self.history["step"].append(step_info["step"])
        self.history["sim_hour"].append(step_info["hour"])
        self.history["temp_c"].append(step_info["temp_c"])
        self.history["amb_temp"].append(step_info["ambient_temp"])
        self.history["rh"].append(step_info["humidity_rh"])
        self.history["amb_rh"].append(step_info["ambient_rh"])
        self.history["co2"].append(step_info["co2_ppm"])
        self.history["pm25"].append(step_info["pm25_ug_m3"])
        self.history["total_kw"].append(step_info["powers_kw"]["total_kw"])
        self.history["cooling_kw"].append(step_info["powers_kw"]["cooling_kw"])
        self.history["vent_kw"].append(step_info["powers_kw"]["ventilation_kw"])
        self.history["filtration_kw"].append(step_info["powers_kw"]["filtration_kw"])
        self.history["cumul_kwh"].append(step_info["cumul_energy_kwh"]["total_kwh"])
        self.history["reward"].append(reward)
        self.history["safe_override"].append(1 if override_reason else 0)

        # Store step diagnostic bundle for interactive tree selection
        step_diag = {
            "step_info": step_info,
            "raw_action": raw_action,
            "exec_action": action_to_exec,
            "reason": override_reason,
            "reward": reward,
            "r_dict": r_dict,
            "is_exploring": is_exploring,
            "q1": q1_val,
            "q2": q2_val,
            "combined_q": combined_q,
            "current_state": current_state
        }
        self.step_diagnostics_log.append(step_diag)

        # Add to Treeview table with clear Vietnamese Action + Device/Window details
        act_name = format_action_short_vi(raw_action)
        exec_name = format_action_short_vi(action_to_exec)
        device_cfg = format_action_devices_vi(action_to_exec)
        self.tree.insert(
            "", 0,
            values=(
                step_info["step"], f"{step_info['hour']:02d}:00",
                f"{step_info['temp_c']:.2f}", f"{step_info['humidity_rh']:.1f}",
                f"{step_info['co2_ppm']:.0f}", f"{step_info['pm25_ug_m3']:.1f}",
                act_name, exec_name, device_cfg, f"{step_info['powers_kw']['total_kw']:.2f}",
                f"{reward:+.2f}", override_reason or "Hợp lệ"
            )
        )

        # 7. Update UI Panels
        self._update_telemetry_ui(
            step_info, raw_action, action_to_exec, override_reason, reward, r_dict,
            is_exploring=is_exploring, q1=q1_val, q2=q2_val, combined_q=combined_q,
            current_state=current_state
        )
        self._update_charts()

        if done:
            self.stop_simulation()
            messagebox.showinfo("Hoàn tất mô phỏng", f"Đã hoàn thành 1 phiên đầy đủ ({self.env.episode_length} bước giờ)!")

    def _update_telemetry_ui(
        self, info, raw_action, exec_action, reason, reward, r_dict,
        is_exploring=False, q1=0.0, q2=0.0, combined_q=0.0, current_state=0
    ):
        """Update KPI cards, physical telemetry badges, equipment cards, and decision trace."""
        step_val = info.get("step", 0)
        hour_val = info.get("hour", 0)
        t_in = info.get("temp_c", 25.0)
        rh_in = info.get("humidity_rh", 55.0)
        co2_in = info.get("co2_ppm", 650.0)
        pm_in = info.get("pm25_ug_m3", 18.0)
        pows = info.get("powers_kw", {})
        cumul = info.get("cumul_energy_kwh", {})
        eff = info.get("effects", {})

        # KPI Updates with color-coded alerting
        self.kpi_temp["val"].config(
            text=f"{t_in:.2f} °C",
            fg=self.c_success if 24.0 <= t_in <= 26.0 else (self.c_warn if 21.0 <= t_in <= 29.0 else self.c_danger)
        )
        self.kpi_rh["val"].config(
            text=f"{rh_in:.1f} %RH",
            fg=self.c_success if 40.0 <= rh_in <= 65.0 else self.c_warn
        )
        self.kpi_co2["val"].config(
            text=f"{co2_in:.0f} ppm",
            fg=self.c_success if co2_in < 800.0 else (self.c_warn if co2_in <= 1200.0 else self.c_danger)
        )
        self.kpi_pm25["val"].config(
            text=f"{pm_in:.1f} µg/m³",
            fg=self.c_success if pm_in <= 35.0 else self.c_danger
        )
        self.kpi_power["val"].config(text=f"{pows.get('total_kw', 0.0):.2f} kW")
        self.kpi_energy["val"].config(text=f"{cumul.get('total_kwh', 0.0):.1f} kWh")
        self.kpi_time["val"].config(text=f"{hour_val:02d}:00")
        self.kpi_time["sub"].config(text=f"Bước {step_val} / {self.env.episode_length}")

        if self.online_learning_enabled:
            self.kpi_updates["val"].config(text=f"#{self.q_updates_count}", fg=self.c_purple)
            self.kpi_updates["sub"].config(text="Đang học trực tuyến", fg=self.c_purple)
        else:
            self.kpi_updates["val"].config(text="#0", fg=self.c_text_muted)
            self.kpi_updates["sub"].config(text="Đóng băng (Eval)", fg=self.c_text_muted)

        # Environmental Telemetry Cards
        self.lbl_in_temp.config(text=f"Nhiệt độ: {t_in:.2f} °C")
        self.lbl_in_rh.config(text=f"Độ ẩm: {rh_in:.1f} %RH")
        self.lbl_in_co2.config(text=f"CO2: {co2_in:.0f} ppm")
        self.lbl_in_pm25.config(text=f"Bụi PM2.5: {pm_in:.1f} µg/m³")
        self.lbl_in_occ.config(text=f"Số người: {info.get('occupancy_count', 0)} người [SUY LUẬN]")

        self.lbl_out_temp.config(text=f"Nhiệt độ: {info.get('ambient_temp', 32.0):.2f} °C")
        self.lbl_out_rh.config(text=f"Độ ẩm: {info.get('ambient_rh', 65.0):.1f} %RH")
        self.lbl_out_co2.config(text=f"CO2 nền: {info.get('ambient_co2', 415.0):.0f} ppm [CƠ SỞ]")
        self.lbl_out_pm25.config(text=f"Bụi PM2.5: {info.get('ambient_pm25', 25.0):.1f} µg/m³")

        p_int = info.get("internal_heat_kw", 2.0)
        self.lbl_room_p_int.config(text=f"Tải nội sinh: {p_int:.2f} kW [CU-BEMS]")

        # Equipment Status Badges (2-line crisp layout, zero clipping)
        ac_st = eff.get("ac_stage", "OFF")
        ac_st_vi = "TẮT" if ac_st == "OFF" else ac_st.replace("SET_", "") + "°C"
        cooling_kw = pows.get("cooling_kw", 0.0)
        cool_th = eff.get("cooling_thermal_kw", 0.0)
        self.eq_ac.config(text=f"{ac_st_vi} | {cooling_kw:.2f} kW\nNhiệt: {cool_th:.2f} kW")

        ht_st = eff.get("heater_stage", "OFF")
        ht_st_vi = "TẮT" if ht_st == "OFF" else ht_st
        heat_kw = pows.get("heating_kw", 0.0)
        heat_th = eff.get("heater_thermal_kw", 0.0)
        self.eq_heat.config(text=f"{ht_st_vi} | {heat_kw:.2f} kW\nNhiệt: {heat_th:.2f} kW")

        v_flow = eff.get("vent_flow_m3_h", 0.0)
        self.eq_vent.config(text=f"{v_flow:.0f} m³/h\n{pows.get('ventilation_kw', 0.0):.2f} kW")

        w_flow = eff.get("window_flow_m3_h", 0.0)
        w_st = "ĐANG MỞ" if info.get("window_open", 0) == 1 else "ĐANG ĐÓNG"
        self.eq_win.config(text=f"{w_st}\n{w_flow:.0f} m³/h")

        self.eq_humid.config(text=f"{pows.get('humidifier_kw', 0.0):.2f} kW\n+{eff.get('moisture_add_kg_h', 0.0):.2f} kg/h")
        self.eq_dehumid.config(text=f"{pows.get('dehumidifier_kw', 0.0):.2f} kW\n-{eff.get('moisture_remove_kg_h', 0.0):.2f} kg/h")
        self.eq_filter.config(text=f"{eff.get('filter_cadr_m3_h', 0.0):.0f} m³/h\n{pows.get('filtration_kw', 0.0):.2f} kW")
        self.eq_total.config(text=f"{pows.get('total_kw', 0.0):.2f} kW\n{cumul.get('total_kwh', 0.0):.1f} kWh")

        # Agent Decision Trace
        st_tup = info.get("state_tuple", (0, 0, 0, 0, 0, 0))
        self.lbl_agent_state.config(
            text=f"Trạng thái s: [Nhiệt:{st_tup[0]}, Ẩm:{st_tup[1]}, CO2:{st_tup[2]}, Bụi:{st_tup[3]}, Tải:{st_tup[4]}, Giờ:{st_tup[5]}] -> Ô {current_state} / 720"
        )

        if is_exploring:
            self.lbl_agent_mode.config(text="Chế độ: THÁM HIỂM (Ngẫu nhiên Epsilon)", fg=self.c_warn)
        else:
            self.lbl_agent_mode.config(text="Chế độ: KHAI THÁC (Chọn max Q)", fg=self.c_success)

        if self.online_learning_enabled:
            self.lbl_q_update.config(text=f"Cập nhật Q-Table: CÓ (Đang học, #{self.q_updates_count})", fg=self.c_purple)
        else:
            self.lbl_q_update.config(text="Cập nhật Q-Table: KHÔNG (Đóng băng đánh giá)", fg=self.c_text_muted)

        if raw_action is not None:
            raw_vi = format_action_short_vi(raw_action)
            self.lbl_agent_choice.config(
                text=f"AI Đề xuất: [{raw_vi}] -> QA: {q1:.2f} | QB: {q2:.2f} | Q chung: {combined_q:.2f}"
            )
            exec_vi = format_action_short_vi(exec_action)
            dev_vi = format_action_devices_vi(exec_action)
            self.lbl_device_config.config(text=f"Thiết lập thực tế ({exec_vi}): {dev_vi}")
        else:
            self.lbl_agent_choice.config(text="AI Đề xuất: Trạng thái khởi tạo (Nhấn CHẠY 1 BƯỚC hoặc BẮT ĐẦU).")
            self.lbl_device_config.config(text="Thiết lập thiết bị: Đang chờ lệnh ở bước 1.")

        # Scientific, transparent Safety Gateway reporting
        if reason is not None:
            exec_vi = format_action_short_vi(exec_action)
            self.lbl_safety_verdict.config(
                text=f"Cổng An toàn: ⚠️ CAN THIỆP! Chặn [A{raw_action}] -> Thực thi [{exec_vi}] | Lý do: {reason}",
                fg=self.c_danger
            )
        elif raw_action is not None:
            exec_vi = format_action_short_vi(exec_action)
            self.lbl_safety_verdict.config(
                text=f"Cổng An toàn: Hợp lệ (Không vượt ngưỡng nguy hiểm). Đã thực thi: [{exec_vi}]",
                fg=self.c_success
            )
        else:
            self.lbl_safety_verdict.config(
                text="Cổng An toàn: Đang trực. Sẵn sàng kiểm tra bước tiếp theo.",
                fg=self.c_text_muted
            )

        if r_dict:
            self.lbl_reward_breakdown.config(
                text=f"Điểm thưởng: Tổng: {reward:+.2f} | Nhiệt: {r_dict.get('reward_temp', 0):+.2f} | Ẩm: {r_dict.get('reward_humidity', 0):+.2f} | "
                     f"CO2: {r_dict.get('reward_co2', 0):+.2f} | Bụi: {r_dict.get('reward_air_quality', 0):+.2f} | Điện: {r_dict.get('reward_energy', 0):+.2f} | "
                     f"Đổi lệnh: {r_dict.get('reward_switching', 0):+.2f} | An toàn: {r_dict.get('reward_safety', 0):+.2f}"
            )

    def _update_charts(self):
        """Re-render the 4 live matplotlib telemetry graphs with dark industrial styling."""
        for ax in [self.ax_temp, self.ax_rh, self.ax_iaq, self.ax_energy]:
            ax.clear()
            ax.set_facecolor(self.c_card_inner)
            ax.tick_params(colors=self.c_text_muted, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(self.c_border)
            ax.grid(True, color=self.c_card, linestyle="--", linewidth=0.5)

        steps = self.history["step"]
        if not steps:
            self.canvas.draw_idle()
            return

        # 1. Temperature Subplot
        self.ax_temp.plot(steps, self.history["temp_c"], label="Nhiệt độ phòng", color=self.c_accent, linewidth=1.6)
        self.ax_temp.plot(steps, self.history["amb_temp"], label="Ngoài trời", color=self.c_warn, linestyle=":", linewidth=1.1)
        self.ax_temp.axhspan(24.0, 26.0, color=self.c_success, alpha=0.15, label="Mục tiêu [24,26]°C")
        self.ax_temp.set_title("Nhiệt độ phòng vs Ngoài trời (°C)", fontsize=8.5, color=self.c_text, pad=3)
        self.ax_temp.legend(fontsize=6.5, loc="upper right", facecolor=self.c_card, edgecolor=self.c_border, labelcolor=self.c_text)

        # 2. Humidity Subplot
        self.ax_rh.plot(steps, self.history["rh"], label="Độ ẩm phòng", color="#67e8f9", linewidth=1.6)
        self.ax_rh.plot(steps, self.history["amb_rh"], label="Ngoài trời", color=self.c_warn, linestyle=":", linewidth=1.1)
        self.ax_rh.axhspan(40.0, 65.0, color=self.c_success, alpha=0.15, label="Thoải mái [40,65]%")
        self.ax_rh.set_title("Độ ẩm tương đối (%RH)", fontsize=8.5, color=self.c_text, pad=3)
        self.ax_rh.legend(fontsize=6.5, loc="upper right", facecolor=self.c_card, edgecolor=self.c_border, labelcolor=self.c_text)

        # 3. IAQ (CO2 & PM2.5) Subplot
        self.ax_iaq.plot(steps, self.history["co2"], label="CO2 (ppm)", color=self.c_success, linewidth=1.6)
        self.ax_iaq.axhline(800.0, color=self.c_warn, linestyle="--", linewidth=0.9, label="Trong lành (800)")
        self.ax_iaq.axhline(1200.0, color=self.c_danger, linestyle="--", linewidth=0.9, label="Ngưỡng max (1200)")
        self.ax_iaq.set_title("Nồng độ khí CO2 trong phòng (ppm)", fontsize=8.5, color=self.c_text, pad=3)
        self.ax_iaq.legend(fontsize=6.5, loc="upper right", facecolor=self.c_card, edgecolor=self.c_border, labelcolor=self.c_text)

        # 4. Power Breakdown Subplot
        self.ax_energy.plot(steps, self.history["total_kw"], label="Tổng công suất", color="#f43f5e", linewidth=1.6)
        self.ax_energy.plot(steps, self.history["cooling_kw"], label="Máy lạnh (AC)", color=self.c_accent, linestyle="--", linewidth=1.1)
        self.ax_energy.plot(steps, self.history["vent_kw"], label="Quạt gió tươi", color=self.c_success, linestyle=":", linewidth=1.1)
        self.ax_energy.set_title("Công suất tiêu thụ điện (kW)", fontsize=8.5, color=self.c_text, pad=3)
        self.ax_energy.legend(fontsize=6.5, loc="upper right", facecolor=self.c_card, edgecolor=self.c_border, labelcolor=self.c_text)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Interactive Event Handlers
    # -----------------------------------------------------------------------
    def _on_scenario_changed(self, event=None):
        choice = self.scenario_var.get()
        if "Nhỏ 20m²" in choice or "Small 20m²" in choice:
            self.env = MultiDeviceHVACEnv(room=ROOM_PRESETS["small"], episode_length=48, scenario_type="heatwave", seed=42)
            self.scenario_desc_lbl.config(
                text="ℹ️ Diện tích 20m², cách nhiệt U=0.65 W/m²K, tải 1-2 người. Nguồn: Mô phỏng nắng nóng 30-38°C."
            )
        elif "Lớn 120m²" in choice or "Large 120m²" in choice:
            self.env = MultiDeviceHVACEnv(room=ROOM_PRESETS["large"], episode_length=48, scenario_type="large_occupancy", seed=42)
            self.scenario_desc_lbl.config(
                text="ℹ️ Diện tích 120m², cách nhiệt U=0.45 W/m²K, tải 8-10 người. Nguồn: Mô phỏng hội trường đông người."
            )
        elif "Ô nhiễm" in choice or "PM2.5" in choice:
            self.env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], episode_length=48, scenario_type="smog_pollution", seed=101)
            self.scenario_desc_lbl.config(
                text="ℹ️ Sương mù ngoại cảnh: Bụi PM2.5 > 60 µg/m³, độ ẩm > 75%. Nguồn: Kịch bản ô nhiễm & mưa ẩm."
            )
        else:
            self.env = MultiDeviceHVACEnv(data_df=self.cubems_df, room=ROOM_PRESETS["medium"], episode_length=48, scenario_type="standard", seed=42)
            src_name = "CU-BEMS 2018 (Tải thực tòa nhà & khí hậu hiệu chuẩn)" if self.cubems_df is not None else "MÔ PHỎNG CHUẨN"
            self.scenario_desc_lbl.config(
                text=f"ℹ️ Diện tích 50m², cách nhiệt U=0.55 W/m²K, tải 3-4 người. Nguồn dữ liệu: {src_name}."
            )
        self.reset_simulation()

    def _update_badge_status(self):
        if not hasattr(self, "badge_mode"):
            return
        is_dql = "Double Q-Learning" in self.agent_name
        is_sql = "Q-Learning" in self.agent_name and not is_dql
        if (is_dql and not self.is_dql_checkpoint_loaded) or (is_sql and not self.is_sql_checkpoint_loaded):
            self.badge_mode.config(text="⚠️ TÁC TỬ CHƯA HUẤN LUYỆN", fg=self.c_warn)
        elif self.online_learning_enabled:
            self.badge_mode.config(text="⚡ HỌC TRỰC TUYẾN ĐANG BẬT", fg=self.c_purple)
        else:
            self.badge_mode.config(text="🔒 CHẾ ĐỘ ĐÁNH GIÁ (Đóng băng Q-Table)", fg=self.c_accent)

    def _on_agent_changed(self, event=None):
        name = self.agent_var.get()
        self.agent_name = name
        self.current_agent = self.agents_dict[name]

        can_learn = getattr(self.current_agent, "is_learning_agent", False)
        if not can_learn:
            self.online_learn_chk.config(state=tk.DISABLED)
            self.online_learn_var.set(False)
            self.online_learning_enabled = False
            self._update_badge_status()
        else:
            self.online_learn_chk.config(state=tk.NORMAL)
            self._update_badge_status()

        if "Double Q-Learning" in name:
            self.agent_desc_lbl.config(
                text="ℹ️ Double Q-Learning: 720 trạng thái x 12 hành động, dùng 2 bảng QA/QB khử thiên lệch. Hỗ trợ học trực tuyến."
            )
        elif "Q-Learning đơn" in name or "Single Q-Learning" in name:
            self.agent_desc_lbl.config(
                text="ℹ️ Q-Learning đơn: Dùng 1 bảng Q duy nhất, dễ dao động hơn theo seed. Hỗ trợ học trực tuyến."
            )
        elif "IF-THEN" in name or "Rule-Based" in name:
            self.agent_desc_lbl.config(
                text="ℹ️ Bộ luật IF-THEN: Luật chuyên gia theo ngưỡng cố định T/RH/CO2/PM2.5. (Không có Q-Table)"
            )
        elif "Rơ-le" in name or "Conventional" in name:
            self.agent_desc_lbl.config(
                text="ℹ️ Rơ-le truyền thống: Bật/tắt theo dải chết nhiệt độ [21.0, 26.0]°C, không điều tiết CO2/ẩm."
            )
        elif "25°C" in name:
            self.agent_desc_lbl.config(
                text="ℹ️ Đặt cố định 25°C: Luôn chạy máy lạnh 25°C + gió nhẹ 150 m³/h suốt 24/24h."
            )
        else:
            self.agent_desc_lbl.config(text="ℹ️ Chính sách ngẫu nhiên: Chọn bừa 1/12 hành động làm mốc sàn kiểm chứng.")

    def _on_speed_changed(self, event=None):
        speed_str = self.speed_var.get()
        if "0.25x" in speed_str:
            self.delay_ms = 2000
        elif "0.5x" in speed_str:
            self.delay_ms = 1000
        elif "1.0x" in speed_str:
            self.delay_ms = 500
        elif "2.0x" in speed_str:
            self.delay_ms = 250
        elif "5.0x" in speed_str:
            self.delay_ms = 100

    def _on_online_learning_toggle(self):
        can_learn = getattr(self.current_agent, "is_learning_agent", False)
        if not can_learn:
            self.online_learn_var.set(False)
            self.online_learning_enabled = False
            messagebox.showinfo("Học trực tuyến", "Bộ điều khiển hiện tại là luật cố định, không có bảng Q để cập nhật.")
            return

        self.online_learning_enabled = self.online_learn_var.get()
        self._update_badge_status()
        if self.online_learning_enabled:
            self.lbl_q_update.config(
                text=f"Cập nhật Q-Table: CÓ (Đang học trực tuyến, #{self.q_updates_count})",
                fg=self.c_purple
            )
        else:
            self.lbl_q_update.config(
                text="Cập nhật Q-Table: KHÔNG (Đóng băng đánh giá)",
                fg=self.c_text_muted
            )

    def _on_tree_select_row(self, event=None):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        item_id = selected_items[0]
        vals = self.tree.item(item_id, "values")
        if not vals:
            return
        step_clicked = int(vals[0])

        # Find matching historical diagnostic bundle
        for diag in self.step_diagnostics_log:
            if diag["step_info"]["step"] == step_clicked:
                self._update_telemetry_ui(
                    diag["step_info"],
                    diag["raw_action"],
                    diag["exec_action"],
                    diag["reason"],
                    diag["reward"],
                    diag["r_dict"],
                    is_exploring=diag["is_exploring"],
                    q1=diag["q1"],
                    q2=diag["q2"],
                    combined_q=diag["combined_q"],
                    current_state=diag["current_state"]
                )
                break

    def _tick_wall_clock(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.root.title(f"Smart HVAC Digital Twin | Phòng Điều Khiển Mô Phỏng | Giờ hệ thống: {now_str}")
        self.root.after(1000, self._tick_wall_clock)


def main():
    root = tk.Tk()
    app = LiveSimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
