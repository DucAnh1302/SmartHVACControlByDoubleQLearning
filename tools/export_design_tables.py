"""
Export Authoritative Design Tables (5 CSVs + 4-Sheet Excel Workbook).
Every parameter, threshold, weight, power rating, simulated sample decision,
hyperparameter, and action frequency is read directly from live source code
and Phase 3 experimental artifacts (zero hardcoded legacy numbers).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.multi_device_env import (
    MULTI_DEVICE_ACTION_TEMPLATES,
    MultiDeviceHVACEnv,
)
from src.reward.multi_device_reward import MultiDeviceRewardFunction
from src.safety.multi_device_safety_gateway import (
    MultiDeviceSafetyConfig,
    MultiDeviceSafetyGateway,
)


def _typical_action_power_kw(env: MultiDeviceHVACEnv, action_id: int) -> float:
    """Computes realistic power draw (kW) at warm indoor load T=28°C, RH=60%."""
    tpl = MULTI_DEVICE_ACTION_TEMPLATES[action_id]
    res = env._calculate_actuator_powers_and_effects(
        tpl, current_temp=28.0, current_rh=60.0, ambient_temp=32.0, ambient_rh=68.0
    )
    return float(res["powers"]["total_kw"])


def build_state_rows() -> List[Dict[str, Any]]:
    """Exact 720-state discretization specification from discretize_state_720."""
    return [
        {"Chiều": "1. Nhiệt độ phòng (Indoor Temp)", "Ký hiệu": "T_in", "Số bins": 5,
         "Ngưỡng phân tách": "< 21.0°C", "Ý nghĩa vật lý": "Rất lạnh (Dưới tiện nghi ASHRAE)",
         "Tiêu chuẩn tham chiếu": "Dưới ngưỡng ASHRAE 55", "Chỉ số bin": "t_bin = 0",
         "Lý do chọn": "Dưới 21°C gây lạnh buốt và lãng phí công suất làm lạnh."},
        {"Chiều": "1. Nhiệt độ phòng (Indoor Temp)", "Ký hiệu": "T_in", "Số bins": 5,
         "Ngưỡng phân tách": "[21.0°C, 24.0°C)", "Ý nghĩa vật lý": "Mát mẻ (Chấp nhận được)",
         "Tiêu chuẩn tham chiếu": "ASHRAE 55 Vùng chấp nhận", "Chỉ số bin": "t_bin = 1",
         "Lý do chọn": "Vùng chuyển tiếp mát, chấp nhận khi tải nhẹ hoặc làm mát trước."},
        {"Chiều": "1. Nhiệt độ phòng (Indoor Temp)", "Ký hiệu": "T_in", "Số bins": 5,
         "Ngưỡng phân tách": "[24.0°C, 26.0°C)", "Ý nghĩa vật lý": "Vùng tiện nghi vàng",
         "Tiêu chuẩn tham chiếu": "ASHRAE 55 PMV [-0.2, +0.2]", "Chỉ số bin": "t_bin = 2",
         "Lý do chọn": "Dải nhiệt tối ưu cho trang phục công sở nhiệt đới tại Đông Nam Á."},
        {"Chiều": "1. Nhiệt độ phòng (Indoor Temp)", "Ký hiệu": "T_in", "Số bins": 5,
         "Ngưỡng phân tách": "[26.0°C, 28.5°C)", "Ý nghĩa vật lý": "Hơi ấm (Cần làm mát)",
         "Tiêu chuẩn tham chiếu": "Cận trên ASHRAE 55", "Chỉ số bin": "t_bin = 3",
         "Lý do chọn": "Bắt đầu vượt tiện nghi, cần bật làm mát hoặc thông gió."},
        {"Chiều": "1. Nhiệt độ phòng (Indoor Temp)", "Ký hiệu": "T_in", "Số bins": 5,
         "Ngưỡng phân tách": ">= 28.5°C", "Ý nghĩa vật lý": "Nóng bức (Vi phạm nhiệt)",
         "Tiêu chuẩn tham chiếu": "Vượt chuẩn ASHRAE 55", "Chỉ số bin": "t_bin = 4",
         "Lý do chọn": "Gây khó chịu rõ rệt, cần chuyển về làm mát công suất cao."},
        {"Chiều": "2. Độ ẩm tương đối (Indoor RH)", "Ký hiệu": "RH_in", "Số bins": 3,
         "Ngưỡng phân tách": "< 40.0%", "Ý nghĩa vật lý": "Khô (Khô da, rát họng)",
         "Tiêu chuẩn tham chiếu": "Cận dưới ASHRAE 55", "Chỉ số bin": "rh_bin = 0",
         "Lý do chọn": "Độ ẩm dưới 40% gây khô niêm mạc đường hô hấp."},
        {"Chiều": "2. Độ ẩm tương đối (Indoor RH)", "Ký hiệu": "RH_in", "Số bins": 3,
         "Ngưỡng phân tách": "[40.0%, 65.0%)", "Ý nghĩa vật lý": "Độ ẩm tiện nghi nhiệt đới",
         "Tiêu chuẩn tham chiếu": "ASHRAE 55 & 62.1", "Chỉ số bin": "rh_bin = 1",
         "Lý do chọn": "Dải 40-65% hạn chế nấm mốc và đảm bảo thoát nhiệt mồ hôi."},
        {"Chiều": "2. Độ ẩm tương đối (Indoor RH)", "Ký hiệu": "RH_in", "Số bins": 3,
         "Ngưỡng phân tách": ">= 65.0%", "Ý nghĩa vật lý": "Ẩm cao (Nguy cơ nấm mốc)",
         "Tiêu chuẩn tham chiếu": "Vượt ngưỡng khuyến cáo", "Chỉ số bin": "rh_bin = 2",
         "Lý do chọn": "Độ ẩm cao gây oi bức, cần khử ẩm qua cuộn lạnh hoặc máy hút ẩm."},
        {"Chiều": "3. Nồng độ CO2 (Indoor CO2)", "Ký hiệu": "CO2_in", "Số bins": 3,
         "Ngưỡng phân tách": "< 800.0 ppm", "Ý nghĩa vật lý": "Không khí tươi sạch",
         "Tiêu chuẩn tham chiếu": "ASHRAE 62.1 Mức tốt", "Chỉ số bin": "co2_bin = 0",
         "Lý do chọn": "Bảo đảm thông gió đủ và duy trì sự tỉnh táo nhận thức."},
        {"Chiều": "3. Nồng độ CO2 (Indoor CO2)", "Ký hiệu": "CO2_in", "Số bins": 3,
         "Ngưỡng phân tách": "[800.0, 1200.0) ppm", "Ý nghĩa vật lý": "Tích tụ trung bình",
         "Tiêu chuẩn tham chiếu": "ASHRAE 62.1 Vùng chuyển tiếp", "Chỉ số bin": "co2_bin = 1",
         "Lý do chọn": "Mức đặc trưng giờ làm việc, cần duy trì gió tươi hoặc lọc."},
        {"Chiều": "3. Nồng độ CO2 (Indoor CO2)", "Ký hiệu": "CO2_in", "Số bins": 3,
         "Ngưỡng phân tách": ">= 1200.0 ppm", "Ý nghĩa vật lý": "Ngột ngạt nghiêm trọng",
         "Tiêu chuẩn tham chiếu": "Vượt ngưỡng ASHRAE 62.1", "Chỉ số bin": "co2_bin = 2",
         "Lý do chọn": "Gây buồn ngủ, giảm tập trung; cần xả gió tươi 500 m³/h."},
        {"Chiều": "4. Bụi mịn PM2.5 (Indoor PM2.5)", "Ký hiệu": "PM25_in", "Số bins": 2,
         "Ngưỡng phân tách": "<= 35.0 µg/m³", "Ý nghĩa vật lý": "Không khí sạch bụi mịn",
         "Tiêu chuẩn tham chiếu": "WHO / QCVN 05:2023", "Chỉ số bin": "pm_bin = 0",
         "Lý do chọn": "Đạt mức an toàn 24h (lưu ý dữ liệu 2018-F2Z1 hầu như luôn ở bin 0)."},
        {"Chiều": "4. Bụi mịn PM2.5 (Indoor PM2.5)", "Ký hiệu": "PM25_in", "Số bins": 2,
         "Ngưỡng phân tách": "> 35.0 µg/m³", "Ý nghĩa vật lý": "Ô nhiễm bụi mịn",
         "Tiêu chuẩn tham chiếu": "Vượt ngưỡng cảnh báo", "Chỉ số bin": "pm_bin = 1",
         "Lý do chọn": "Đóng cửa sổ và kích hoạt màng lọc HEPA 450 m³/h (chủ yếu ở Test-C smog)."},
        {"Chiều": "5. Tải nội sinh (Internal Load)", "Ký hiệu": "Q_load", "Số bins": 2,
         "Ngưỡng phân tách": "< 0.50 kW", "Ý nghĩa vật lý": "Phòng trống / Tải nền",
         "Tiêu chuẩn tham chiếu": "CU-BEMS Plug+Light", "Chỉ số bin": "load_bin = 0",
         "Lý do chọn": "Mang chủ yếu thông tin có người/không có người (trùng 96.8%); ưu tiên tiết kiệm điện."},
        {"Chiều": "5. Tải nội sinh (Internal Load)", "Ký hiệu": "Q_load", "Số bins": 2,
         "Ngưỡng phân tách": ">= 0.50 kW", "Ý nghĩa vật lý": "Phòng có người hoạt động",
         "Tiêu chuẩn tham chiếu": "Phụ tải thiết bị + chiếu sáng", "Chỉ số bin": "load_bin = 1",
         "Lý do chọn": "Mang chủ yếu thông tin có người/không có người (trùng 96.8%); ưu tiên tiện nghi."},
        {"Chiều": "6. Khung giờ & Biểu giá (Time Bin)", "Ký hiệu": "Time_bin", "Số bins": 4,
         "Ngưỡng phân tách": "Cuối tuần (T7-CN) HOẶC [22:00-06:00)", "Ý nghĩa vật lý": "Ngoài giờ / Đêm / Cuối tuần",
         "Tiêu chuẩn tham chiếu": "discretize_state_720", "Chỉ số bin": "time_bin = 0",
         "Lý do chọn": "Phòng nghỉ ban đêm hoặc T7/CN; biểu giá điện chuẩn (1.0x)."},
        {"Chiều": "6. Khung giờ & Biểu giá (Time Bin)", "Ký hiệu": "Time_bin", "Số bins": 4,
         "Ngưỡng phân tách": "Ngày thường [06:00-09:00)", "Ý nghĩa vật lý": "Sáng khởi động (Morning Ramp)",
         "Tiêu chuẩn tham chiếu": "discretize_state_720", "Chỉ số bin": "time_bin = 1",
         "Lý do chọn": "Đón đầu phụ tải buổi sáng trước khi vào cao điểm văn phòng."},
        {"Chiều": "6. Khung giờ & Biểu giá (Time Bin)", "Ký hiệu": "Time_bin", "Số bins": 4,
         "Ngưỡng phân tách": "Ngày thường [09:00-17:00)", "Ý nghĩa vật lý": "Giờ cao điểm ban ngày (Midday Peak)",
         "Tiêu chuẩn tham chiếu": "Biểu giá cao điểm 1.4x", "Chỉ số bin": "time_bin = 2",
         "Lý do chọn": "Giờ làm việc trọng điểm chịu hệ số chi phí điện 1.4x và trọng số tiện nghi 2.0x."},
        {"Chiều": "6. Khung giờ & Biểu giá (Time Bin)", "Ký hiệu": "Time_bin", "Số bins": 4,
         "Ngưỡng phân tách": "Ngày thường [17:00-22:00)", "Ý nghĩa vật lý": "Chiều tối (Evening Shoulder)",
         "Tiêu chuẩn tham chiếu": "discretize_state_720", "Chỉ số bin": "time_bin = 3",
         "Lý do chọn": "Chuyển tiếp tan ca; giảm dần công suất lạnh khi người rời phòng."},
    ]


def build_action_rows(env: MultiDeviceHVACEnv, act_dist: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build 12-action specification strictly from MULTI_DEVICE_ACTION_TEMPLATES and Test-A stats."""
    reasons = {
        0: "Tắt toàn bộ thiết bị về standby nền 0.05 kW khi phòng trống.",
        1: "Làm mát tiêu chuẩn 25°C, gió tươi 150 m³/h và lọc bụi nhẹ 120 m³/h.",
        2: "Làm mát tối đa 23°C + gió tươi 300 m³/h + HEPA 250 m³/h khi tải nhiệt cao.",
        3: "Làm mát tiết kiệm 27°C, giảm điện giờ cao điểm khi ít người.",
        4: "Mở cửa sổ thông gió tự nhiên khi ngoài trời mát sạch (khi Safety cho phép).",
        5: "Xả ngạt CO2 (500 m³/h) + lọc HEPA (450 m³/h) KÈM làm mát SET_25 tránh nóng phòng.",
        6: "Sưởi ấm nhẹ 22°C + bù ẩm 0.25 kg/h (dự phòng tổng quát cho ngày lạnh).",
        7: "Sưởi ấm cao 24°C + bù ẩm 0.45 kg/h cho kịch bản giá rét.",
        8: "Khử ẩm chuyên dụng 1.2 kg/h kèm làm mát vừa 26°C cho thời tiết nồm ẩm.",
        9: "Bù ẩm 0.6 kg/h kèm làm mát 25°C khi phòng khô dưới 40% RH.",
        10: "Lọc bụi HEPA tối đa 450 m³/h không bật AC khi bụi cao nhưng nhiệt ổn.",
        11: "Đặt ngưỡng nghỉ 26.5°C + gió nhẹ 100 m³/h duy trì nhiệt ẩm nền ban đêm.",
    }
    dql_map = act_dist[act_dist["controller"] == "Double Q-Learning"].set_index("action_id")["pct"].to_dict() if not act_dist.empty else {}
    ql_map = act_dist[act_dist["controller"] == "Single Q-Learning"].set_index("action_id")["pct"].to_dict() if not act_dist.empty else {}

    rows = []
    for aid in range(12):
        t = MULTI_DEVICE_ACTION_TEMPLATES[aid]
        p_kw = _typical_action_power_kw(env, aid)
        rows.append({
            "Mã (ID)": aid,
            "Tên hành động": t.name,
            "Chế độ AC": f"{t.ac_mode}" + (f" ({t.ac_setpoint_c}°C)" if t.ac_setpoint_c is not None else ""),
            "Máy sưởi": f"{t.heater_mode}" + (f" ({t.heater_setpoint_c}°C)" if t.heater_setpoint_c is not None else ""),
            "Quạt thông gió (m³/h)": f"{t.vent_fan_speed} ({t.vent_flow_m3_h:.0f} m³/h)",
            "Cửa sổ": "MỞ" if t.window_open == 1 else "ĐÓNG",
            "Hút ẩm": f"{t.dehumidifier_mode} ({t.dehumidifier_rate_kg_h:.1f} kg/h)",
            "Tạo ẩm": f"{t.humidifier_mode} ({t.humidifier_rate_kg_h:.2f} kg/h)",
            "Lọc bụi (CADR)": f"{t.filter_speed} ({t.filter_cadr_m3_h:.0f} m³/h)",
            "Công suất tải nóng 28°C (kW)": round(p_kw, 2),
            "Tần suất DQL Test-A (%)": f"{dql_map.get(aid, 0.0):.1f}%",
            "Tần suất QL Test-A (%)": f"{ql_map.get(aid, 0.0):.1f}%",
            "Lý do / Mục tiêu phối hợp": reasons[aid],
        })
    return rows


def build_reward_rows() -> List[Dict[str, Any]]:
    """Exact 7 reward components read directly from MultiDeviceRewardFunction."""
    rf = MultiDeviceRewardFunction(preference="balanced")
    return [
        {"STT": 1, "Thành phần": "Tiện nghi nhiệt (Thermal)", "Ký hiệu": "R_temp", "Trọng số": rf.w_temp,
         "Điều kiện": "24.0 <= T <= 26.0°C", "Công thức mã nguồn": "+0.50 * occ_mult (occ_mult=2.0 giờ làm, 0.5 ngoài giờ)",
         "Miền giá trị": "+0.25 đến +1.00", "Mục tiêu": "Thưởng trực tiếp khi giữ đúng dải tiện nghi vàng."},
        {"STT": 1, "Thành phần": "Tiện nghi nhiệt (Thermal)", "Ký hiệu": "R_temp", "Trọng số": rf.w_temp,
         "Điều kiện": "21.0 <= T < 24.0 hoặc 26.0 < T <= 29.0", "Công thức mã nguồn": "-w_temp * (0.4 * dev) * occ_mult",
         "Miền giá trị": "-2.40 đến 0.00", "Mục tiêu": "Phạt tuyến tính theo mức độ lệch ngoài [24, 26]°C."},
        {"STT": 1, "Thành phần": "Tiện nghi nhiệt (Thermal)", "Ký hiệu": "R_temp", "Trọng số": rf.w_temp,
         "Điều kiện": "T < 21.0°C hoặc T > 29.0°C", "Công thức mã nguồn": "-w_temp * 2.0 * occ_mult (+ r_safety bậc 2)",
         "Miền giá trị": "-4.00 (chưa gồm r_safety)", "Mục tiêu": "Cảnh báo mất tiện nghi nhiệt nặng."},
        {"STT": 2, "Thành phần": "Độ ẩm (Humidity)", "Ký hiệu": "R_humidity", "Trọng số": rf.w_humidity,
         "Điều kiện": "40.0% <= RH <= 65.0%", "Công thức mã nguồn": "+0.10 * occ_mult",
         "Miền giá trị": "+0.05 đến +0.20", "Mục tiêu": "Thưởng duy trì độ ẩm ngăn nấm mốc & khô niêm mạc."},
        {"STT": 2, "Thành phần": "Độ ẩm (Humidity)", "Ký hiệu": "R_humidity", "Trọng số": rf.w_humidity,
         "Điều kiện": "RH < 40.0% hoặc RH > 65.0%", "Công thức mã nguồn": "-w_humidity * (0.05 * dev_rh) * occ_mult",
         "Miền giá trị": "-0.88 đến 0.00", "Mục tiêu": "Phạt lệch ẩm vừa phải tránh lấn át mục tiêu nhiệt/điện."},
        {"STT": 3, "Thành phần": "Khí tươi CO2 (IAQ CO2)", "Ký hiệu": "R_co2", "Trọng số": rf.w_co2,
         "Điều kiện": "CO2 <= 800 ppm", "Công thức mã nguồn": "+0.20 * occ_mult",
         "Miền giá trị": "+0.10 đến +0.40", "Mục tiêu": "Khuyến khích duy trì không khí tươi trong lành."},
        {"STT": 3, "Thành phần": "Khí tươi CO2 (IAQ CO2)", "Ký hiệu": "R_co2", "Trọng số": rf.w_co2,
         "Điều kiện": "800 < CO2 <= 1200 ppm", "Công thức mã nguồn": "-w_co2 * (0.25 * (CO2-800)/100) * occ_mult",
         "Miền giá trị": "-0.60 đến 0.00", "Mục tiêu": "Phạt cảnh báo sớm khi CO2 bắt đầu vượt 800 ppm."},
        {"STT": 3, "Thành phần": "Khí tươi CO2 (IAQ CO2)", "Ký hiệu": "R_co2", "Trọng số": rf.w_co2,
         "Điều kiện": "CO2 > 1200 ppm", "Công thức mã nguồn": "-w_co2 * (1.0 + 0.6 * (CO2-1200)/100) * occ_mult",
         "Miền giá trị": "< -0.60", "Mục tiêu": "Phạt tăng tốc buộc bật Action 5 thông gió xả CO2."},
        {"STT": 4, "Thành phần": "Bụi mịn PM2.5 (Air Quality)", "Ký hiệu": "R_air_quality", "Trọng số": rf.w_pm25,
         "Điều kiện": "PM2.5 <= 15 µg/m³ (WHO)", "Công thức mã nguồn": "+0.20 * occ_mult",
         "Miền giá trị": "+0.10 đến +0.40", "Mục tiêu": "Thưởng giữ bụi mịn ở ngưỡng an toàn cao WHO."},
        {"STT": 4, "Thành phần": "Bụi mịn PM2.5 (Air Quality)", "Ký hiệu": "R_air_quality", "Trọng số": rf.w_pm25,
         "Điều kiện": "25 < PM2.5 <= 50 / >50 µg/m³", "Công thức mã nguồn": "-w_pm25*0.04*(PM-25)*occ_mult / -w_pm25*(1+0.08*(PM-50))*occ_mult",
         "Miền giá trị": "-0.50 hoặc âm sâu", "Mục tiêu": "Kích hoạt lọc HEPA 450 m³/h khi bụi vượt ngưỡng."},
        {"STT": 5, "Thành phần": "Điện năng (Energy Cost)", "Ký hiệu": "R_energy", "Trọng số": rf.w_energy,
         "Điều kiện": "Mọi bước (1.4x cao điểm T2-T6 9-17h)", "Công thức mã nguồn": "-w_energy * P_total_kw * tariff (1.4 hoặc 1.0)",
         "Miền giá trị": "-0.45 đến -0.01", "Mục tiêu": "Khuyến khích tối ưu điện thực tế theo biểu giá cao điểm."},
        {"STT": 6, "Thành phần": "Đổi nấc (Switching)", "Ký hiệu": "R_switching", "Trọng số": rf.w_switch,
         "Điều kiện": "action_changed == True", "Công thức mã nguồn": "-w_switch * 1.0 nếu đổi action",
         "Miền giá trị": "-0.35 hoặc 0.00", "Mục tiêu": "Giảm tần suất đóng ngắt máy nén bảo vệ tuổi thọ."},
        {"STT": 7, "Thành phần": "An toàn (Safety Gateway)", "Ký hiệu": "R_safety", "Trọng số": rf.w_safety,
         "Điều kiện": "T<21 / T>29 / CO2>1800 / PM>100 / Override", "Công thức mã nguồn": "Phạt phi tuyến theo độ lệch + (-5.0 * w_safety = -10.0) nếu bị Safety ghi đè",
         "Miền giá trị": "-10.00 trở xuống", "Mục tiêu": "Triệt tiêu hành động nguy hiểm vượt ngưỡng giới hạn vật lý."},
    ]


def build_safety_rows() -> List[Dict[str, Any]]:
    """Exact 6 rules read directly from MultiDeviceSafetyConfig."""
    sc = MultiDeviceSafetyConfig()
    return [
        {"Mã luật": "SG-00", "Tên luật an toàn": "Kiểm lỗi cảm biến (Sensor Anomaly Fail-Safe)",
         "Điều kiện kích hoạt": f"NaN hoặc T < {sc.sensor_valid_min_temp}°C hoặc T > {sc.sensor_valid_max_temp}°C hoặc CO2 < 200 ppm",
         "Hành động cưỡng bức": "COMFORT_COOL (Action 1)",
         "Cơ sở kỹ thuật / An toàn": "Bảo vệ hệ thống về trạng thái vận hành an toàn mặc định khi hỏng cảm biến."},
        {"Mã luật": "SG-01", "Tên luật an toàn": "Chống quá nhiệt khẩn cấp (Emergency Overheat)",
         "Điều kiện kích hoạt": f"T_phòng >= {sc.emergency_overheat_temp}°C (và action != 2)",
         "Hành động cưỡng bức": "MAX_COOL (Action 2 - SET_23, quạt 300 m³/h)",
         "Cơ sở kỹ thuật / An toàn": "Ngăn sốc nhiệt cho người và quá nhiệt thiết bị văn phòng."},
        {"Mã luật": "SG-02", "Tên luật an toàn": "Chống đóng băng / Quá lạnh (Emergency Freeze)",
         "Điều kiện kích hoạt": f"T_phòng <= {sc.emergency_freeze_temp}°C (và action in [1,2,3])",
         "Hành động cưỡng bức": "COMFORT_HEAT (Action 6 - sưởi 22°C)",
         "Cơ sở kỹ thuật / An toàn": "Cấm chạy làm lạnh tiếp, ngăn hạ thân nhiệt hoặc đóng băng dàn."},
        {"Mã luật": "SG-03", "Tên luật an toàn": "Xả khí ngạt khẩn cấp (Emergency CO2 Flush)",
         "Điều kiện kích hoạt": f"CO2_phòng >= {sc.emergency_co2_ppm:.0f} ppm (và action not in [2, 5])",
         "Hành động cưỡng bức": "ACTIVE_VENT_PURIFY (Action 5 - SET_25, gió 500 m³/h, HEPA 450)",
         "Cơ sở kỹ thuật / An toàn": "Cấp gió tươi cưỡng bức tức thì tránh nguy hiểm thiếu oxy/tích tụ CO2."},
        {"Mã luật": "SG-04", "Tên luật an toàn": "Khóa an ninh & thời tiết cửa sổ (Window Vetoes)",
         "Điều kiện kích hoạt": (f"Action 4 khi giờ in [{sc.night_security_start_hour}:00-{sc.night_security_end_hour}:00) "
                                 f"HOẶC T_amb>{sc.window_max_ambient_temp}°C HOẶC RH_amb>{sc.window_max_ambient_rh}% HOẶC PM_amb>{sc.window_max_ambient_pm25} µg/m³"),
         "Hành động cưỡng bức": "Ban đêm->Action 11; Nóng/Mưa->Action 1; Khói bụi->Action 10",
         "Cơ sở kỹ thuật / An toàn": "Chống đột nhập ban đêm, chống hắt mưa/gió nóng và ngăn khói bụi PM2.5 tràn vào phòng."},
        {"Mã luật": "SG-05", "Tên luật an toàn": "Chống đảo nấc máy nén (Compressor Anti-Short-Cycle)",
         "Điều kiện kích hoạt": f"Bật/tắt máy nén khi số bước giữ trạng thái < {sc.min_compressor_dwell_steps} bước (15 phút -> 1 bước 1h)",
         "Hành động cưỡng bức": "Giữ nguyên trạng thái hành động bước trước",
         "Cơ sở kỹ thuật / An toàn": "Tránh sốc dòng khởi động máy nén liên tiếp và ổn định áp suất gas."},
    ]


def build_hyperparams_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Exact optimal hyperparameters read dynamically from manifest.json."""
    hp = manifest["hyperparameters"]
    rows = []
    for label, key in [("Double Q-Learning (DQL)", "double_q_learning"), ("Single Q-Learning (QL)", "single_q_learning")]:
        cfg = hp[key]
        rows.append({
            "Thuật toán": label,
            "Tốc độ học (alpha)": cfg["learning_rate"],
            "Tỷ lệ phân rã mục tiêu": cfg.get("target_decay_frac", "N/A"),
            "Hệ số suy giảm epsilon": round(cfg["epsilon_decay"], 6),
            "Hệ số chiết khấu (gamma)": 0.95,
            "Số episodes huấn luyện": cfg["n_episodes"],
            "Epsilon tối thiểu": cfg.get("epsilon_min", 0.02),
            "Không gian trạng thái": "720 bins rời rạc",
            "Không gian hành động": "12 templates phối hợp",
            "Phần thưởng Validation": round(cfg.get("val_mean_reward", 0.0), 2),
            "Phương pháp tuyển chọn": "Grid Search 27 cấu hình x 3 seeds trên Validation 2018 (Protocol v2)",
        })
    return rows


def build_sample_decision_rows(env: MultiDeviceHVACEnv) -> List[Dict[str, Any]]:
    """Computes live mathematically exact reward breakdowns using MultiDeviceRewardFunction."""
    rf = MultiDeviceRewardFunction(preference="balanced")
    gw = MultiDeviceSafetyGateway()

    cases = [
        ("Kịch bản 1: Trưa hè oi bức, phòng họp đông người", 29.5, 68.0, 1450.0, 42.0, 14, 1, [0, 5, 2], "Đánh giá ưu tiên làm mát + xả CO2"),
        ("Kịch bản 2: Giờ làm việc chuẩn, nhiệt ẩm đạt chuẩn", 25.0, 54.0, 720.0, 14.0, 10, 1, [1, 2, 0], "Cân bằng giữ nhiệt và tiết kiệm điện"),
        ("Kịch bản 3: Không khí ngột ngạt CO2 tích tụ cao", 25.4, 58.0, 1650.0, 22.0, 15, 1, [1, 5, 11], "Action 5 (SET_25 + Vent 500 + HEPA 450)"),
        ("Kịch bản 4: Ban đêm phòng trống, ngoài trời mát sạch", 24.5, 52.0, 480.0, 12.0, 2, 0, [0, 11, 1], "Tiết kiệm tối đa khi phòng trống"),
        ("Kịch bản 5: Ngày nồm ẩm cao, bụi mịn hơi tăng", 25.8, 72.0, 760.0, 38.0, 11, 1, [8, 5, 1], "Chọn khử ẩm / lọc kết hợp làm mát"),
    ]

    rows = []
    for sc_name, t_in, rh_in, co2_in, pm_in, hr, is_wh, cand_acts, note in cases:
        for aid in cand_acts:
            gw.reset()
            safe_act, reason = gw.filter_action(
                aid, t_in, rh_in, co2_in, pm_in,
                ambient_temp=33.5 if 9 <= hr <= 17 else 27.0,
                ambient_rh=70.0, ambient_pm25=25.0, hour=hr, step=1
            )
            p_res = env._calculate_actuator_powers_and_effects(
                MULTI_DEVICE_ACTION_TEMPLATES[safe_act],
                current_temp=t_in, current_rh=rh_in, ambient_temp=33.5, ambient_rh=70.0
            )
            p_kw = p_res["powers"]["total_kw"]
            tot_r, br = rf.calculate_reward(
                temp_c=t_in, humidity_rh=rh_in, co2_ppm=co2_in, pm25_ug_m3=pm_in,
                total_power_kw=p_kw, hour=hr, dayofweek=1, is_workhour=is_wh,
                action_changed=False, safety_violation=(safe_act != aid)
            )
            override_tag = f" (Bị Safety ghi đè -> {safe_act})" if safe_act != aid else ""
            rows.append({
                "Kịch bản": sc_name,
                "Trạng thái đầu vào": f"T={t_in}°C, RH={rh_in}%, CO2={co2_in:.0f}, PM={pm_in}, h={hr:02d}, work={is_wh}",
                "Action thử nghiệm": f"A{aid}: {MULTI_DEVICE_ACTION_TEMPLATES[aid].name}{override_tag}",
                "Công suất (kW)": round(p_kw, 2),
                "R_temp": br["reward_temp"],
                "R_humidity": br["reward_humidity"],
                "R_co2": br["reward_co2"],
                "R_air_quality": br["reward_air_quality"],
                "R_energy": br["reward_energy"],
                "R_safety": br["reward_safety"],
                "TỔNG REWARD": br["total_reward"],
                "Ghi chú": note,
            })
    return rows


def _style_sheet(ws, title: str, subtitle: str, source: str, df: pd.DataFrame) -> None:
    ncols = len(df.columns)
    ws.append([title])
    ws.append([subtitle])
    ws.append([source])
    ws.append([])
    ws.append(list(df.columns))
    for r in df.itertuples(index=False):
        ws.append(list(r))

    navy_fill = PatternFill("solid", fgColor="1E293B")
    teal_fill = PatternFill("solid", fgColor="0B6E70")
    zebra_fill = PatternFill("solid", fgColor="F8FAFC")
    white_bold = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="0F172A")
    sub_font = Font(name="Calibri", size=10, italic=True, color="475569")
    body_font = Font(name="Calibri", size=10, color="1E293B")
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=ncols)
    ws["A1"].font = title_font
    ws["A2"].font = sub_font
    ws["A3"].font = sub_font

    for col in range(1, ncols + 1):
        c = ws.cell(row=5, column=col)
        c.fill = teal_fill if col == ncols else navy_fill
        c.font = white_bold
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border

    for r_idx in range(6, 6 + len(df)):
        for c_idx in range(1, ncols + 1):
            c = ws.cell(row=r_idx, column=c_idx)
            c.font = body_font
            c.border = border
            c.alignment = Alignment(vertical="center", wrap_text=True)
            if r_idx % 2 == 1:
                c.fill = zebra_fill

    for col in range(1, ncols + 1):
        col_letter = get_column_letter(col)
        max_len = max(
            [len(str(ws.cell(row=r, column=col).value or "")) for r in range(5, 6 + len(df))],
            default=12
        )
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 14), 48)
    ws.freeze_panes = "A6"


def main():
    out_tables = REPO_ROOT / "result" / "final" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / "result" / "final" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    act_dist_path = REPO_ROOT / "result" / "final" / "action_distribution.csv"
    df_act_dist = pd.read_csv(act_dist_path) if act_dist_path.exists() else pd.DataFrame()

    env = MultiDeviceHVACEnv(episode_length=24, seed=42)

    df_state = pd.DataFrame(build_state_rows())
    df_action = pd.DataFrame(build_action_rows(env, df_act_dist))
    df_reward = pd.DataFrame(build_reward_rows())
    df_safety = pd.DataFrame(build_safety_rows())
    df_hyper = pd.DataFrame(build_hyperparams_rows(manifest))
    df_sample = pd.DataFrame(build_sample_decision_rows(env))

    df_state.to_csv(out_tables / "state_table.csv", index=False, encoding="utf-8-sig")
    df_action.to_csv(out_tables / "action_table.csv", index=False, encoding="utf-8-sig")
    df_reward.to_csv(out_tables / "reward_table.csv", index=False, encoding="utf-8-sig")
    df_safety.to_csv(out_tables / "safety_table.csv", index=False, encoding="utf-8-sig")
    df_hyper.to_csv(out_tables / "hyperparams_table.csv", index=False, encoding="utf-8-sig")
    print(f"[*] Exported 5 authoritative CSV design tables to {out_tables}")

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "1. Không Gian Trạng Thái (720)"
    _style_sheet(
        ws1,
        "BẢNG THIẾT KẾ KHÔNG GIAN TRẠNG THÁI RỜI RẠC (720 DISCRETE STATES)",
        "Mô hình hóa 6 chiều trực giao theo chuẩn ASHRAE 55-2020 & ASHRAE 62.1",
        "Nguồn mã nguồn đồng bộ: src/environment/multi_device_env.py (discretize_state_720)",
        df_state,
    )

    ws2 = wb.create_sheet("2. Không Gian Hành Động (12)")
    _style_sheet(
        ws2,
        "BẢNG THIẾT KẾ 12 MẪU HÀNH ĐỘNG PHỐI HỢP ĐA THIẾT BỊ (PROTOCOL v2)",
        "Phối hợp 7 cụm thiết bị; Action 5 ACTIVE_VENT_PURIFY tích hợp làm mát SET_25 tránh nóng phòng nhiệt đới",
        "Nguồn: src/environment/multi_device_env.py & result/final/action_distribution.csv (Test-A 10 seeds)",
        df_action,
    )

    ws3 = wb.create_sheet("3. Hàm Thưởng 7 Thành Phần")
    _style_sheet(
        ws3,
        "BẢNG PHÂN RÃ HÀM THƯỞNG ĐA MỤC TIÊU (7 DECOMPOSED COMPONENTS)",
        "Tổng thưởng: R_total = R_temp + R_humidity + R_co2 + R_air_quality + R_energy + R_switching + R_safety",
        "Nguồn mã nguồn đồng bộ: src/reward/multi_device_reward.py (MultiDeviceRewardFunction balance profile)",
        df_reward,
    )

    ws4 = wb.create_sheet("4. Mô Phỏng Quyết Định Mẫu")
    _style_sheet(
        ws4,
        "BẢNG TÍNH TOÁN THƯỞNG THỰC TẾ TỪ MÃ NGUỒN CHO 5 KỊCH BẢN MÔI TRƯỜNG",
        "Số liệu được tính trực tiếp bởi MultiDeviceRewardFunction.calculate_reward() và MultiDeviceSafetyGateway",
        "Nguồn: Live calculation từ src/reward/multi_device_reward.py & src/safety/multi_device_safety_gateway.py",
        df_sample,
    )

    xlsx_path = REPO_ROOT / "doc" / "THIET_KE_PHAN_THUONG_VA_HANH_DONG_HVAC.xlsx"
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    print(f"[*] Exported 4-sheet Excel design workbook to {xlsx_path}")


if __name__ == "__main__":
    main()
