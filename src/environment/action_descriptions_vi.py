"""Canonical Vietnamese labels and exact equipment-window mappings for Smart HVAC.

Ensures 100% consistency across the GUI, Word report, Defense Handbook,
charts, Excel workbook, and PowerPoint presentation slides.
"""

from typing import Dict

ACTION_META_VI: Dict[int, Dict[str, str]] = {
    0: {
        "code": "ALL_OFF_STANDBY",
        "short_vi": "A0: Tắt toàn bộ (Chờ tiết kiệm)",
        "name_only_vi": "Tắt toàn bộ (Chờ tiết kiệm)",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: Tắt | Sưởi: Tắt | Quạt: 0 m³/h | Ẩm: Tắt | HEPA: Tắt",
        "summary_vi": "Ngắt toàn bộ 7 nhóm thiết bị và đóng kín cửa sổ để đưa công suất về 0 kW khi phòng trống hoặc đã cân bằng.",
    },
    1: {
        "code": "COMFORT_COOL",
        "short_vi": "A1: Làm mát chuẩn 25°C",
        "name_only_vi": "Làm mát chuẩn 25°C + Lọc nhẹ",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 25°C | Sưởi: Tắt | Quạt: 150 m³/h | HEPA: 120 m³/h",
        "summary_vi": "Bật máy lạnh mục tiêu 25°C, cấp gió tươi nhẹ 150 m³/h, lọc bụi HEPA 120 m³/h và đóng cửa giữ nhiệt.",
    },
    2: {
        "code": "MAX_COOL",
        "short_vi": "A2: Làm lạnh mạnh 23°C + Gió vừa",
        "name_only_vi": "Làm lạnh mạnh 23°C + Thông gió vừa",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 23°C | Sưởi: Tắt | Quạt: 300 m³/h | HEPA: 250 m³/h",
        "summary_vi": "Hạ máy lạnh xuống 23°C, tăng quạt cấp gió 300 m³/h, lọc HEPA 250 m³/h và đóng cửa để hạ nhiệt nhanh giờ nóng.",
    },
    3: {
        "code": "ECONOMY_COOL",
        "short_vi": "A3: Làm mát tiết kiệm 27°C",
        "name_only_vi": "Làm mát tiết kiệm 27°C",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 27°C | Sưởi: Tắt | Quạt: 150 m³/h | HEPA: 100 m³/h",
        "summary_vi": "Đặt máy lạnh 27°C để giảm tải nén giờ cao điểm, giữ gió tươi 150 m³/h và lọc nhẹ 100 m³/h.",
    },
    4: {
        "code": "NATURAL_VENTILATION",
        "short_vi": "A4: Mở cửa sổ thông gió tự nhiên",
        "name_only_vi": "Mở cửa sổ thông gió tự nhiên",
        "window_vi": "MỞ",
        "devices_vi": "Cửa sổ: MỞ | AC: Tắt | Sưởi: Tắt | Quạt: 0 m³/h | HEPA: 80 m³/h",
        "summary_vi": "Tắt AC/sưởi, mở hoàn toàn cửa sổ tận dụng gió trời miễn phí khi thời tiết dịu mát, chạy lọc HEPA nền 80 m³/h.",
    },
    5: {
        "code": "ACTIVE_VENT_PURIFY",
        "short_vi": "A5: Xả CO2 + Lọc tối đa + AC 25°C",
        "name_only_vi": "Xả CO2 + Lọc HEPA tối đa + AC 25°C",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 25°C | Quạt: 500 m³/h | HEPA: 450 m³/h",
        "summary_vi": "Đẩy quạt gió tươi lên 500 m³/h và lọc HEPA 450 m³/h để xả nhanh CO2/PM2.5, kèm AC 25°C ổn định nhiệt vùng nhiệt đới.",
    },
    6: {
        "code": "COMFORT_HEAT",
        "short_vi": "A6: Sưởi ấm chuẩn 22°C + Bù ẩm nhẹ",
        "name_only_vi": "Sưởi ấm chuẩn 22°C + Bù ẩm nhẹ",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: Tắt | Sưởi: 22°C | Quạt: 150 m³/h | Phun ẩm: 0.25 kg/h | HEPA: 100 m³/h",
        "summary_vi": "Bật sưởi 22°C kèm phun ẩm nhẹ 0.25 kg/h, gió tươi 150 m³/h, lọc 100 m³/h và đóng cửa giữ ấm.",
    },
    7: {
        "code": "MAX_HEAT",
        "short_vi": "A7: Sưởi ấm mạnh 24°C + Bù ẩm vừa",
        "name_only_vi": "Sưởi ấm mạnh 24°C + Bù ẩm vừa",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: Tắt | Sưởi: 24°C | Quạt: 150 m³/h | Phun ẩm: 0.45 kg/h | HEPA: 100 m³/h",
        "summary_vi": "Bật sưởi công suất cao 24°C khi trời lạnh sâu, phun ẩm 0.45 kg/h, quạt 150 m³/h và đóng kín cửa.",
    },
    8: {
        "code": "DEHUMIDIFY_ACTIVE",
        "short_vi": "A8: Hút ẩm tích cực + AC 26°C",
        "name_only_vi": "Hút ẩm tích cực + Làm mát 26°C",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 26°C | Hút ẩm: 1.2 kg/h | Quạt: 150 m³/h | HEPA: 100 m³/h",
        "summary_vi": "Bật máy hút ẩm 1.2 kg/h khi ẩm cao kết hợp AC 26°C, quạt 150 m³/h và đóng cửa ngăn ẩm từ ngoài.",
    },
    9: {
        "code": "HUMIDIFY_COMFORT",
        "short_vi": "A9: Tạo ẩm cao + AC 25°C",
        "name_only_vi": "Tạo ẩm cao + Làm mát 25°C",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 25°C | Phun ẩm: 0.6 kg/h | Quạt: 150 m³/h | HEPA: 100 m³/h",
        "summary_vi": "Bật phun ẩm 0.6 kg/h khi không khí khô (<40% RH) kèm AC 25°C, quạt 150 m³/h và đóng cửa.",
    },
    10: {
        "code": "ECO_AIR_PURIFY",
        "short_vi": "A10: Chỉ lọc bụi HEPA tối đa (Tắt AC)",
        "name_only_vi": "Chỉ lọc bụi HEPA tối đa (Tắt AC)",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: Tắt | Sưởi: Tắt | Quạt: 100 m³/h | HEPA: 450 m³/h",
        "summary_vi": "Tắt điều hòa nhiệt độ để tiết kiệm điện, bật tối đa lọc bụi HEPA 450 m³/h và gió tươi 100 m³/h khi nhiệt độ đã ổn.",
    },
    11: {
        "code": "NIGHT_SETBACK",
        "short_vi": "A11: Ban đêm 26.5°C + Yên tĩnh",
        "name_only_vi": "Chế độ ban đêm 26.5°C + Yên tĩnh",
        "window_vi": "Đóng",
        "devices_vi": "Cửa: Đóng | AC: 26.5°C | Quạt: 100 m³/h | HEPA: 80 m³/h",
        "summary_vi": "Duy trì AC 26.5°C tiết kiệm ban đêm, giảm quạt xuống 100 m³/h và lọc 80 m³/h chạy êm, đóng cửa sổ an toàn.",
    },
}

CONTROLLER_META_VI: Dict[str, Dict[str, str]] = {
    "Double Q-Learning": {
        "display_vi": "Double Q-Learning (Học tăng cường kép)",
        "short_vi": "Double Q-Learning",
        "meaning_vi": "Dùng 2 bảng QA & QB cập nhật chéo nhau nhằm triệt tiêu thiên lệch cực đại (Maximization Bias), hội tụ ổn định với 720 trạng thái.",
    },
    "Single Q-Learning": {
        "display_vi": "Q-Learning đơn (Học tăng cường đơn)",
        "short_vi": "Single Q-Learning",
        "meaning_vi": "Dùng 1 bảng Q duy nhất vừa chọn hành động vừa định giá cực đại max Q(s',a'), dễ dao động hơn giữa các seed.",
    },
    "Fixed Setpoint (25°C Cool)": {
        "display_vi": "Đặt cố định 25°C (Làm mát & thông gió nền)",
        "short_vi": "Đặt cố định 25°C",
        "meaning_vi": "Cố định chạy Hành động A1 (AC 25°C, quạt 150 m³/h) 24/24h không đổi theo giờ vắng hay nồng độ CO2 cao.",
    },
    "Rule-Based Smart HVAC": {
        "display_vi": "Bộ luật chuyên gia IF-THEN (Theo ngưỡng)",
        "short_vi": "Bộ luật IF-THEN",
        "meaning_vi": "Tập luật ngưỡng cố định do kỹ sư lập trình sẵn (ví dụ >26.5°C -> A2, <20.5°C -> A6, ban đêm -> A11), chưa tối ưu đồng thời CO2/giá điện.",
    },
    "Conventional Thermostat": {
        "display_vi": "Rơ-le nhiệt truyền thống (Bật/Tắt đơn thuần)",
        "short_vi": "Rơ-le truyền thống",
        "meaning_vi": "Chỉ đo nhiệt độ để Bật/Tắt máy lạnh hoặc sưởi (không có quạt gió tươi chủ động và không điều khiển ẩm/HEPA).",
    },
    "Random Policy": {
        "display_vi": "Chính sách ngẫu nhiên (Chọn bừa 1/12 hành động)",
        "short_vi": "Ngẫu nhiên (Baseline)",
        "meaning_vi": "Mốc sàn kiểm chứng (lower bound): chọn ngẫu nhiên 1 trong 12 hành động mỗi giờ để chứng minh Agent thực sự học được quy luật tối ưu.",
    },
}


def format_action_short_vi(action_id: int) -> str:
    meta = ACTION_META_VI.get(int(action_id))
    return meta["short_vi"] if meta else f"A{action_id}"


def format_action_devices_vi(action_id: int) -> str:
    meta = ACTION_META_VI.get(int(action_id))
    return meta["devices_vi"] if meta else ""
