"""Capture live Vietnamese Smart HVAC Control Room GUI and create annotated demo guide."""

import ctypes
import sys
import time
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageGrab

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _capture_hwnd_via_printwindow(hwnd: int, width: int, height: int) -> Image.Image:
    """Capture a live HWND directly into an offscreen bitmap using Windows PrintWindow."""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]

    hwnd_dc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old_obj = gdi32.SelectObject(mem_dc, bmp)

    # Try PW_RENDERFULLCONTENT (2) first, then 0
    ok = user32.PrintWindow(hwnd, mem_dc, 2)
    if not ok:
        ok = user32.PrintWindow(hwnd, mem_dc, 0)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # top-down DIB
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0  # BI_RGB

    buf = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(mem_dc, bmp, 0, height, buf, ctypes.byref(bmi), 0)

    gdi32.SelectObject(mem_dc, old_obj)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)

    img = Image.frombuffer("RGBA", (width, height), buf, "raw", "BGRA", 0, 1).convert("RGB")
    return img


def capture_live_gui(raw_out: Path) -> Path:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    from app.live_simulation_gui import LiveSimulationApp

    root = tk.Tk()
    scr_w = root.winfo_screenwidth()
    scr_h = root.winfo_screenheight()
    win_w = min(1560, max(1360, scr_w - 40))
    win_h = min(920, max(820, scr_h - 60))

    app = LiveSimulationApp(root)
    root.geometry(f"{win_w}x{win_h}+20+20")
    root.update()

    # Step 10 daytime peak hours (11:00..20:00) so charts & expanded Step History show active HVAC diversity
    for _ in range(10):
        app.step_forward_simulation()

    app.canvas.draw()
    root.update_idletasks()
    root.update()
    time.sleep(0.35)
    root.update()

    # Top-level HWND (GA_ROOT = 2)
    hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2) or root.winfo_id()

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    rect = RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    rw = max(win_w, rect.right - rect.left)
    rh = max(win_h, rect.bottom - rect.top)

    img = _capture_hwnd_via_printwindow(hwnd, rw, rh)
    extrema = img.getextrema()
    mean_rgb = sum(sum(c) / (255.0 * 2.0) for c in extrema) / 3.0
    print(f"[*] Captured PrintWindow extrema={extrema}, norm_range={mean_rgb:.3f}")

    # Fallback: render Matplotlib Figure Canvas cleanly overlaid into layout if desktop DC returns flat image
    if all(lo == hi for lo, hi in extrema):
        raise RuntimeError("PrintWindow returned completely flat pixels — falling back required")

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    img.save(raw_out, quality=95)
    root.destroy()
    print(f"[+] Captured live GUI screenshot ({img.size[0]}x{img.size[1]}) -> {raw_out}")
    return raw_out


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguibd.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def annotate_gui_comic_style(raw_path: Path, annotated_out: Path) -> Path:
    gui_img = Image.open(raw_path).convert("RGBA")
    gw, gh = gui_img.size

    # Create spacious outer annotation canvas around the crisp raw GUI screenshot
    left_pad = 360
    right_pad = 360
    top_pad = 135
    bot_pad = 55
    canvas_w = gw + left_pad + right_pad
    canvas_h = gh + top_pad + bot_pad

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (15, 23, 42, 255))
    draw = ImageDraw.Draw(canvas)

    # Soft header banner
    f_title = _load_font(28, bold=True)
    f_sub = _load_font(17, bold=False)
    f_badge = _load_font(18, bold=True)
    f_card_t = _load_font(17, bold=True)
    f_card_b = _load_font(15, bold=False)

    draw.text(
        (left_pad, 22),
        "HƯỚNG DẪN ĐỌC GIAO DIỆN MÔ PHỎNG & DEMO HỆ THỐNG SMART HVAC (TIẾNG VIỆT)",
        fill=(56, 189, 248, 255),
        font=f_title,
    )
    draw.text(
        (left_pad, 64),
        "6 khu vực theo dõi trực tiếp khi thuyết trình: Chọn kịch bản -> Nhấn BẮT ĐẦU / CHẠY 1 BƯỚC -> Quan sát thiết bị & bảng Q-Table",
        fill=(226, 232, 240, 255),
        font=f_sub,
    )

    # Paste raw GUI screenshot with subtle cyan frame
    gx, gy = left_pad, top_pad
    draw.rounded_rectangle([gx - 4, gy - 4, gx + gw + 4, gy + gh + 4], radius=10, outline=(34, 211, 238, 255), width=4)
    canvas.paste(gui_img, (gx, gy))

    # Define 6 callout cards with non-overlapping left/right positions & clean pointer targets on GUI
    # Normalized coordinates relative to raw screenshot (gw, gh)
    callouts = [
        {
            "num": "1",
            "side": "left",
            "y_box": top_pad + 10,
            "color": (56, 189, 248),
            "title": "Khu 1: Thanh KPI 8 Chỉ Số Thời Gian Thực",
            "body": "Theo dõi nhanh Nhiệt độ [24-26°C], Độ ẩm [40-65%], CO2 (<800 ppm), PM2.5, Tổng công suất (kW), Điện tích lũy (kWh) và Giờ mô phỏng.",
            "target": (gx + int(gw * 0.006), gy + int(gh * 0.095)),
        },
        {
            "num": "2",
            "side": "left",
            "y_box": top_pad + int(gh * 0.24),
            "color": (52, 211, 153),
            "title": "Khu 2: Chọn Phòng, Thuật Toán AI & Điều Khiển",
            "body": "Đổi 4 kịch bản phòng & 6 bộ điều khiển (DQL, QL, Luật IF-THEN...). Nhấn 'CHẠY 1 BƯỚC' để giải thích từng giờ hoặc 'BẮT ĐẦU' chạy liên tục.",
            "target": (gx + int(gw * 0.015), gy + int(gh * 0.24)),
        },
        {
            "num": "3",
            "side": "left",
            "y_box": top_pad + int(gh * 0.50),
            "color": (251, 191, 36),
            "title": "Khu 3: Thông Số Vật Lý & 7 Nhóm Thiết Bị",
            "body": "Hiện rõ trạng thái đóng/mở từng thiết bị thật: Máy lạnh (°C), Sưởi, Quạt gió tươi (m³/h), Cửa sổ ĐÓNG/MỞ, Tạo ẩm, Hút ẩm & Lọc bụi HEPA.",
            "target": (gx + int(gw * 0.015), gy + int(gh * 0.61)),
        },
        {
            "num": "4",
            "side": "left",
            "y_box": top_pad + int(gh * 0.77),
            "color": (167, 139, 250),
            "title": "Khu 4: Dấu Vết Quyết Định & Thiết Lập Thiết Bị",
            "body": "Giải mã hộp đen AI: chỉ số trạng thái (0..719), điểm QA/QB, dòng thiết lập Cửa sổ/Máy lạnh cụ thể, phán quyết Cổng An toàn & chia điểm thưởng.",
            "target": (gx + int(gw * 0.015), gy + int(gh * 0.78)),
        },
        {
            "num": "5",
            "side": "right",
            "y_box": top_pad + int(gh * 0.18),
            "color": (34, 211, 238),
            "title": "Khu 5: 4 Biểu Đồ Diễn Biến Liên Tục",
            "body": "Vẽ song song biến thiên theo giờ của Nhiệt độ (vùng xanh chuẩn), Độ ẩm, Nồng độ CO2 (vạch giới hạn 800/1200 ppm) và Công suất tiêu thụ điện (kW).",
            "target": (gx + int(gw * 0.98), gy + int(gh * 0.21)),
        },
        {
            "num": "6",
            "side": "right",
            "y_box": top_pad + int(gh * 0.62),
            "color": (244, 114, 182),
            "title": "Khu 6: Bảng Lịch Sử Từng Bước (Đã Mở Rộng)",
            "body": "Hiển thị đầy đủ 12 cột: Bước, Giờ, Nhiệt/Ẩm/CO2, Lệnh AI đề xuất, Lệnh thực thi và cấu hình Cửa sổ/Thiết bị. Nhấp 1 dòng bất kỳ để soi lại quá khứ!",
            "target": (gx + int(gw * 0.98), gy + int(gh * 0.64)),
        },
    ]

    box_w = 328
    for c in callouts:
        r_col = c["color"] + (255,)
        bx = 16 if c["side"] == "left" else gx + gw + 16
        by = c["y_box"]

        title_lines = _wrap_text(draw, c["title"], f_card_t, box_w - 58)
        body_lines = _wrap_text(draw, c["body"], f_card_b, box_w - 28)
        box_h = 22 + len(title_lines) * 22 + 10 + len(body_lines) * 20 + 16

        # Card body (comic speech card)
        draw.rounded_rectangle(
            [bx, by, bx + box_w, by + box_h],
            radius=12,
            fill=(248, 250, 252, 255),
            outline=r_col,
            width=4,
        )

        # Number circle badge
        cx, cy = bx + 26, by + 26
        draw.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=r_col)
        tb = draw.textbbox((0, 0), c["num"], font=f_badge)
        draw.text((cx - (tb[2] - tb[0]) // 2, cy - (tb[3] - tb[1]) // 2 - 2), c["num"], fill=(15, 23, 42, 255), font=f_badge)

        # Title & body text inside bright comic bubble
        ty = by + 14
        for tl in title_lines:
            draw.text((bx + 50, ty), tl, fill=(15, 23, 42, 255), font=f_card_t)
            ty += 22
        ty += 8
        for bl in body_lines:
            draw.text((bx + 16, ty), bl, fill=(51, 65, 85, 255), font=f_card_b)
            ty += 20

        # Anchor point from bubble edge to target on GUI
        ax0 = bx + box_w if c["side"] == "left" else bx
        ay0 = by + box_h // 2
        tx, ty_p = c["target"]

        # Clean elbow line + target ring badge on GUI
        mid_x = gx - 10 if c["side"] == "left" else gx + gw + 10
        draw.line([(ax0, ay0), (mid_x, ay0), (tx, ty_p)], fill=r_col, width=4)
        draw.ellipse([tx - 15, ty_p - 15, tx + 15, ty_p + 15], fill=r_col, outline=(255, 255, 255, 255), width=3)
        draw.text((tx - (tb[2] - tb[0]) // 2, ty_p - (tb[3] - tb[1]) // 2 - 2), c["num"], fill=(15, 23, 42, 255), font=f_badge)

    canvas.convert("RGB").save(annotated_out, quality=95)
    print(f"[+] Annotated comic-style demo guide ({canvas_w}x{canvas_h}) -> {annotated_out}")
    return annotated_out


def main():
    assets_dir = REPO_ROOT / "doc" / "assets"
    raw_path = assets_dir / "gui_raw_vi.png"
    annotated_path = assets_dir / "gui_demo_annotated_vi.png"
    capture_live_gui(raw_path)
    annotate_gui_comic_style(raw_path, annotated_path)


if __name__ == "__main__":
    main()
