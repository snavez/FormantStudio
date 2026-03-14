"""Generate a FormantStudio app icon (.ico) with multiple sizes."""
import math
from PIL import Image, ImageDraw, ImageFont

SIZES = [256, 128, 64, 48, 32, 16]


def draw_icon(size: int) -> Image.Image:
    """Draw a single icon at the given size."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # --- Rounded-rect background: dark blue-grey gradient feel ---
    pad = max(1, size // 32)
    radius = max(2, size // 6)
    # Background
    draw.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=radius,
        fill=(30, 35, 55, 255),
    )
    # Subtle lighter border
    draw.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=radius,
        outline=(80, 120, 180, 120),
        width=max(1, size // 64),
    )

    # --- Spectrogram bars (vertical bars with varying heights) ---
    inner_l = pad + max(2, size // 10)
    inner_r = size - pad - max(2, size // 10)
    inner_t = pad + max(2, size // 6)
    inner_b = size - pad - max(2, size // 4)
    bar_region_w = inner_r - inner_l
    bar_region_h = inner_b - inner_t

    n_bars = max(8, size // 8)
    bar_w = max(1, bar_region_w // n_bars)

    for i in range(n_bars):
        x = inner_l + i * (bar_region_w / n_bars)
        # Create a waveform-like height pattern
        t = i / max(1, n_bars - 1)
        h_frac = 0.3 + 0.5 * math.sin(t * math.pi * 2.2) ** 2 + 0.2 * math.sin(t * math.pi * 4.5) ** 2
        h = int(bar_region_h * h_frac)
        y_top = inner_b - h

        # Gradient color: teal at bottom to warm orange at top
        for row in range(h):
            frac = row / max(1, h - 1)
            r = int(50 + 200 * frac)
            g = int(180 - 60 * frac)
            b = int(200 - 160 * frac)
            alpha = int(180 + 75 * (1 - frac))
            y = inner_b - row
            x0 = int(x)
            x1 = int(x + bar_w - 1)
            draw.line([(x0, y), (x1, y)], fill=(r, g, b, alpha))

    # --- Formant curves (2-3 smooth curves overlaid) ---
    curves = [
        (0.65, 0.08, 1.8, (255, 130, 50, 220)),   # F1 - orange
        (0.40, 0.12, 2.5, (100, 200, 255, 200)),   # F2 - light blue
        (0.22, 0.06, 3.2, (180, 255, 160, 180)),   # F3 - green
    ]
    for base_y_frac, amp_frac, freq, color in curves:
        pts = []
        for px in range(inner_l, inner_r + 1):
            t = (px - inner_l) / max(1, inner_r - inner_l)
            y_frac = base_y_frac + amp_frac * math.sin(t * math.pi * freq)
            y = inner_t + int(bar_region_h * y_frac)
            pts.append((px, y))
        if len(pts) > 1:
            lw = max(1, size // 64)
            draw.line(pts, fill=color, width=lw)

    # --- "FS" text at bottom ---
    text_size = max(6, size // 5)
    try:
        font = ImageFont.truetype("segoeuib.ttf", text_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arialbd.ttf", text_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    text = "FS"
    text_y = size - pad - max(2, size // 5) - max(1, size // 32)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    text_x = (size - tw) // 2
    # Shadow
    draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0, 140), font=font)
    # Main text
    draw.text((text_x, text_y), text, fill=(220, 230, 255, 240), font=font)

    return img


def main():
    images = [draw_icon(s) for s in SIZES]
    # Save as .ico with all sizes
    images[0].save(
        "formant_studio.ico",
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=images[1:],
    )
    # Also save a large PNG for reference
    images[0].save("formant_studio_256.png")
    print("Generated formant_studio.ico and formant_studio_256.png")


if __name__ == "__main__":
    main()
