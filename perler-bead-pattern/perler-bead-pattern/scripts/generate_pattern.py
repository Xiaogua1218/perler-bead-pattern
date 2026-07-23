#!/usr/bin/env python3
"""
QQ拼豆施工图生成器

将任意图片转换为专业拼豆施工图，基于 MARD 221 标准色卡。
主体强制像素化，禁止柔边/渐变/照片质感。
支持智能抠底检测。

Usage:
  python generate_pattern.py <image> -o <output.png> [options]

Options:
  --size WxH        网格尺寸，默认auto
  --cellsize N      每格像素大小，默认24
  --max-colors N    最大颜色数：0=自动，221=全色卡
  --bg-remove       强制去除背景
  --bg-threshold N  背景检测阈值，默认30
  --no-symbols      不标注色号
  --no-legend       不显示色卡图例

Dependencies: Pillow (pip install Pillow)
"""

import argparse
import json
import math
import os
import sys
import textwrap
from collections import Counter

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


# ──────────────────────────────────────────────────────────────
# Color conversion utilities
# ──────────────────────────────────────────────────────────────

def srgb_to_lab(r, g, b):
    """sRGB (0-255) → CIELAB D65."""
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    rl, gl, bl = linearize(r), linearize(g), linearize(b)
    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    xn, yn, zn = 95.047, 100.000, 108.883
    d3 = (6.0 / 29.0) ** 3
    def f(t):
        return t ** (1.0 / 3.0) if t > d3 else t / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0
    L = 116.0 * f(y / yn) - 16.0
    a = 500.0 * (f(x / xn) - f(y / yn))
    b = 200.0 * (f(y / yn) - f(z / zn))
    return (L, a, b)


def delta_e(lab1, lab2):
    """CIE76 ΔE — Euclidean distance in LAB space."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def nearest_color(rgb, palette):
    """Find the nearest palette color by ΔE."""
    lab_in = srgb_to_lab(*rgb)
    best, best_dist = palette[0], float('inf')
    for entry in palette:
        d = delta_e(lab_in, entry['lab'])
        if d < best_dist:
            best_dist = d
            best = entry
    return best


def analyze_image_colors(img, sample_step=4):
    """Analyze image colors and return (dominant_labs, suggested_max_colors)."""
    w, h = img.size
    pixels = []
    for y in range(0, h, sample_step):
        for x in range(0, w, sample_step):
            pixels.append(img.getpixel((x, y))[:3])

    if not pixels:
        return [], 221

    STEP = 32
    bucket_map = {}
    for r, g, b in pixels:
        key = (r // STEP, g // STEP, b // STEP)
        bucket_map.setdefault(key, []).append((r, g, b))

    dominant_rgbs = []
    for group in bucket_map.values():
        avg_r = sum(c[0] for c in group) // len(group)
        avg_g = sum(c[1] for c in group) // len(group)
        avg_b = sum(c[2] for c in group) // len(group)
        dominant_rgbs.append((avg_r, avg_g, avg_b))

    n_distinct = len(dominant_rgbs)
    if n_distinct <= 6:
        suggested = max(n_distinct, 8)
    elif n_distinct <= 15:
        suggested = min(n_distinct * 2, 48)
    elif n_distinct <= 40:
        suggested = min(n_distinct * 3, 96)
    else:
        suggested = 221

    dominant_labs = [srgb_to_lab(r, g, b) for r, g, b in dominant_rgbs]
    return dominant_labs, suggested


# ──────────────────────────────────────────────────────────────
# Background detection and removal
# ──────────────────────────────────────────────────────────────

def detect_background(img, threshold=30):
    """Detect if image has a uniform background that can be removed.
    
    Returns (has_background: bool, bg_color: tuple or None)
    """
    if img.mode == 'RGBA':
        alpha = img.split()[3]
        # Check if alpha channel has significant transparency
        alpha_pixels = list(alpha.getdata())
        transparent_count = sum(1 for a in alpha_pixels if a < 128)
        if transparent_count > len(alpha_pixels) * 0.05:
            return True, None  # Already has transparency
    
    # Sample border pixels to detect background color
    w, h = img.size
    rgb_img = img.convert('RGB') if img.mode == 'RGBA' else img
    border_pixels = []
    # Top and bottom edges
    for x in range(0, w, max(1, w // 50)):
        border_pixels.append(rgb_img.getpixel((x, 0))[:3])
        border_pixels.append(rgb_img.getpixel((x, h - 1))[:3])
    # Left and right edges
    for y in range(0, h, max(1, h // 50)):
        border_pixels.append(rgb_img.getpixel((0, y))[:3])
        border_pixels.append(rgb_img.getpixel((w - 1, y))[:3])
    
    if not border_pixels:
        return False, None
    
    # Find most common border color
    counts = Counter(border_pixels)
    most_common_color, most_common_count = counts.most_common(1)[0]
    
    # Check if borders are mostly one color (>70%)
    if most_common_count < len(border_pixels) * 0.70:
        return False, None
    
    # Check if that color is near-white or a solid uniform color
    r, g, b = most_common_color
    brightness = (r + g + b) / 3
    
    # Count how many border pixels match this color closely
    matching = 0
    for px in border_pixels:
        pr, pg, pb = px
        if abs(pr - r) + abs(pg - g) + abs(pb - b) < threshold * 3:
            matching += 1
    
    if matching < len(border_pixels) * 0.70:
        return False, None
    
    return True, most_common_color


def remove_background(img, bg_color, threshold=30):
    """Remove background color from image, returning RGBA with transparency."""
    img_rgb = img.convert('RGB')
    result = img_rgb.copy().convert('RGBA')
    w, h = img_rgb.size
    
    for y in range(h):
        for x in range(w):
            r, g, b = img_rgb.getpixel((x, y))[:3]
            br, bg, bb = bg_color
            if abs(r - br) + abs(g - bg) + abs(b - bb) < threshold * 3:
                result.putpixel((x, y), (r, g, b, 0))  # Transparent
    
    return result


# ──────────────────────────────────────────────────────────────
# Pixel-art processing
# ──────────────────────────────────────────────────────────────

def enforce_pixel_art(img_resized, grid_w, grid_h, palette):
    """Ensure every cell has exactly ONE pure color from palette — no blending.
    
    This post-processing step removes any intermediate colors that might
    result from resampling, enforcing strict pixel-art style.
    """
    result = Image.new('RGB', (grid_w, grid_h))
    for y in range(grid_h):
        for x in range(grid_w):
            rgb = img_resized.getpixel((x, y))[:3]
            best = nearest_color(rgb, palette)
            result.putpixel((x, y), tuple(best['rgb']))
    return result


# ──────────────────────────────────────────────────────────────
# Font loading
# ──────────────────────────────────────────────────────────────

def load_font(size=12):
    """Load a suitable font, trying multiple fallbacks."""
    candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def auto_size_grid(img_w, img_h, target_beads=2000):
    """Auto-calculate grid size from image dimensions, preserving aspect ratio.
    
    当选择"不限制网格尺寸"时使用此函数。
    根据图片复杂度动态调整目标豆数，优先还原图片细节。
    """
    aspect = img_w / img_h
    
    # 根据图片尺寸和复杂度动态调整目标豆数
    # 大图或高细节图片需要更多豆子来还原细节
    image_pixels = img_w * img_h
    
    # 基础目标豆数：根据图片面积调整
    if image_pixels > 2000000:  # 大于200万像素
        target_beads = 4000  # 大图需要更多豆子
    elif image_pixels > 500000:  # 大于50万像素
        target_beads = 3000  # 中等图片
    else:
        target_beads = 2000  # 小图使用较少豆子
    
    # 计算网格尺寸
    grid_h = int(math.sqrt(target_beads / aspect))
    grid_w = int(grid_h * aspect)
    
    # 限制在合理范围内（15-120），允许更大网格以还原细节
    grid_w = max(15, min(120, grid_w))
    grid_h = max(15, min(120, grid_h))
    
    # 四舍五入到5的倍数，便于定位
    grid_w = 5 * round(grid_w / 5)
    grid_h = 5 * round(grid_h / 5)
    
    # 确保最小尺寸
    grid_w = max(15, grid_w)
    grid_h = max(15, grid_h)
    
    return grid_w, grid_h


def analyze_image_complexity(img, sample_step=4):
    """分析图片复杂度，返回复杂度分数。
    
    复杂度考虑因素：
    1. 颜色数量：颜色越多越复杂
    2. 边缘密度：边缘越多越复杂
    3. 细节丰富度：小区域变化越多越复杂
    
    返回：complexity_score (0-10)，越高越复杂
    """
    w, h = img.size
    pixels = []
    
    # 采样像素
    for y in range(0, h, sample_step):
        for x in range(0, w, sample_step):
            pixels.append(img.getpixel((x, y))[:3])
    
    if not pixels:
        return 5.0  # 默认中等复杂度
    
    # 1. 颜色数量分析
    STEP = 32
    bucket_map = {}
    for r, g, b in pixels:
        key = (r // STEP, g // STEP, b // STEP)
        bucket_map.setdefault(key, []).append((r, g, b))
    
    n_colors = len(bucket_map)
    
    # 2. 边缘密度分析（简化版）
    edge_count = 0
    grid_size = 8  # 将图片分成8x8的块
    block_w = w // grid_size
    block_h = h // grid_size
    
    for by in range(grid_size):
        for bx in range(grid_size):
            # 计算块内平均颜色
            x0 = bx * block_w
            y0 = by * block_h
            x1 = min(x0 + block_w, w)
            y1 = min(y0 + block_h, h)
            
            block_pixels = []
            for py in range(y0, y1, sample_step):
                for px in range(x0, x1, sample_step):
                    if px < w and py < h:
                        block_pixels.append(img.getpixel((px, py))[:3])
            
            if block_pixels:
                avg_r = sum(p[0] for p in block_pixels) // len(block_pixels)
                avg_g = sum(p[1] for p in block_pixels) // len(block_pixels)
                avg_b = sum(p[2] for p in block_pixels) // len(block_pixels)
                
                # 与相邻块比较
                if bx < grid_size - 1:
                    # 右边块
                    rx0 = (bx + 1) * block_w
                    right_pixels = []
                    for py in range(y0, y1, sample_step):
                        for px in range(rx0, min(rx0 + block_w, w), sample_step):
                            if px < w and py < h:
                                right_pixels.append(img.getpixel((px, py))[:3])
                    
                    if right_pixels:
                        r_avg = sum(p[0] for p in right_pixels) // len(right_pixels)
                        g_avg = sum(p[1] for p in right_pixels) // len(right_pixels)
                        b_avg = sum(p[2] for p in right_pixels) // len(right_pixels)
                        
                        diff = abs(avg_r - r_avg) + abs(avg_g - g_avg) + abs(avg_b - b_avg)
                        if diff > 50:  # 颜色差异阈值
                            edge_count += 1
    
    # 3. 计算复杂度分数
    # 颜色数量分数 (0-4)
    if n_colors <= 10:
        color_score = 1
    elif n_colors <= 30:
        color_score = 2
    elif n_colors <= 60:
        color_score = 3
    else:
        color_score = 4
    
    # 边缘密度分数 (0-3)
    max_edges = grid_size * (grid_size - 1) * 2  # 最大可能的边缘数
    edge_ratio = edge_count / max_edges if max_edges > 0 else 0
    if edge_ratio < 0.2:
        edge_score = 1
    elif edge_ratio < 0.4:
        edge_score = 2
    else:
        edge_score = 3
    
    # 细节丰富度分数 (0-3) - 基于颜色变化
    # 简化版：使用颜色数量作为细节指标
    detail_score = min(3, n_colors // 20)
    
    # 总复杂度分数 (0-10)
    complexity_score = color_score + edge_score + detail_score
    
    return min(10, complexity_score)


def auto_size_grid_with_complexity(img_w, img_h, complexity_score):
    """根据图片复杂度自动计算网格尺寸。
    
    复杂度越高，网格尺寸越大，以还原更多细节。
    
    Args:
        img_w: 图片宽度
        img_h: 图片高度
        complexity_score: 复杂度分数 (0-10)
    
    Returns:
        (grid_w, grid_h): 网格尺寸
    """
    aspect = img_w / img_h
    
    # 根据复杂度调整目标豆数
    # 复杂度 0-3: 简单图片，2000-3000颗
    # 复杂度 4-6: 中等图片，3000-5000颗
    # 复杂度 7-10: 复杂图片，5000-8000颗
    
    if complexity_score <= 3:
        target_beads = 2000 + (complexity_score * 333)  # 2000-3000
    elif complexity_score <= 6:
        target_beads = 3000 + ((complexity_score - 3) * 666)  # 3000-5000
    else:
        target_beads = 5000 + ((complexity_score - 6) * 750)  # 5000-8000
    
    # 根据图片尺寸调整
    image_pixels = img_w * img_h
    if image_pixels > 2000000:  # 大于200万像素
        target_beads = int(target_beads * 1.5)  # 大图增加50%
    elif image_pixels > 500000:  # 大于50万像素
        target_beads = int(target_beads * 1.2)  # 中等图片增加20%
    
    # 计算网格尺寸
    grid_h = int(math.sqrt(target_beads / aspect))
    grid_w = int(grid_h * aspect)
    
    # 限制在合理范围内（15-120），允许更大网格以还原细节
    grid_w = max(15, min(120, grid_w))
    grid_h = max(15, min(120, grid_h))
    
    # 四舍五入到5的倍数，便于定位
    grid_w = 5 * round(grid_w / 5)
    grid_h = 5 * round(grid_h / 5)
    
    # 确保最小尺寸
    grid_w = max(15, grid_w)
    grid_h = max(15, grid_h)
    
    return grid_w, grid_h


def trim_whitespace(img, threshold=250):
    """Trim near-white / transparent borders from an image."""
    if img.mode == 'RGBA':
        alpha = img.split()[3]
        bbox = alpha.getbbox()
    else:
        gray = img.convert('L')
        inverted = ImageOps.invert(gray)
        bbox = inverted.getbbox()
    if bbox:
        return img.crop(bbox), bbox
    return img, None


def parse_size(s):
    """Parse WxH string like '29x29' or '29'."""
    if 'x' in s.lower():
        parts = s.lower().split('x')
        return int(parts[0].strip()), int(parts[1].strip())
    else:
        v = int(s.strip())
        return v, v


# ──────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────

def generate_pattern(image_path, output_path, grid_w=29, grid_h=29,
                     cell_size=24, max_colors=221, show_symbols=True,
                     show_legend=True, bg_remove=False, bg_threshold=30):
    """Generate QQ拼豆施工图 PNG."""

    # ── Load image ──
    img = Image.open(image_path).convert('RGBA')

    # ── Background removal ──
    has_bg, bg_color = detect_background(img, bg_threshold)
    if bg_remove and has_bg:
        if bg_color:
            img = remove_background(img, bg_color, bg_threshold)
            print(f'🗑️  Removed background (color: rgb{bg_color})')
        else:
            # Already has transparency from alpha
            print('🗑️  Image already has transparent background')

    # ── Trim whitespace ──
    img, _ = trim_whitespace(img)

    # ── Composite on white background ──
    if img.mode == 'RGBA':
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert('RGB')
    else:
        img = img.convert('RGB')

    # ── Sharpen ──
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # ── Area-aware resize with color guard ──
    orig_w, orig_h = img.size
    cell_w = orig_w / grid_w
    cell_h = orig_h / grid_h
    img_resized = Image.new('RGB', (grid_w, grid_h))

    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = int(gx * cell_w)
            x1 = min(int((gx + 1) * cell_w) + 1, orig_w)
            y0 = int(gy * cell_h)
            y1 = min(int((gy + 1) * cell_h) + 1, orig_h)
            region = img.crop((x0, y0, x1, y1))
            pixels = list(region.getdata())
            if not pixels:
                img_resized.putpixel((gx, gy), (255, 255, 255))
                continue

            counts = Counter(pixels)
            most_common = counts.most_common(1)[0]
            base_color = most_common[0][:3]
            base_brightness = sum(base_color) / 3

            # Find most vivid color that's different from base
            vivid_color = None
            vivid_count = 0
            for color, count in counts.items():
                c = color[:3]
                if c == base_color:
                    continue
                sat = max(c) - min(c)
                if sat > 30 and count > vivid_count:
                    vivid_color = c
                    vivid_count = count

            # Color guard: if base is gray/white AND vivid color >= 20% presence
            threshold = max(1, len(pixels) * 0.20)
            if vivid_color and vivid_count >= threshold and base_brightness > 180:
                img_resized.putpixel((gx, gy), vivid_color)
            else:
                img_resized.putpixel((gx, gy), base_color)

    # ── Load palette ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    palette_path = os.path.join(script_dir, '..', 'references', 'mard221_palette.json')
    with open(palette_path, 'r', encoding='utf-8') as f:
        full_palette = json.load(f)

    for entry in full_palette:
        if 'lab' not in entry:
            entry['lab'] = list(srgb_to_lab(*entry['rgb']))

    # ── Auto-detect palette ──
    if max_colors == 0:
        dominant_labs, suggested = analyze_image_colors(img_resized)
        if dominant_labs:
            seen_ids = set()
            matched_palette = []
            for dlab in dominant_labs:
                best, best_dist = full_palette[0], float('inf')
                for entry in full_palette:
                    d = delta_e(dlab, entry['lab'])
                    if d < best_dist:
                        best_dist = d
                        best = entry
                if best['id'] not in seen_ids:
                    seen_ids.add(best['id'])
                    matched_palette.append(best)
            if len(matched_palette) < suggested:
                for entry in full_palette:
                    if entry['id'] not in seen_ids and len(matched_palette) < suggested:
                        matched_palette.append(entry)
            palette = matched_palette
            print(f'🎨 Auto-detected {len(palette)} colors from image')
        else:
            palette = full_palette[:suggested]
            print(f'🎨 Auto-detected {suggested} colors from image')
    else:
        palette = full_palette[:max_colors]

    # ── Enforce pixel art: snap every cell to exact palette color ──
    img_pixel = enforce_pixel_art(img_resized, grid_w, grid_h, palette)

    # ── Map colors and count usage ──
    color_map = {}
    usage_counter = {}
    for y in range(grid_h):
        for x in range(grid_w):
            entry = nearest_color(img_pixel.getpixel((x, y)), palette)
            color_map[(y, x)] = entry
            cid = entry['id']
            usage_counter[cid] = usage_counter.get(cid, 0) + 1

    used_colors = sorted(usage_counter.items(), key=lambda kv: -kv[1])
    id_to_palette = {e['id']: e for e in palette}

    # ══════════════════════════════════════════════════════════
    # Layout: QQ拼豆施工图版式
    # ══════════════════════════════════════════════════════════

    fname = os.path.basename(os.path.splitext(image_path)[0])
    n_colors = len(used_colors)
    total_beads = grid_w * grid_h

    # Layout dimensions
    padding = 16
    title_h = 48          # Title + stats (two lines)
    col_header_h = 30     # Column numbers on top
    row_header_w = 44     # Row numbers on left
    legend_h = 0          # Will be calculated

    # Calculate legend height (bottom, horizontal rectangular style)
    if show_legend:
        legend_cols = 8   # 8 items per row (like reference)
        legend_rows = math.ceil(len(used_colors) / legend_cols)
        legend_h = 36 + legend_rows * 34 + 8  # title + items + padding
    else:
        legend_h = 0

    grid_px_w = grid_w * cell_size
    grid_px_h = grid_h * cell_size

    canvas_w = padding + row_header_w + grid_px_w + padding
    canvas_h = padding + title_h + col_header_h + grid_px_h + legend_h + padding

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_title = load_font(22)
    font_subtitle = load_font(13)
    font_header = load_font(12)
    font_symbol = load_font(max(8, min(11, cell_size // 3)))
    font_legend_title = load_font(14)
    font_legend = load_font(11)

    # ── Title: QQ拼豆施工图 + stats on second line ──
    title_y = padding
    title_text = f"QQ拼豆施工图 — {fname}"
    draw.text((padding, title_y), title_text, fill=(20, 20, 20), font=font_title)
    # Stats line (replaces old bottom stats)
    stats_text = f"总豆数: {total_beads}颗  |  网格: {grid_w}×{grid_h}  |  颜色: {n_colors}色"
    draw.text((padding, title_y + 28), stats_text, fill=(100, 100, 100), font=font_subtitle)

    # Grid origin
    origin_x = padding + row_header_w
    origin_y = padding + title_h + col_header_h

    # ── Column headers (numbers from 1) ──
    for x in range(grid_w):
        cx = origin_x + x * cell_size + cell_size // 2
        cy = origin_y - 20
        num = str(x + 1)
        tw = draw.textlength(num, font=font_header)
        draw.text((cx - tw / 2, cy), num, fill=(90, 90, 90), font=font_header)

    # ── Row headers (numbers from 1) ──
    for y in range(grid_h):
        cy = origin_y + y * cell_size + cell_size // 2 - 7
        cx = origin_x - 14
        num = str(y + 1)
        tw = draw.textlength(num, font=font_header)
        draw.text((cx - tw, cy), num, fill=(90, 90, 90), font=font_header)

    # ── Draw grid cells ──
    for y in range(grid_h):
        for x in range(grid_w):
            entry = color_map[(y, x)]
            px = origin_x + x * cell_size
            py = origin_y + y * cell_size

            # Pure fill — pixel art style
            draw.rectangle(
                [px, py, px + cell_size - 1, py + cell_size - 1],
                fill=tuple(entry['rgb'])
            )

            # Grid lines
            draw.rectangle(
                [px, py, px + cell_size - 1, py + cell_size - 1],
                outline=(180, 180, 180), width=1
            )

            # Color symbol in cell center
            if show_symbols and cell_size >= 12:
                sym = entry['id']
                r, g, b = entry['rgb']
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = (255, 255, 255) if lum < 128 else (20, 20, 20)
                out_color = (0, 0, 0) if lum < 128 else (255, 255, 255)

                tw = draw.textlength(sym, font=font_symbol)
                tx = px + cell_size / 2 - tw / 2
                ty = py + cell_size / 2 - font_symbol.size * 0.6

                # Outline for readability
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            draw.text((tx + dx, ty + dy), sym, fill=out_color, font=font_symbol)
                draw.text((tx, ty), sym, fill=txt_color, font=font_symbol)

    # ── 5×5 bold reference lines ──
    line_color = (80, 80, 80)
    for x in range(0, grid_w + 1, 5):
        lx = origin_x + x * cell_size
        draw.line([(lx, origin_y), (lx, origin_y + grid_px_h)], fill=line_color, width=3)
    for y in range(0, grid_h + 1, 5):
        ly = origin_y + y * cell_size
        draw.line([(origin_x, ly), (origin_x + grid_px_w, ly)], fill=line_color, width=3)

    # ── Bottom color legend (horizontal rectangular swatch style) ──
    # Style reference: wide rectangular swatches, compact layout, 8 per row
    if show_legend and used_colors:
        legend_x = padding
        legend_y = origin_y + grid_px_h + 16
        draw.text((legend_x, legend_y), "🎨 色卡图例", fill=(20, 20, 20), font=font_legend_title)
        legend_y += 28

        # Horizontal layout: 8 items per row, rectangular swatches
        legend_cols = 8
        swatch_w = 22      # Rectangular swatch width
        swatch_h = 16      # Rectangular swatch height
        item_w = (grid_w * cell_size) // legend_cols  # Evenly distributed
        item_h = 34

        for i, (cid, count) in enumerate(used_colors):
            col = i % legend_cols
            row = i // legend_cols
            entry = id_to_palette.get(cid, {})
            rgb = tuple(entry.get('rgb', [200, 200, 200]))

            ix = legend_x + col * item_w
            iy = legend_y + row * item_h

            # Rectangular color swatch (wider than tall)
            draw.rectangle(
                [ix, iy + 4, ix + swatch_w, iy + swatch_h + 4],
                fill=rgb, outline=(120, 120, 120), width=1
            )
            # Label: ID (count)
            label = f"{cid} ({count})"
            draw.text((ix + swatch_w + 5, iy + 3), label, fill=(40, 40, 40), font=font_legend)

    # ── Save ──
    canvas.save(output_path, 'PNG')

    return {
        'total_beads': total_beads,
        'color_count': n_colors,
        'usage': used_colors,
        'output_path': os.path.abspath(output_path),
    }


# ──────────────────────────────────────────────────────────────
# CLI entry
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QQ拼豆施工图生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python generate_pattern.py cat.jpg -o cat_pattern.png
              python generate_pattern.py photo.png -o pattern.png --size 60x60
              python generate_pattern.py pic.jpg -o big.png --bg-remove --size 80x80
              python generate_pattern.py complex.jpg -o detail.png --detail-mode
        """)
    )
    parser.add_argument('image', help='输入图片路径 (JPG/PNG/WEBP/BMP)')
    parser.add_argument('-o', '--output', required=True, help='输出PNG路径')
    parser.add_argument('--size', default='auto', help='网格尺寸: auto (默认) 或 WxH')
    parser.add_argument('--cellsize', type=int, default=24, help='每格像素大小 (默认24)')
    parser.add_argument('--max-colors', type=int, default=0, help='最大颜色数: 0=自动 (默认), 221=全色卡')
    parser.add_argument('--bg-remove', action='store_true', help='强制去除背景')
    parser.add_argument('--bg-threshold', type=int, default=30, help='背景检测阈值 (默认30)')
    parser.add_argument('--no-symbols', action='store_true', help='不标注色号符号')
    parser.add_argument('--no-legend', action='store_true', help='不显示色卡图例')
    parser.add_argument('--detail-mode', action='store_true', help='细节模式：优先还原图片细节（自动分析复杂度）')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f'[ERROR] File not found: {args.image}', file=sys.stderr)
        sys.exit(1)

    # Auto-size
    if args.size.lower() == 'auto':
        img_check = Image.open(args.image).convert('RGB')
        img_w, img_h = img_check.size
        
        # 根据复杂度自动计算网格尺寸
        complexity_score = analyze_image_complexity(img_check)
        grid_w, grid_h = auto_size_grid_with_complexity(img_w, img_h, complexity_score)
        
        print(f'📐 Auto grid (complexity: {complexity_score:.1f}/10): {img_w}×{img_h}px → {grid_w}×{grid_h} ({grid_w*grid_h} beads)')
    else:
        grid_w, grid_h = parse_size(args.size)

    # Background detection (when not auto-removing)
    if not args.bg_remove:
        img_check = Image.open(args.image)
        has_bg, bg_color = detect_background(img_check, args.bg_threshold)
        if has_bg:
            color_desc = f"rgb{bg_color}" if bg_color else "transparent"
            print(f'🔍 检测到明显背景 ({color_desc})，建议去除背景')
            print(f'   提示: 如需去除，请加 --bg-remove 参数重新运行')

    result = generate_pattern(
        image_path=args.image,
        output_path=args.output,
        grid_w=grid_w,
        grid_h=grid_h,
        cell_size=args.cellsize,
        max_colors=args.max_colors,
        show_symbols=not args.no_symbols,
        show_legend=not args.no_legend,
        bg_remove=args.bg_remove,
        bg_threshold=args.bg_threshold,
    )

    print(f'✅ 施工图已生成: {result["output_path"]}')
    print(f'   网格: {grid_w}×{grid_h}')
    print(f'   总豆数: {result["total_beads"]}')
    print(f'   颜色: {result["color_count"]}色')
    print(f'   用量: {", ".join(f"{c}×{n}" for c, n in result["usage"][:8])}{"..." if len(result["usage"]) > 8 else ""}')
    print(f'OUTPUT={result["output_path"]}')


if __name__ == '__main__':
    main()
