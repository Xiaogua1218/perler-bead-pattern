#!/usr/bin/env python3
"""
Build MARD 221 palette with pre-computed CIELAB values.
Reads mard221_palette.json (RGB only), computes LAB, writes enriched JSON.

Usage:
  python build_palette.py [input.json] [-o output.json]
"""

import json
import os
import sys
import math


def srgb_to_lab(r, g, b):
    """Convert sRGB (0-255) to CIELAB (D65 illuminant)."""
    # sRGB → linear RGB
    def linearize(c):
        c = c / 255.0
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)

    # Linear RGB → XYZ (D65)
    x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    # D65 reference white
    xn, yn, zn = 95.047, 100.000, 108.883

    def f(t):
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        return t / (3.0 * delta ** 2) + 4.0 / 29.0

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return round(L, 2), round(a, 2), round(b_val, 2)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'mard221_palette.json'
    output_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '-o' else input_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, '..', 'references', 'mard221_palette.json')

    with open(input_path, 'r', encoding='utf-8') as f:
        palette = json.load(f)

    for entry in palette:
        r, g, b = entry['rgb']
        L, a, b_val = srgb_to_lab(r, g, b)
        entry['lab'] = [L, a, b_val]

    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, '..', 'references', 'mard221_palette.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(palette, f, ensure_ascii=False, indent=2)

    print(f'✅ Built palette: {len(palette)} colors with LAB values → {output_path}')
    print(f'OUTPUT={os.path.abspath(output_path)}')


if __name__ == '__main__':
    main()
