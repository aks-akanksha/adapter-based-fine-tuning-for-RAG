#!/usr/bin/env python3
"""Convert PNG plots to JPG format."""
from PIL import Image
import os

plots = [
    'plot_1_performance_vs_efficiency.png',
    'plot_2_performance_comparison.png',
    'plot_3_cost_comparison.png',
    'plot_4_model_deep_dive.png'
]

for png_file in plots:
    if os.path.exists(png_file):
        print(f"Converting {png_file} to JPG...")
        img = Image.open(png_file)
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        jpg_file = png_file.replace('.png', '.jpg')
        img.save(jpg_file, 'JPEG', quality=95, optimize=True)
        print(f"✅ Saved: {jpg_file}")

print("\n✅ All plots converted to JPG format!")

