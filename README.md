# Tire color line and imprinted character vision system development
![ori_8](https://user-images.githubusercontent.com/86955204/207246545-a3669cc4-fb01-45d4-b4e9-605dd27b35e5.jpg)


# Tire Line Segmentation and Analysis

This project uses a pretrained CNN model to segment and analyze key regions in tire images, such as center lines, color lines, and text lines. It outputs visual overlays and position annotations for each detected feature.

## 🔍 Features

- Segment tire-related regions using a sliding window approach
- Predict patch-level class using a CNN (`tire_64CNN101.h5`)
- Reconstruct full-size segmentation maps for:
  - Tire center line
  - Color guide line
  - Text region
- Compute centroid positions of each region
- Generate annotated images and text descriptions

## 🧠 Model

- Model: `tire_64CNN101.h5`
- Input: RGB image
- Output: Patch-level classification (background / color line / text)

## 🗂 Output Files

- `ori_<name>.jpg` – original image with annotations
- `tire_seg_<name>.jpg` – binary mask of tire region
- `line_seg_<name>.jpg` – binary mask of color guide
- `text_seg_<name>.jpg` – binary mask of text
- `color_patch_<name>.jpg` – cropped patch around color line
- `<name>.txt` – relative position (상 / 하) of color/text lines vs. tire center

