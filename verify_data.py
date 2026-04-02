import matplotlib.pyplot as plt
from data.dicom_loader import load_dicom_volume
import numpy as np

DATA_PATH = "data/samples/chest"
AI_DATA_PATH = "data/temp_ai_out"

try:
    # Load original volume
    volume, volume_scale = load_dicom_volume(DATA_PATH)

    # Load AI denoised volume (assuming same shape)
    ai_volume, ai_volume_scale = load_dicom_volume(AI_DATA_PATH)

    slice_idx = [volume.shape[0] // 2]  

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # Plot normal slice
    img1 = axes[0].imshow(volume[slice_idx[0], :, :], cmap="bone", vmin=-200, vmax=500)
    axes[0].set_title(f"Normal Chest Slice {slice_idx[0]} - HU Normalized")
    cbar1 = plt.colorbar(img1, ax=axes[0], label="Hounsfield Units (HU)")

    # Plot AI denoised slice
    img2 = axes[1].imshow(ai_volume[slice_idx[0], :, :], cmap="bone", vmin=-200, vmax=500)
    axes[1].set_title(f"AI Denoised Chest Slice {slice_idx[0]} - HU Normalized")
    cbar2 = plt.colorbar(img2, ax=axes[1], label="Hounsfield Units (HU)")

    def update_slice(event):
        if event.key == "right":
            slice_idx[0] = min(slice_idx[0] + 1, volume.shape[0] - 1)
        elif event.key == "left":
            slice_idx[0] = max(slice_idx[0] - 1, 0)

        # Update both images
        img1.set_data(volume[slice_idx[0], :, :])
        axes[0].set_title(f"Normal Chest Slice {slice_idx[0]} - HU Normalized")

        img2.set_data(ai_volume[slice_idx[0], :, :])
        axes[1].set_title(f"AI Denoised Chest Slice {slice_idx[0]} - HU Normalized")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", update_slice)
    plt.show()

except Exception as e:
    print(f"Error {e}")