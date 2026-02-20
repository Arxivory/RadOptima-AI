import matplotlib.pyplot as plt
from data.dicom_loader import load_dicom_volume

DATA_PATH = "data/samples/chest"

try:
    volume, volume_scale = load_dicom_volume(DATA_PATH)

    slice_idx = [volume.shape[0] // 2]  

    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(volume[slice_idx[0], :, :], cmap="bone", vmin=-200, vmax=500)
    ax.set_title(f"Chest Slice {slice_idx[0]} - HU Normalized")
    cbar = plt.colorbar(img, ax=ax, label="Hounsfield Units (HU)")

    def update_slice(event):
        if event.key == "right":
            slice_idx[0] = min(slice_idx[0] + 1, volume.shape[0] - 1)
        elif event.key == "left":
            slice_idx[0] = max(slice_idx[0] - 1, 0)

        img.set_data(volume[slice_idx[0], :, :])
        ax.set_title(f"Chest Slice {slice_idx[0]} - HU Normalized")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", update_slice)
    plt.show()

except Exception as e:
    print(f"Error {e}")