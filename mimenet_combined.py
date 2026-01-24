import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read the image files
img1_path = "/users/antonios/mimenet/MiMeNet/results/bacteriome-metabolome_num_layer_2_lr_0.001_layer_nodes_256_l2_0.001_l1_0.0001_dropout_0.25_100back_micro_after_grid/Images/cv_bg_correlation_distributions.png"
img2_path = "/users/antonios/mimenet/MiMeNet/results/bacteriome-metabolome_num_layer_2_lr_0.001_layer_nodes_256_l2_0.001_l1_0.0001_dropout_0.25_100back_micro_after_grid/Images/top_correlated_metabolites.png"

# Load images
img1 = mpimg.imread(img1_path)
img2 = mpimg.imread(img2_path)

# Create figure with 1 row and 2 columns (side by side)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Display images
axes[0].imshow(img1)

axes[0].axis('off')
axes[0].set_title('a', fontsize=18, fontweight='bold', loc='left', pad=10)

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('b', fontsize=18, fontweight='bold', loc='left', pad=10)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined panel
output_dir = "/users/antonios/mimenet/MiMeNet/results/bacteriome-metabolome_num_layer_2_lr_0.001_layer_nodes_256_l2_0.001_l1_0.0001_dropout_0.25_100back_micro_after_grid/Images"
plt.savefig(f'{output_dir}/mimenet_combined_panel.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/mimenet_combined_panel.pdf', dpi=300, bbox_inches='tight')

print(f"Combined panel saved as:")
print(f"  - {output_dir}/mimenet_combined_panel.png")
print(f"  - {output_dir}/mimenet_combined_panel.pdf")

plt.close()