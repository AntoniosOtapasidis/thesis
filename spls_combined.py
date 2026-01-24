import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader
import fitz  # PyMuPDF

# Read the PDF files
pdf2_path = "/users/antonios/saspls_viral_loadings_species_level_no_normalization.pdf"
pdf1_path = "/users/antonios/saspls_X__MAG_loadings_species_level_no_normalization.pdf"

# Create figure with 1 row and 2 columns (side by side)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Convert PDFs to images and display them
doc1 = fitz.open(pdf1_path)
doc2 = fitz.open(pdf2_path)

# Get first page of each PDF
page1 = doc1[0]
page2 = doc2[0]

# Render pages as images
pix1 = page1.get_pixmap(dpi=300)
pix2 = page2.get_pixmap(dpi=300)

# Convert to numpy arrays
import numpy as np
img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, pix1.n)
img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, pix2.n)

# Display images
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('a', fontsize=16, fontweight='bold', loc='left')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('b', fontsize=16, fontweight='bold', loc='left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined panel
plt.savefig('/users/antonios/combine_spls_panel.png', dpi=300, bbox_inches='tight')
plt.savefig('/users/antonios/combine_spls_panel.pdf', dpi=300, bbox_inches='tight')

print("Combined panel saved as combined_panel.png and combined_panel.pdf")

# Close the PDF documents
doc1.close()
doc2.close()