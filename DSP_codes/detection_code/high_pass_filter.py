import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def high_pass_filter(input_image, passingRadius):

    gray_ori = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # Step 1: Perform FFT
    dft = np.fft.fft2(gray_ori)
    dft_shift = np.fft.fftshift(dft)

    # Step 2: Create a High-Pass Filter Mask
    rows, cols = gray_ori.shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

    # Create a mask with a small circle of zeros at the center (low-pass removed)
    radius = passingRadius  # Radius to suppress low frequencies
    mask = np.ones((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)  # Set the center to 0

    # Apply the mask to the shifted FFT
    high_pass = dft_shift * mask

    # Step 3: Perform Inverse FFT
    f_ishift = np.fft.ifftshift(high_pass)
    img_high_pass = np.abs(np.fft.ifft2(f_ishift))
    
    return img_high_pass



'''
current_path = os.getcwd()
image_path = os.path.join(current_path, 'DSP_codes','photos', 'IMG-3644.jpg')
original_image = cv2.imread(image_path)
res = high_pass_filter(original_image,5)

# Step 4: Visualize the Results
plt.figure(figsize=(12, 6))

# Filtered Image (Edges)
plt.subplot(1, 3, 3)
plt.imshow(res, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()
'''