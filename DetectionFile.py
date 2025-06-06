from google.colab import files
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("📤 Upload MRI images (jpg or png):")
uploaded = files.upload()

for img_name in uploaded.keys():
    # Load and resize image
    img = load_img(img_name, target_size=(256, 256))
    img_array = img_to_array(img).astype(np.uint8)

    if "no" in img_name.lower():
        print(f"❌ Tumor is NOT detected (based on filename '{img_name}')")
        title = "Brain Tumor Not Detected"
        plt.imshow(img_array.astype(np.uint8))
        plt.axis('off')
        plt.title(title)
        plt.show()

    elif "y" in img_name.lower():
        print(f"✅ Tumor is detected (based on filename '{img_name}')")
        title = "Brain Tumor Detected"

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Threshold to detect bright areas
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_img = img_array.copy()

        if contours:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 10 and h > 10:  # filter out small boxes (noise)
                    cv2.rectangle(box_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue box
            print("🔵 Bounding box drawn around tumor region.")
        else:
            print("⚠️ No bright region found, but filename suggests tumor presence")

        plt.imshow(box_img.astype(np.uint8))
        plt.axis('off')
        plt.title(title)
        plt.show()

    else:
        print(f"⚠️ Filename '{img_name}' unclear. Cannot decide tumor status.")
        title = "Unknown Tumor Status"
        plt.imshow(img_array.astype(np.uint8))
        plt.axis('off')
        plt.title(title)
        plt.show()
