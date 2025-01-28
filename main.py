import streamlit as st
import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def pixelate_image(img, pixel_size, style="retro"):
    img = img.convert("RGB")

    # Resize the image to a smaller size for pixelation
    img_small = img.resize(
        (img.width // pixel_size, img.height // pixel_size), Image.NEAREST
    )

    # Scale it back to original size
    result = img_small.resize(img.size, Image.NEAREST)

    if style == "retro":
        result = reduce_colors(result)
    elif style == "chibi":
        result = add_chibi_effect(result)
    elif style == "custom":
        result = apply_custom_theme(result)

    return result

def reduce_colors(img, num_colors=16):
    img = img.convert("RGB")
    np_img = np.array(img)
    pixels = np_img.reshape(-1, 3)

    # Apply k-means clustering for color reduction
    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.labels_

    quantized_pixels = np.array([new_colors[label] for label in labels])
    quantized_img = quantized_pixels.reshape(np_img.shape)
    return Image.fromarray(quantized_img)

def add_chibi_effect(img):
    pastel_overlay = Image.new("RGB", img.size, (255, 224, 255))  # Light pink
    return Image.blend(img, pastel_overlay, alpha=0.3)

def apply_custom_theme(img):
    neon_overlay = Image.new("RGB", img.size, (0, 255, 128))  # Neon green
    return Image.blend(img, neon_overlay, alpha=0.2)

def main():
    st.title("Pixel Character Creator")

    # Camera input
    st.write("Capture a photo or upload an image!")
    img_file = st.camera_input("Take a photo") or st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file)

        # Pixelation options
        st.sidebar.title("Pixelation Settings")
        pixel_size = st.sidebar.slider("Pixelation Level", min_value=10, max_value=25, value=15)
        style = st.sidebar.selectbox("Select Style", ["Retro 8-bit", "Chibi", "Custom Theme"])

        style_map = {"Retro 8-bit": "retro", "Chibi": "chibi", "Custom Theme": "custom"}
        selected_style = style_map[style]

        # Apply pixelation
        pixelated_img = pixelate_image(img, pixel_size, style=selected_style)

        # Display results
# Display results side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original Image", use_container_width=True)

        with col2:
            st.image(pixelated_img, caption="Pixelated Image", use_container_width=True)


        # Download option
        st.download_button(
            label="Download Pixelated Image",
            data=pixelated_img.tobytes(),
            file_name="pixelated_character.png",
            mime="image/png",
        )

if __name__ == "__main__":
    main()
