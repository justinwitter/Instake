import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def find_best_match(original, cutout):
    """Find the best matching position for the cutout in the original image using template matching"""
    # Convert PIL images to OpenCV format
    orig_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    cutout_cv = cv2.cvtColor(np.array(cutout), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for template matching
    orig_gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
    cutout_gray = cv2.cvtColor(cutout_cv, cv2.COLOR_BGR2GRAY)
    
    # Check if cutout has enough content
    cutout_content = np.sum(cutout_gray < 240)  # Count non-white pixels
    if cutout_content < 100:  # Too sparse for reliable matching
        st.warning("⚠️ Cutout appears to have very little content. Automatic matching may be unreliable.")
    
    # Apply edge detection with multiple parameter sets for technical drawings
    edge_params = [(50, 150), (30, 100), (100, 200)]
    
    best_match_method = None
    best_confidence = -1
    best_position = (0, 0)
    debug_info = []
    
    for low_thresh, high_thresh in edge_params:
        orig_edges = cv2.Canny(orig_gray, low_thresh, high_thresh)
        cutout_edges = cv2.Canny(cutout_gray, low_thresh, high_thresh)
        
        # Skip if cutout edges are too sparse
        if np.sum(cutout_edges) < 50:
            continue
        
        # Try multiple template matching methods
        methods = [
            ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
            ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
            ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
        ]
        
        for method_name, method in methods:
            # Try both original grayscale and edge-detected versions
            test_cases = [
                ('Original', cutout_gray, orig_gray),
                ('Edges', cutout_edges, orig_edges)
            ]
            
            for case_name, template_img, search_img in test_cases:
                if np.sum(template_img) == 0:  # Skip if template is empty
                    continue
                    
                try:
                    result = cv2.matchTemplate(search_img, template_img, method)
                    
                    if method == cv2.TM_SQDIFF_NORMED:
                        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                        confidence = 1 - min_val  # Convert to higher-is-better
                        position = min_loc
                    else:
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        confidence = max_val
                        position = max_loc
                    
                    debug_info.append({
                        'method': f"{method_name}_{case_name}",
                        'confidence': confidence,
                        'position': position,
                        'edge_params': f"{low_thresh}-{high_thresh}"
                    })
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_position = position
                        best_match_method = f"{method_name}_{case_name}"
                        
                except Exception as e:
                    continue
    
    # Return debug info as well
    return best_position, best_confidence, debug_info, best_match_method

def overlay_images(original, cutout, position, opacity=0.7):
    """Overlay the cutout on the original image at the specified position"""
    orig_array = np.array(original)
    cutout_array = np.array(cutout)
    
    x, y = position
    h, w = cutout_array.shape[:2]
    
    # Ensure the cutout fits within the original image bounds
    if x + w > orig_array.shape[1]:
        w = orig_array.shape[1] - x
        cutout_array = cutout_array[:, :w]
    if y + h > orig_array.shape[0]:
        h = orig_array.shape[0] - y
        cutout_array = cutout_array[:h, :]
    
    # Create a copy of the original image
    result = orig_array.copy()
    
    # Blend the images
    result[y:y+h, x:x+w] = (
        opacity * cutout_array + 
        (1 - opacity) * result[y:y+h, x:x+w]
    ).astype(np.uint8)
    
    return Image.fromarray(result)

def create_bordered_overlay(original, cutout, position, border_color, border_width):
    """Create overlay with a colored border that follows the cutout's outline/perimeter only"""
    orig_array = np.array(original)
    cutout_array = np.array(cutout)
    
    x, y = position
    h, w = cutout_array.shape[:2]
    
    # Ensure the cutout fits within the original image bounds
    if x + w > orig_array.shape[1]:
        w = orig_array.shape[1] - x
        cutout_array = cutout_array[:, :w]
    if y + h > orig_array.shape[0]:
        h = orig_array.shape[0] - y
        cutout_array = cutout_array[:h, :]
    
    # Create a copy of the original image
    result = orig_array.copy()
    
    # Place the cutout first
    result[y:y+h, x:x+w] = cutout_array
    
    # Create mask from cutout (identify non-white pixels)
    cutout_gray = cv2.cvtColor(cutout_array, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(cutout_gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to get the outline
    contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour_result) == 3:
        # OpenCV 3.x returns (image, contours, hierarchy)
        contours = contour_result[1]
    else:
        # OpenCV 4.x returns (contours, hierarchy)
        contours = contour_result[0]
    
    # Create a mask for just the border/outline
    border_mask = np.zeros_like(mask)
    
    # Draw the contours with the specified border width
    cv2.drawContours(border_mask, contours, -1, 255, thickness=border_width)
    
    # Apply border color only to the outline pixels
    border_indices = border_mask > 0
    result[y:y+h, x:x+w][border_indices] = border_color
    
    return Image.fromarray(result)

def calculate_cutout_measurements(cutout_img, pixels_per_inch):
    """Calculate area and perimeter of the cutout based on non-white pixels"""
    cutout_array = np.array(cutout_img)
    
    # Convert to grayscale if needed
    if len(cutout_array.shape) == 3:
        gray = cv2.cvtColor(cutout_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = cutout_array
    
    # Create binary mask (assuming black is background)
    # Threshold to identify non-background pixels
    #_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY) 
    
    # Calculate area in pixels (count of non-white pixels)
    area_pixels = np.sum(binary > 0)
    
    # Find contours for perimeter calculation (handle different OpenCV versions)
    contour_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour_result) == 3:
        # OpenCV 3.x returns (image, contours, hierarchy)
        contours = contour_result[1]
    else:
        # OpenCV 4.x returns (contours, hierarchy)
        contours = contour_result[0]
    
    # Calculate total perimeter in pixels
    perimeter_pixels = sum(cv2.arcLength(contour, True) for contour in contours)
    
    # Convert to real-world measurements
    pixels_per_inch_squared = pixels_per_inch ** 2
    area_square_inches = area_pixels / pixels_per_inch_squared
    perimeter_inches = perimeter_pixels / pixels_per_inch
    
    return {
        'area_pixels': area_pixels,
        'perimeter_pixels': perimeter_pixels,
        'area_square_inches': area_square_inches,
        'perimeter_inches': perimeter_inches,
        'binary_mask': binary
    }
    """Create overlay with a colored border around the cutout"""
    orig_array = np.array(original)
    cutout_array = np.array(cutout)
    
    x, y = position
    h, w = cutout_array.shape[:2]
    
    # Ensure the cutout fits within the original image bounds
    if x + w > orig_array.shape[1]:
        w = orig_array.shape[1] - x
        cutout_array = cutout_array[:, :w]
    if y + h > orig_array.shape[0]:
        h = orig_array.shape[0] - y
        cutout_array = cutout_array[:h, :]
    
    # Create a copy of the original image
    result = orig_array.copy()
    
    # Place the cutout
    result[y:y+h, x:x+w] = cutout_array
    
    # Add border
    cv2.rectangle(result, (x-border_width, y-border_width), 
                 (x+w+border_width, y+h+border_width), border_color, border_width)
    
    return Image.fromarray(result)

st.set_page_config(page_title="Image Cutout Overlay", layout="wide")

st.title(":toolbox: Instake")
st.markdown("Upload an original image and a cutout, and this app will automatically find its area and perimeter.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Images")
    
    original_file = st.file_uploader(
        "Choose the original image",
        type=['png', 'jpg', 'jpeg'],
        key="original"
    )
    
    cutout_file = st.file_uploader(
        "Choose the cutout image",
        type=['png', 'jpg', 'jpeg'],
        key="cutout"
    )

with col2:
    st.subheader("⚙️ Settings")
    
    # Add pixels per inch input
    pixels_per_inch = st.number_input(
        "Pixels per inch (for measurements)",
        min_value=.00000001,
        max_value=1200.0,
        value=0.985,
        step=1.0,
        help="Resolution of the original image - needed for accurate area/perimeter calculations"
    )
    
    overlay_mode = st.radio(
        "Overlay mode:",
        ["Automatic positioning", "Manual positioning"],
        help="Automatic uses template matching to find the best position",
    )
    
    display_mode = st.selectbox(
        "Display mode:",
        ["Blended overlay", "Cutout with border", "Side by side"],
        help="Choose how to display the result",
        index=1
    )
    
    if display_mode == "Blended overlay":
        opacity = st.slider("Overlay opacity", 0.1, 1.0, 0.7, 0.1)
    
    if display_mode == "Cutout with border":
        border_color = st.color_picker("Border color", "#FF0000")
        border_width = st.slider("Border width", 1, 10, 3)

if original_file and cutout_file:
    # Load images
    original_img = Image.open(original_file).convert('RGB')
    cutout_img = Image.open(cutout_file).convert('RGB')
    
    st.markdown("---")
    
    # Display original images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_img, use_column_width=True)
        st.caption(f"Size: {original_img.size[0]} x {original_img.size[1]} pixels")
    
    with col2:
        st.subheader("Cutout Image")
        st.image(cutout_img, use_column_width=True)
        st.caption(f"Size: {cutout_img.size[0]} x {cutout_img.size[1]} pixels")
    
    # Process overlay
    if overlay_mode == "Automatic positioning":
        with st.spinner("Finding best match position..."):
            result = find_best_match(original_img, cutout_img)
            position, confidence, debug_info, best_method = result
            st.success(f"Best match found at position ({position[0]}, {position[1]}) with confidence: {confidence:.3f}")
            st.info(f"Best method: {best_method}")
            
            # Show debug information
            with st.expander("🔍 Debug Information"):
                st.write("**All matching attempts:**")
                for i, info in enumerate(debug_info[:10]):  # Show top 10
                    st.write(f"{i+1}. {info['method']} (edges: {info['edge_params']}): "
                           f"confidence {info['confidence']:.3f} at {info['position']}")
            
            # Add option to try different scales if confidence is low
            if confidence < 0.3:
                st.warning("⚠️ Very low confidence detected. The cutout might be at a different scale or not from this image.")
                try_different_scale = st.checkbox("Try matching at different scales (slower but may improve results)")
                
                if try_different_scale:
                    with st.spinner("Trying different scales..."):
                        best_pos, best_conf, best_debug = position, confidence, debug_info
                        best_scale = 1.0
                        scales = [0.5, 0.8, 1.2, 1.5, 2.0]
                        
                        for scale in scales:
                            # Resize cutout
                            new_size = (int(cutout_img.size[0] * scale), int(cutout_img.size[1] * scale))
                            scaled_cutout = cutout_img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # Skip if scaled cutout is larger than original
                            if scaled_cutout.size[0] >= original_img.size[0] or scaled_cutout.size[1] >= original_img.size[1]:
                                continue
                                
                            try:
                                scale_result = find_best_match(original_img, scaled_cutout)
                                pos, conf, scale_debug, method = scale_result
                                st.info(f"Scale {scale}x: confidence {conf:.3f} at {pos}")
                                
                                if conf > best_conf:
                                    best_conf = conf
                                    best_pos = pos
                                    best_debug = scale_debug
                                    best_scale = scale
                                    cutout_img = scaled_cutout
                                    st.success(f"Better match found at scale {scale}x!")
                            except:
                                continue
                        
                        position, confidence, debug_info = best_pos, best_conf, best_debug
                        if best_scale != 1.0:
                            st.info(f"Final scale used: {best_scale}x")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            scale_factor = st.slider("Scale factor", 0.1, 3.0, 1.0, 0.05, help="Resize the cutout before positioning")
        
        # Apply scaling to cutout if different from 1.0
        if scale_factor != 1.0:
            new_size = (int(cutout_img.size[0] * scale_factor), int(cutout_img.size[1] * scale_factor))
            cutout_img = cutout_img.resize(new_size, Image.Resampling.LANCZOS)
            st.info(f"Cutout resized to: {cutout_img.size[0]} x {cutout_img.size[1]} pixels")
        
        with col2:
            x_pos = st.slider("X position", 0, max(0, original_img.size[0] - cutout_img.size[0]), 0, 1)
        with col3:
            y_pos = st.slider("Y position", 0, max(0, original_img.size[1] - cutout_img.size[1]), 0, 1)
        position = (x_pos, y_pos)
    
    # Create overlay based on display mode
    if display_mode == "Blended overlay":
        result_img = overlay_images(original_img, cutout_img, position, opacity)
        st.subheader("🎯 Result: Blended Overlay")
        st.image(result_img, use_column_width=True)
        
    elif display_mode == "Cutout with border":
        # Convert hex color to RGB
        border_rgb = tuple(int(border_color[i:i+2], 16) for i in (1, 3, 5))
        result_img = create_bordered_overlay(original_img, cutout_img, position, border_rgb, border_width)
        st.subheader("🎯 Result: Cutout with Border")
        st.image(result_img, use_column_width=True)
        
    else:  # Side by side
        st.subheader("🎯 Result: Side by Side Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(original_img, use_column_width=True)
        with col2:
            st.markdown("**With Cutout Overlay**")
            blended_result = overlay_images(original_img, cutout_img, position, 0.8)
            st.image(blended_result, use_column_width=True)
    
    # Calculate measurements
    st.markdown("---")
    st.subheader("📏 Measurements")
    
    measurements = calculate_cutout_measurements(cutout_img, pixels_per_inch)
    # Convert to other units
    area_sq_ft = measurements['area_square_inches'] / 144
    area_sq_cm = measurements['area_square_inches'] * 6.4516
    perimeter_ft = measurements['perimeter_inches'] / 12
    perimeter_cm = measurements['perimeter_inches'] * 2.54
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Area",
            f"{area_sq_ft:.3f} sq ft",
            help=f"Based on {measurements['area_pixels']} pixels"
        )
    
    with col2:
        st.metric(
            "Perimeter", 
            f"{perimeter_ft:.3f} ft",
            help=f"Based on {measurements['perimeter_pixels']:.1f} pixels"
        )
    
    with col3:
        st.metric(
            "Resolution",
            f"{pixels_per_inch} PPI",
            help="Pixels per inch used for calculations"
        )
    
    # Show measurement details
    with st.expander("📊 Measurement Details"):
        st.write("**Pixel Measurements:**")
        st.write(f"- Area: {measurements['area_pixels']:,} pixels")
        st.write(f"- Perimeter: {measurements['perimeter_pixels']:.1f} pixels")
        st.write(f"- Cutout dimensions: {cutout_img.size[0]} × {cutout_img.size[1]} pixels")
        
        st.write("**Real-world Measurements:**")
        st.write(f"- Area: {measurements['area_square_inches']:.6f} square inches")
        st.write(f"- Perimeter: {measurements['perimeter_inches']:.6f} inches")
        st.write(f"- Resolution: {pixels_per_inch} pixels per inch")
        
        
        
        st.write("**Alternative Units:**")
        st.write(f"- Area: {area_sq_ft:.6f} sq ft, {area_sq_cm:.3f} sq cm")
        st.write(f"- Perimeter: {perimeter_ft:.6f} ft, {perimeter_cm:.3f} cm")
        
        # Show the binary mask used for calculation
        st.write("**Detection Mask:**")
        st.image(measurements['binary_mask'], caption="White areas are counted for measurements", use_column_width=True)
        
        st.info("💡 The measurements are based on non-white pixels. Adjust the threshold in the code if your cutout has different background colors.")
    
    # Download button for measurements
    measurement_data = f"""Cutout Measurements Report
=====================================

Image Information:
- Cutout dimensions: {cutout_img.size[0]} × {cutout_img.size[1]} pixels
- Resolution: {pixels_per_inch} pixels per inch
- Scale factor: {scale_factor if 'scale_factor' in locals() else 1.0}

Area Measurements:
- {measurements['area_pixels']:,} pixels
- {measurements['area_square_inches']:.6f} square inches
- {area_sq_ft:.6f} square feet
- {area_sq_cm:.3f} square centimeters

Perimeter Measurements:
- {measurements['perimeter_pixels']:.1f} pixels
- {measurements['perimeter_inches']:.6f} inches
- {perimeter_ft:.6f} feet
- {perimeter_cm:.3f} centimeters

Position (if manually set):
- X: {position[0]} pixels
- Y: {position[1]} pixels
"""
    
    st.download_button(
        label="📊 Download Measurements Report",
        data=measurement_data,
        file_name="cutout_measurements.txt",
        mime="text/plain"
    )
    
    # Show matching details for automatic mode
    if overlay_mode == "Automatic positioning":
        with st.expander("🔍 Matching Details"):
            st.write(f"**Best match position:** ({position[0]}, {position[1]})")
            st.write(f"**Matching confidence:** {confidence:.3f}")
            st.write(f"**Cutout size:** {cutout_img.size[0]} x {cutout_img.size[1]} pixels")
            
            if confidence < 0.3:
                st.error("❌ Very low matching confidence. The cutout is likely not from this original image, or may need manual positioning.")
            elif confidence < 0.5:
                st.warning("⚠️ Low matching confidence. Consider trying manual positioning or checking if images match.")
            elif confidence > 0.7:
                st.success("✅ High matching confidence. Great match found!")
            else:
                st.info("ℹ️ Moderate matching confidence. Result should be reasonable.")
                
            st.markdown("""
            **Troubleshooting tips:**
            - If confidence is very low, try manual positioning
            - For line drawings/technical diagrams, edge detection is used to improve matching
            - Different scales are tried automatically for low-confidence matches
            - Make sure the cutout is actually from the original image
            """)

    # Download buttons
    col1, col2 = st.columns(2)
    
    if 'result_img' in locals():
        with col1:
            buf = io.BytesIO()
            result_img.save(buf, format='PNG')
            st.download_button(
                label="📥 Download Overlay Image",
                data=buf.getvalue(),
                file_name="overlay_result.png",
                mime="image/png"
            )


else:
    st.info("👆 Please upload a base image and a cutout to get started.")
    
    st.markdown("""
    ### 📋 How to use:
    1. **Clean floorplan**: Remove all text from floorplan using [ChatGPT](https://chatgpt.com/)
    2. **Extract cutout**: Use [Meta's SAM demo](https://segment-anything.com/demo) to extract area for measurement
    3. **Upload images**: Upload the base image and the extracted cutout
    4. **Choose overlay mode**: 
       - *Automatic*: Uses computer vision to find the best matching position
       - *Manual*: Lets you specify exact coordinates
    5. **Refine overlay**: Modify the scale of the cutout to match the base image if needed
    6. **View Results**: Check calculated perimeter and area based on set PPF (pixels per foot) 
    """)
