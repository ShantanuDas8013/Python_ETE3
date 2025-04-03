import os
import io
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import streamlit as st
from datetime import datetime
import base64
from io import BytesIO

# Cache the placeholder sports images to avoid reloading
@st.cache_data
def get_placeholder_sports_images():
    """Get placeholder sports images for each sport from local sample_images folder"""
    sport_image_files = {
        "Basketball": "Basketball.jpeg",
        "Tennis": "Tennis.jpeg",
        "Football": "Football.jpeg",
        "Cricket": "Cricket.jpeg",
    }
    
    sport_images = {}
    # Use absolute path for sample_images directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_images_dir = os.path.join(current_dir, "sample_images")
    
    # Create sample_images directory if it doesn't exist
    if not os.path.exists(sample_images_dir):
        os.makedirs(sample_images_dir)
        st.error(f"""
        Sample images folder not found. Please:
        1. Create a folder named 'sample_images' at: {sample_images_dir}
        2. Add the following image files: {', '.join(sport_image_files.values())}
        """)
        
    for sport, filename in sport_image_files.items():
        image_path = os.path.join(sample_images_dir, filename)
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                sport_images[sport] = img
                st.success(f"Successfully loaded {filename}")
            else:
                st.warning(f"Missing image file: {filename}")
                sport_images[sport] = create_text_image(f"Missing {filename}")
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
            sport_images[sport] = create_text_image(f"Error loading {filename}")
    
    return sport_images

def create_text_image(text, size=(300, 200)):
    """Create a simple image with text for placeholders"""
    img = Image.new('RGB', size, color=(245, 245, 245))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a system font, falling back to default if not available
        font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    # Handle different versions of PIL/Pillow
    try:
        # For newer versions of Pillow (>=8.0.0)
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
    except AttributeError:
        try:
            # For Pillow 7.x.x
            text_width, text_height = font.getsize(text)
        except AttributeError:
            # For older versions of Pillow
            text_width, text_height = draw.textsize(text, font=font)
    
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Add text to image
    draw.text(position, text, fill=(100, 100, 100), font=font)
    return img

def load_image(image_file):
    """Load an image file"""
    if image_file is not None:
        img = Image.open(image_file)
        return img
    return None

def resize_image(image, width=None, height=None, maintain_aspect=True):
    """Resize the image while optionally maintaining aspect ratio"""
    if image is None:
        return None
    
    img_copy = image.copy()
    if width and height and not maintain_aspect:
        return img_copy.resize((width, height), Image.LANCZOS)
    
    if width:
        wpercent = width / float(img_copy.size[0])
        hsize = int(float(img_copy.size[1]) * float(wpercent))
        return img_copy.resize((width, hsize), Image.LANCZOS)
    
    if height:
        hpercent = height / float(img_copy.size[1])
        wsize = int(float(img_copy.size[0]) * float(hpercent))
        return img_copy.resize((wsize, height), Image.LANCZOS)
    
    return img_copy

def apply_filter(image, filter_name="none"):
    """Apply different filters to an image"""
    if image is None:
        return None
    
    img_copy = image.copy()
    
    if filter_name == "grayscale":
        return ImageOps.grayscale(img_copy)
    elif filter_name == "sepia":
        # Apply sepia tone filter
        sepia_arr = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        # Convert to numpy array, apply matrix, and back to PIL
        img_arr = np.array(img_copy)
        sepia_img = Image.fromarray(np.uint8(np.clip(img_arr.dot(sepia_arr.T), 0, 255)))
        return sepia_img
    elif filter_name == "blur":
        return img_copy.filter(ImageFilter.BLUR)
    elif filter_name == "edge_enhance":
        return img_copy.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_name == "sharpen":
        return img_copy.filter(ImageFilter.SHARPEN)
    elif filter_name == "emboss":
        return img_copy.filter(ImageFilter.EMBOSS)
    elif filter_name == "contour":
        return img_copy.filter(ImageFilter.CONTOUR)
    elif filter_name == "invert":
        if img_copy.mode == 'RGBA':
            r, g, b, a = img_copy.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            inverted = ImageOps.invert(rgb_image)
            r2, g2, b2 = inverted.split()
            return Image.merge('RGBA', (r2, g2, b2, a))
        else:
            return ImageOps.invert(img_copy)
    
    # Default: return original image
    return img_copy

def adjust_brightness(image, factor=1.0):
    """Adjust image brightness"""
    if image is None:
        return None
    
    enhancer = ImageEnhance.Brightness(image.copy())
    return enhancer.enhance(factor)

def adjust_contrast(image, factor=1.0):
    """Adjust image contrast"""
    if image is None:
        return None
    
    enhancer = ImageEnhance.Contrast(image.copy())
    return enhancer.enhance(factor)

def adjust_color(image, factor=1.0):
    """Adjust image color saturation"""
    if image is None:
        return None
    
    enhancer = ImageEnhance.Color(image.copy())
    return enhancer.enhance(factor)

def adjust_sharpness(image, factor=1.0):
    """Adjust image sharpness"""
    if image is None:
        return None
    
    enhancer = ImageEnhance.Sharpness(image.copy())
    return enhancer.enhance(factor)

def add_sport_overlay(image, sport_name, position="bottom"):
    """Add sport name as text overlay to image"""
    if image is None or not sport_name:
        return image
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a nice font, falling back to default
    try:
        font = ImageFont.truetype("Arial Bold", 30)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
    
    # Calculate text size with compatibility for different Pillow versions
    try:
        # For newer versions of Pillow (>=9.0.0)
        bbox = font.getbbox(sport_name)
        text_width = bbox[2] - bbox[0]  # right - left
        text_height = bbox[3] - bbox[1]  # bottom - top
    except AttributeError:
        try:
            # For Pillow 8.x.x
            text_width, text_height = font.getsize(sport_name)
        except AttributeError:
            # For older versions of Pillow
            text_width, text_height = draw.textsize(sport_name, font=font)
    
    width, height = img_copy.size
    
    # Create semi-transparent overlay rectangle
    if position == "bottom":
        rect_bbox = [(0, height - text_height - 20), (width, height)]
        text_pos = ((width - text_width) // 2, height - text_height - 10)
    elif position == "top":
        rect_bbox = [(0, 0), (width, text_height + 20)]
        text_pos = ((width - text_width) // 2, 10)
    else:  # center
        rect_bbox = [(width//2 - text_width//2 - 10, height//2 - text_height//2 - 10),
                     (width//2 + text_width//2 + 10, height//2 + text_height//2 + 10)]
        text_pos = (width//2 - text_width//2, height//2 - text_height//2)
    
    # Create overlay
    overlay = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(rect_bbox, fill=(0, 0, 0, 128))
    
    # Convert to RGBA if not already
    if img_copy.mode != 'RGBA':
        img_copy = img_copy.convert('RGBA')
    
    # Composite the images
    img_copy = Image.alpha_composite(img_copy, overlay)
    draw = ImageDraw.Draw(img_copy)
    
    # Add text
    draw.text(text_pos, sport_name, fill=(255, 255, 255), font=font)
    
    return img_copy

def get_image_for_date_sport(date, sport, image_dict=None):
    """Get image for a specific date and sport from the uploaded images or placeholders"""
    # If we have a dictionary of uploaded images, check there first
    if image_dict and (date, sport) in image_dict:
        return image_dict[(date, sport)]
    
    # Otherwise use placeholder
    placeholder_images = get_placeholder_sports_images()
    if sport in placeholder_images:
        return placeholder_images[sport]
    
    # Ultimate fallback
    return create_text_image(f"{sport} - {date}")

def create_image_gallery(df, date_column, sport_column, image_dict=None, 
                         filters=None, max_images_per_row=3):
    """
    Create an image gallery grouped by date with sport labels.
    
    Args:
        df: DataFrame containing tournament data
        date_column: Name of date column in DataFrame
        sport_column: Name of sport column in DataFrame
        image_dict: Dictionary of uploaded images keyed by (date, sport)
        filters: Optional filters to apply to gallery images
        max_images_per_row: Maximum number of images per row
    """
    if df is None or df.empty:
        st.warning("No data available to display in the gallery.")
        return
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date
    df[date_column] = df[date_column].dt.date  # Extract date part
    dates = sorted(df[date_column].unique())
    
    # Apply default filters if none specified
    if filters is None:
        filters = {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "filter": "none",
            "sport_overlay": True,
            "overlay_position": "bottom"
        }
    
    # Display gallery by date
    for date in dates:
        st.subheader(f"📅 {date.strftime('%B %d, %Y')}")
        
        # Get sports for this date
        date_sports = df[df[date_column] == date][sport_column].unique()
        
        # Create rows of images
        for i in range(0, len(date_sports), max_images_per_row):
            row_sports = date_sports[i:i+max_images_per_row]
            cols = st.columns(len(row_sports))
            
            for j, sport in enumerate(row_sports):
                with cols[j]:
                    # Get image for this date and sport
                    img = get_image_for_date_sport(date, sport, image_dict)
                    
                    # Apply requested processing
                    img = apply_filter(img, filters["filter"])
                    img = adjust_brightness(img, filters["brightness"])
                    img = adjust_contrast(img, filters["contrast"])
                    img = adjust_color(img, filters["saturation"])
                    
                    # Add sport name overlay if requested
                    if filters["sport_overlay"]:
                        img = add_sport_overlay(img, sport, filters["overlay_position"])
                    
                    # Display image with caption
                    sport_counts = df[(df[date_column] == date) & (df[sport_column] == sport)].shape[0]
                    st.image(img, caption=f"{sport} ({sport_counts} participants)")
                    
                    # Add actions menu
                    with st.expander("📊 Event Stats"):
                        # Calculate and display stats for this event
                        event_df = df[(df[date_column] == date) & (df[sport_column] == sport)]
                        st.write(f"**Participants:** {len(event_df)}")
                        if 'Performance' in event_df.columns:
                            st.write(f"**Avg. Performance:** {event_df['Performance'].mean():.1f}")
                        if 'Satisfaction' in event_df.columns:
                            st.write(f"**Avg. Satisfaction:** {event_df['Satisfaction'].mean():.1f}")
                        if 'Gender' in event_df.columns:
                            gender_counts = event_df['Gender'].value_counts()
                            st.write("**Gender Distribution:**")
                            for gender, count in gender_counts.items():
                                st.write(f"- {gender}: {count} ({count/len(event_df):.0%})")

def create_image_uploader_section():
    """Create a section for uploading and managing tournament images"""
    st.subheader("Upload Tournament Images")
    
    # Initialize session state for storing images
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = {}
    
    # Create date and sport selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_date = st.date_input("Select Event Date")
    
    with col2:
        # Get unique sports from the dataset if available
        if 'data' in st.session_state:
            sport_options = sorted(st.session_state.data['Sport'].unique())
        else:
            sport_options = [
                "Basketball", "Volleyball", "Cricket", "Football",
                "Badminton", "Table Tennis", "Athletics", "Swimming"
            ]
        selected_sport = st.selectbox("Select Sport", options=sport_options)
    
    # Image upload section
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = load_image(uploaded_file)
        if image:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image processing options
            with st.expander("Image Adjustments"):
                filter_col, brightness_col = st.columns(2)
                
                with filter_col:
                    filter_option = st.selectbox(
                        "Apply Filter",
                        ["none", "grayscale", "sepia", "blur", "edge_enhance", 
                         "sharpen", "emboss", "contour", "invert"]
                    )
                
                with brightness_col:
                    brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
                
                contrast_col, saturation_col = st.columns(2)
                
                with contrast_col:
                    contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
                
                with saturation_col:
                    saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
                
                overlay_col, position_col = st.columns(2)
                
                with overlay_col:
                    add_overlay = st.checkbox("Add Sport Label", True)
                
                with position_col:
                    overlay_position = st.selectbox(
                        "Label Position",
                        ["bottom", "top", "center"]
                    )
            
            # Preview the processed image
            processed_image = apply_filter(image, filter_option)
            processed_image = adjust_brightness(processed_image, brightness)
            processed_image = adjust_contrast(processed_image, contrast)
            processed_image = adjust_color(processed_image, saturation)
            
            if add_overlay:
                processed_image = add_sport_overlay(processed_image, selected_sport, overlay_position)
            
            st.subheader("Processed Image Preview")
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            
            # Save button
            if st.button("Save Image"):
                # Store the processed image in session state
                img_key = (selected_date, selected_sport)
                st.session_state.uploaded_images[img_key] = processed_image
                st.success(f"Image saved for {selected_sport} on {selected_date}!")
    
    # Show currently uploaded images
    if st.session_state.uploaded_images:
        st.subheader("Uploaded Event Images")
        st.write(f"You have uploaded {len(st.session_state.uploaded_images)} images.")
        
        # Create a table of uploaded images
        image_data = []
        for (date, sport), _ in st.session_state.uploaded_images.items():
            image_data.append({"Date": date, "Sport": sport})
        
        if image_data:
            image_df = pd.DataFrame(image_data)
            st.dataframe(image_df)

# Import pandas for the gallery function
import pandas as pd
