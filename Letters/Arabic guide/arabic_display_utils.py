"""
Arabic Text Display Utilities
Handles Right-to-Left (RTL) text rendering for Arabic
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import os

def format_arabic_text(text):
    """
    Format Arabic text for proper RTL display
    
    Args:
        text: Arabic text string
        
    Returns:
        Formatted text ready for display
    """
    if not text:
        return ""
    
    # Reshape Arabic text
    reshaped = arabic_reshaper.reshape(text)
    # Apply bidirectional algorithm for RTL
    bidi_text = get_display(reshaped)
    return bidi_text

def get_arabic_font(font_size=40):
    """
    Get Arabic font (tries multiple options)
    
    Args:
        font_size: Font size in pixels
        
    Returns:
        PIL Font object
    """
    # Try different Arabic font paths
    font_paths = [
        "fonts/NotoSansArabic-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                continue
    
    # Fallback to default font
    return ImageFont.load_default()

def draw_arabic_text_on_frame(frame, text, position, font_size=40, color=(255, 255, 255)):
    """
    Draw Arabic text on OpenCV frame with proper RTL support
    
    Args:
        frame: OpenCV frame (BGR format)
        text: Arabic text to display
        position: (x, y) tuple for text position
        font_size: Font size
        color: Text color (B, G, R)
        
    Returns:
        Frame with Arabic text drawn
    """
    # Format Arabic text
    formatted_text = format_arabic_text(text)
    
    # Convert OpenCV frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Get Arabic font
    font = get_arabic_font(font_size)
    
    # Convert BGR color to RGB for PIL
    rgb_color = (color[2], color[1], color[0])
    
    # Draw text
    draw.text(position, formatted_text, fill=rgb_color, font=font)
    
    # Convert back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame

def create_arabic_sentence_bar(frame, sentence, bar_height=100):
    """
    Create a sentence display bar with Arabic text (RTL)
    
    Args:
        frame: OpenCV frame
        sentence: Arabic sentence to display
        bar_height: Height of the bar
        
    Returns:
        Frame with sentence bar
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Create dark bar at bottom
    cv2.rectangle(frame, (0, frame_height - bar_height), 
                 (frame_width, frame_height), (20, 20, 30), -1)
    cv2.rectangle(frame, (0, frame_height - bar_height), 
                 (frame_width, frame_height - bar_height + 3), (100, 150, 255), -1)
    
    # Display Arabic sentence (centered, RTL)
    if sentence:
        # Format for RTL display
        formatted_sentence = format_arabic_text(sentence)
        
        # Draw using PIL for better Arabic support
        frame = draw_arabic_text_on_frame(
            frame, 
            formatted_sentence,
            (frame_width // 2 - 200, frame_height - 60),
            font_size=50,
            color=(255, 255, 255)
        )
    else:
        # Show placeholder
        cv2.putText(frame, "Show your hand sign...", 
                   (frame_width // 2 - 150, frame_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    
    return frame

# Example usage
if __name__ == "__main__":
    # Test Arabic text formatting
    test_text = "مرحبا"
    formatted = format_arabic_text(test_text)
    print(f"Original: {test_text}")
    print(f"Formatted: {formatted}")
    
    # Test on frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame = create_arabic_sentence_bar(test_frame, "مرحبا بك")
    cv2.imshow("Arabic Text Test", test_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







