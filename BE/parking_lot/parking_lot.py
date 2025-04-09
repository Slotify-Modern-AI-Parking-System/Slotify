
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_parking_slots_grid(image_path, visualize=True):
    """
    Detect individual parking slots from an image with a grid of blue rectangular outlines
    
    Args:
        image_path (str): Path to the image with blue-marked parking slots
        visualize (bool): Whether to visualize the results
    
    Returns:
        list: List of dictionaries containing information about each parking slot
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create a copy for visualization
    result_img = image.copy()
    
    # Convert to HSV color space for better blue detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color - targeting the bright blue in your image
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask for blue pixels
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find the region of interest - the entire grid area
    # First, find the bounding box of all blue elements
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not blue_contours:
        print("No blue contours found in the image")
        return []
    
    # Combine all contours to get the overall grid area
    all_cnts = np.vstack([cnt for cnt in blue_contours])
    x, y, w, h = cv2.boundingRect(all_cnts)
    
    print(f"Found overall grid area: x={x}, y={y}, width={w}, height={h}")
    
    # Draw the overall grid boundary for reference
    grid_img = result_img.copy()
    cv2.rectangle(grid_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Now let's approach this differently - we'll look for horizontal and vertical lines in the grid
    # Apply edge detection to the blue mask
    edges = cv2.Canny(blue_mask, 50, 150)
    
    # Use Hough Line Transform to find lines in the grid
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=10)
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify as horizontal or vertical
            if angle < 45:  # Horizontal lines
                horizontal_lines.append((y1, x1, x2))  # store y-coordinate and x-range
            elif angle > 45:  # Vertical lines
                vertical_lines.append((x1, y1, y2))  # store x-coordinate and y-range
    
    print(f"Found {len(horizontal_lines)} horizontal lines and {len(vertical_lines)} vertical lines")
    
    # Alternative approach - let's use a grid-based detection
    # Since we know there's a grid pattern of parking slots
    
    # Crop the blue mask to the grid area
    grid_mask = blue_mask[y:y+h, x:x+w]
    
    # Count how many rows and columns we have by looking at pixel distribution
    row_projection = np.sum(grid_mask, axis=1)  # Sum each row
    col_projection = np.sum(grid_mask, axis=0)  # Sum each column
    
    # Find the rows and columns by looking for peaks in the projections
    # We'll use a simple thresholding approach
    row_threshold = np.max(row_projection) * 0.3
    col_threshold = np.max(col_projection) * 0.3
    
    # Find row positions
    row_positions = []
    in_row = False
    for i, val in enumerate(row_projection):
        if not in_row and val > row_threshold:
            row_start = i
            in_row = True
        elif in_row and val < row_threshold:
            row_end = i
            row_positions.append((row_start + row_end) // 2)  # Center of the row
            in_row = False
    
    # Find column positions
    col_positions = []
    in_col = False
    for i, val in enumerate(col_projection):
        if not in_col and val > col_threshold:
            col_start = i
            in_col = True
        elif in_col and val < col_threshold:
            col_end = i
            col_positions.append((col_start + col_end) // 2)  # Center of the column
            in_col = False
    
    print(f"Detected {len(row_positions)} rows and {len(col_positions)} columns")
    
    # If the detection failed, we'll use a fixed grid approach
    # For your specific image, we know there are roughly 4 rows and 6 columns
    if len(row_positions) < 2 or len(col_positions) < 2:
        print("Using fixed grid approach...")
        num_rows = 4
        num_cols = 6
        
        # Create evenly spaced grid positions
        row_positions = [y + int(i * h / num_rows) for i in range(1, num_rows+1)]
        col_positions = [x + int(i * w / num_cols) for i in range(1, num_cols+1)]
    
    # Determine cell dimensions (adjust these based on your actual image)
    cell_height = h // len(row_positions) if row_positions else h // 4
    cell_width = w // len(col_positions) if col_positions else w // 6
    
    # Direct detection of blue rectangles
    # This should work better for your image
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size - we want the individual rectangles
    parking_slots = []
    slot_id = 1
    
    rectangle_contours = []
    for contour in contours:
        # Get the bounding rectangle
        rx, ry, rw, rh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter by size - exclude very small or very large contours
        # These thresholds may need adjustment based on your image
        if area > 50 and area < 5000:
            rect_area = rw * rh
            # Check if the shape is roughly rectangular
            if area / rect_area > 0.2:  # This threshold may need adjustment
                rectangle_contours.append(contour)
                
                # Add to parking slots
                slot_info = {
                    "id": slot_id,
                    "x": rx,
                    "y": ry,
                    "width": rw,
                    "height": rh,
                    "area": area
                }
                parking_slots.append(slot_info)
                slot_id += 1
                
                # Draw rectangle and label
                cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                cv2.putText(result_img, str(slot_id-1), (rx+5, ry+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    print(f"Detected {len(parking_slots)} individual parking slots")
    
    # If we still don't have enough slots, try a different approach
    if len(parking_slots) < 20:  # We expect around 24 slots
        print("Using template matching approach...")
        
        # Let's try to find individual cells directly using contour hierarchy
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset the parking slots list
        parking_slots = []
        slot_id = 1
        
        # Extract the contour hierarchy to find the cells
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, contour in enumerate(contours):
                # Skip small contours
                if cv2.contourArea(contour) < 50:
                    continue
                
                # Get the bounding rectangle
                rx, ry, rw, rh = cv2.boundingRect(contour)
                
                # Check if this contour has a parent (which could be the grid outline)
                if hierarchy[i][3] != -1:  # If it has a parent
                    # Add to parking slots
                    slot_info = {
                        "id": slot_id,
                        "x": rx,
                        "y": ry,
                        "width": rw,
                        "height": rh,
                        "area": cv2.contourArea(contour)
                    }
                    parking_slots.append(slot_info)
                    slot_id += 1
                    
                    # Draw rectangle and label
                    cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                    cv2.putText(result_img, str(slot_id-1), (rx+5, ry+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # If we STILL don't have enough slots, use a grid-based approach
    if len(parking_slots) < 20:
        print("Using uniform grid division approach...")
        # Reset the parking slots list
        parking_slots = []
        slot_id = 1
        
        # Determine the number of rows and columns from the image
        # For your image, it appears to be 4 rows and 6 columns
        num_rows = 4
        num_cols = 6
        
        # Calculate cell dimensions
        cell_height = h // num_rows
        cell_width = w // num_cols
        
        # Create a grid of cells
        for row in range(num_rows):
            for col in range(num_cols):
                rx = x + col * cell_width
                ry = y + row * cell_height
                
                # Add to parking slots
                slot_info = {
                    "id": slot_id,
                    "x": rx,
                    "y": ry,
                    "width": cell_width,
                    "height": cell_height,
                    "area": cell_width * cell_height,
                    "row": row + 1,
                    "column": col + 1
                }
                parking_slots.append(slot_info)
                slot_id += 1
                
                # Draw rectangle and label
                cv2.rectangle(result_img, (rx, ry), (rx+cell_width, ry+cell_height), (0, 255, 0), 2)
                cv2.putText(result_img, str(slot_id-1), (rx+5, ry+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Visualization
    if visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(blue_mask, cmap='gray')
        plt.title("Blue Mask")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        plt.title("Grid Boundary")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Parking Slots: {len(parking_slots)}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return parking_slots

def main():
    # Path to your image with blue-marked parking slots
    image_path = "Screenshot (4).png"  # Change this to your image path
    
    # Detect parking slots
    slots = detect_parking_slots_grid(image_path, visualize=True)
    
    # Print information about each detected slot
    print(f"\nDetected {len(slots)} parking slots")
    
    if slots:
        print("\nSlot Details:")
        print("-" * 50)
        print(f"{'ID':<5}{'Position (x,y)':<20}{'Size (w×h)':<15}{'Area':<10}")
        print("-" * 50)
        
        for slot in slots:
            print(f"{slot['id']:<5}({slot['x']}, {slot['y']})  {' ':<5}{slot['width']}×{slot['height']}{' ':<5}{slot['area']:<10.1f}")
        
        print(f"\nTotal parking slots: {len(slots)}")
    else:
        print("No parking slots detected. Try adjusting parameters.")

if __name__ == "__main__":
    main()