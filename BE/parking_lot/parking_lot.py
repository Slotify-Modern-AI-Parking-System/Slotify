
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_parking_slots_grid(image_path, visualize=True):
#     """
#     Detect individual parking slots from an image with a grid of blue rectangular outlines
    
#     Args:
#         image_path (str): Path to the image with blue-marked parking slots
#         visualize (bool): Whether to visualize the results
    
#     Returns:
#         list: List of dictionaries containing information about each parking slot
#     """
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     # Create a copy for visualization
#     result_img = image.copy()
    
#     # Convert to HSV color space for better blue detection
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define range for blue color - targeting the bright blue in your image
#     lower_blue = np.array([90, 100, 100])
#     upper_blue = np.array([130, 255, 255])
    
#     # Create a mask for blue pixels
#     blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
#     # Find the region of interest - the entire grid area
#     # First, find the bounding box of all blue elements
#     blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not blue_contours:
#         print("No blue contours found in the image")
#         return []
    
#     # Combine all contours to get the overall grid area
#     all_cnts = np.vstack([cnt for cnt in blue_contours])
#     x, y, w, h = cv2.boundingRect(all_cnts)
    
#     print(f"Found overall grid area: x={x}, y={y}, width={w}, height={h}")
    
#     # Draw the overall grid boundary for reference
#     grid_img = result_img.copy()
#     cv2.rectangle(grid_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
#     # Now let's approach this differently - we'll look for horizontal and vertical lines in the grid
#     # Apply edge detection to the blue mask
#     edges = cv2.Canny(blue_mask, 50, 150)
    
#     # Use Hough Line Transform to find lines in the grid
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=10)
    
#     # Separate horizontal and vertical lines
#     horizontal_lines = []
#     vertical_lines = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             # Calculate line angle
#             angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
#             # Classify as horizontal or vertical
#             if angle < 45:  # Horizontal lines
#                 horizontal_lines.append((y1, x1, x2))  # store y-coordinate and x-range
#             elif angle > 45:  # Vertical lines
#                 vertical_lines.append((x1, y1, y2))  # store x-coordinate and y-range
    
#     print(f"Found {len(horizontal_lines)} horizontal lines and {len(vertical_lines)} vertical lines")
    
#     # Alternative approach - let's use a grid-based detection
#     # Crop the blue mask to the grid area
#     grid_mask = blue_mask[y:y+h, x:x+w]
    
#     # Count how many rows and columns we have by looking at pixel distribution
#     row_projection = np.sum(grid_mask, axis=1)  # Sum each row
#     col_projection = np.sum(grid_mask, axis=0)  # Sum each column
    
#     # Find the rows and columns by looking for peaks in the projections
#     row_threshold = np.max(row_projection) * 0.3
#     col_threshold = np.max(col_projection) * 0.3
    
#     # Find row positions
#     row_positions = []
#     in_row = False
#     for i, val in enumerate(row_projection):
#         if not in_row and val > row_threshold:
#             row_start = i
#             in_row = True
#         elif in_row and val < row_threshold:
#             row_end = i
#             row_positions.append((row_start + row_end) // 2)  # Center of the row
#             in_row = False
    
#     # Find column positions
#     col_positions = []
#     in_col = False
#     for i, val in enumerate(col_projection):
#         if not in_col and val > col_threshold:
#             col_start = i
#             in_col = True
#         elif in_col and val < col_threshold:
#             col_end = i
#             col_positions.append((col_start + col_end) // 2)  # Center of the column
#             in_col = False
    
#     print(f"Detected {len(row_positions)} rows and {len(col_positions)} columns")
    
#     # If the detection failed, we'll use a fixed grid approach
#     # For your specific image, we know there are roughly 4 rows and 6 columns
#     if len(row_positions) < 2 or len(col_positions) < 2:
#         print("Using fixed grid approach...")
#         num_rows = 4
#         num_cols = 6
        
#         # Create evenly spaced grid positions
#         row_positions = [y + int(i * h / num_rows) for i in range(1, num_rows+1)]
#         col_positions = [x + int(i * w / num_cols) for i in range(1, num_cols+1)]
    
#     # Determine cell dimensions (adjust these based on your actual image)
#     cell_height = h // len(row_positions) if row_positions else h // 4
#     cell_width = w // len(col_positions) if col_positions else w // 6
    
#     # Direct detection of blue rectangles
#     # This should work better for your image
#     contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter contours by size - we want the individual rectangles
#     parking_slots = []
#     slot_id = 1
    
#     rectangle_contours = []
#     for contour in contours:
#         # Get the bounding rectangle
#         rx, ry, rw, rh = cv2.boundingRect(contour)
#         area = cv2.contourArea(contour)
        
#         # Filter by size - exclude very small or very large contours
#         # These thresholds may need adjustment based on your image
#         if area > 50 and area < 5000:
#             rect_area = rw * rh
#             # Check if the shape is roughly rectangular
#             if area / rect_area > 0.2:  # This threshold may need adjustment
#                 rectangle_contours.append(contour)
                
#                 # Add to parking slots
#                 slot_info = {
#                     "id": slot_id,
#                     "x": rx,
#                     "y": ry,
#                     "width": rw,
#                     "height": rh,
#                     "area": area
#                 }
#                 parking_slots.append(slot_info)
#                 slot_id += 1
                
#                 # Draw rectangle and label
#                 cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
#                 cv2.putText(result_img, str(slot_id-1), (rx+5, ry+20), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
#     print(f"Detected {len(parking_slots)} individual parking slots")
    
#     # If we still don't have enough slots, try a different approach
#     if len(parking_slots) < 20:  # We expect around 24 slots
#         print("Using template matching approach...")
        
#         # Let's try to find individual cells directly using contour hierarchy
#         contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Reset the parking slots list
#         parking_slots = []
#         slot_id = 1
        
#         # Extract the contour hierarchy to find the cells
#         if hierarchy is not None:
#             hierarchy = hierarchy[0]
#             for i, contour in enumerate(contours):
#                 # Skip small contours
#                 if cv2.contourArea(contour) < 50:
#                     continue
                
#                 # Get the bounding rectangle
#                 rx, ry, rw, rh = cv2.boundingRect(contour)
                
#                 # Check if this contour has a parent (which could be the grid outline)
#                 if hierarchy[i][3] != -1:  # If it has a parent
#                     # Add to parking slots
#                     slot_info = {
#                         "id": slot_id,
#                         "x": rx,
#                         "y": ry,
#                         "width": rw,
#                         "height": rh,
#                         "area": cv2.contourArea(contour)
#                     }
#                     parking_slots.append(slot_info)
#                     slot_id += 1
                    
#                     # Draw rectangle and label
#                     cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
#                     cv2.putText(result_img, str(slot_id-1), (rx+5, ry+20), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
#     # If we STILL don't have enough slots, use a grid-based approach
#     if len(parking_slots) < 20:
#         print("Using uniform grid division approach...")
#         # Reset the parking slots list
#         parking_slots = []
#         slot_id = 1
        
#         # Determine the number of rows and columns from the image
#         # For your image, it appears to be 4 rows and 6 columns
#         num_rows = 4
#         num_cols = 6
        
#         # Calculate cell dimensions
#         cell_height = h // num_rows
#         cell_width = w // num_cols
        
#         # Create a grid of cells
#         for row in range(num_rows):
#             for col in range(num_cols):
#                 rx = x + col * cell_width
#                 ry = y + row * cell_height
                
#                 # Add to parking slots
#                 slot_info = {
#                     "id": slot_id,
#                     "x": rx,
#                     "y": ry,
#                     "width": cell_width,
#                     "height": cell_height,
#                     "area": cell_width * cell_height,
#                     "row": row + 1,
#                     "column": col + 1
#                 }
#                 parking_slots.append(slot_info)
#                 slot_id += 1
                
#                 # Draw rectangle and label
#                 cv2.rectangle(result_img, (rx, ry), (rx+cell_width, ry+cell_height), (0, 255, 0), 2)
#                 cv2.putText(result_img, str(slot_id-1), (rx+5, ry+20), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(15, 10))
        
#         plt.subplot(2, 2, 1)
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title("Original Image")
#         plt.axis('off')
        
#         plt.subplot(2, 2, 2)
#         plt.imshow(blue_mask, cmap='gray')
#         plt.title("Blue Mask")
#         plt.axis('off')
        
#         plt.subplot(2, 2, 3)
#         plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
#         plt.title("Grid Boundary")
#         plt.axis('off')
        
#         plt.subplot(2, 2, 4)
#         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#         plt.title(f"Detected Parking Slots: {len(parking_slots)}")
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.show()
    
#     return parking_slots

# def main():
#     # Path to your image with blue-marked parking slots
#     image_path = "Screenshot (4).png"  # Change this to your image path
    
#     # Detect parking slots
#     slots = detect_parking_slots_grid(image_path, visualize=True)
    
#     # Print information about each detected slot
#     print(f"\nDetected {len(slots)} parking slots")
    
#     if slots:
#         print("\nSlot Details:")
#         print("-" * 50)
#         print(f"{'ID':<5}{'Position (x,y)':<20}{'Size (w×h)':<15}{'Area':<10}")
#         print("-" * 50)
        
#         for slot in slots:
#             print(f"{slot['id']:<5}({slot['x']}, {slot['y']})  {' ':<5}{slot['width']}×{slot['height']}{' ':<5}{slot['area']:<10.1f}")
        
#         print(f"\nTotal parking slots: {len(slots)}")
#     else:
#         print("No parking slots detected. Try adjusting parameters.")

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_individual_parking_slots(image_path, visualize=True):
#     """
#     Detect individual parking slots within colored sections:
#     - Red: Entry
#     - Purple: Accessible
#     - Yellow: Reservation
#     - Blue: Regular Parking Slots
    
#     Args:
#         image_path (str): Path to the image with colored parking slots
#         visualize (bool): Whether to visualize the results
    
#     Returns:
#         dict: Dictionary containing individual slots by category with coordinates
#     """
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     # Create a copy for visualization
#     result_img = image.copy()
    
#     # Convert to HSV and grayscale
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Define color ranges for each parking type
#     color_ranges = {
#         'Entry': {
#             'lower': np.array([0, 100, 100]),
#             'upper': np.array([10, 255, 255]),
#             'color': (0, 0, 255),
#             'prefix': 'E'
#         },
#         'Accessible': {
#             'lower': np.array([120, 100, 100]),
#             'upper': np.array([160, 255, 255]),
#             'color': (128, 0, 128),
#             'prefix': 'A'
#         },
#         'Reservation': {
#             'lower': np.array([20, 100, 100]),
#             'upper': np.array([30, 255, 255]),
#             'color': (0, 255, 255),
#             'prefix': 'R'
#         },
#         'Regular': {
#             'lower': np.array([90, 100, 100]),
#             'upper': np.array([130, 255, 255]),
#             'color': (255, 0, 0),
#             'prefix': 'P'
#         }
#     }
    
#     # Additional red range for wrap-around hue
#     red_range_2 = {
#         'lower': np.array([170, 100, 100]),
#         'upper': np.array([180, 255, 255])
#     }
    
#     # Dictionary to store all individual parking slots by category
#     parking_slots = {
#         'Entry': [],
#         'Accessible': [],
#         'Reservation': [],
#         'Regular': []
#     }
    
#     # Process each color category
#     for category, color_info in color_ranges.items():
#         print(f"Processing {category} parking slots...")
        
#         # Create mask for this color
#         if category == 'Entry':
#             mask1 = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
#             mask2 = cv2.inRange(hsv, red_range_2['lower'], red_range_2['upper'])
#             mask = cv2.bitwise_or(mask1, mask2)
#         else:
#             mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
        
#         # Find contours for colored regions
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         slot_id = 1
#         for contour in contours:
#             area = cv2.contourArea(contour)
            
#             # Only process significant colored areas
#             if area > 500:  # Minimum area threshold
#                 # Get bounding rectangle of the colored region
#                 x, y, w, h = cv2.boundingRect(contour)
                
#                 # Extract the region of interest
#                 roi = gray[y:y+h, x:x+w]
#                 roi_color = image[y:y+h, x:x+w]
                
#                 # Detect individual parking slots within this colored region
#                 individual_slots = detect_slots_in_region(roi, roi_color, x, y, w, h)
                
#                 # Add slots to the category
#                 for slot in individual_slots:
#                     slot_info = {
#                         "id": slot_id,
#                         "label": f"{color_info['prefix']}{slot_id}",
#                         "x": slot['x'],
#                         "y": slot['y'],
#                         "width": slot['width'],
#                         "height": slot['height'],
#                         "center_x": slot['x'] + slot['width'] // 2,
#                         "center_y": slot['y'] + slot['height'] // 2,
#                         "area": slot['area'],
#                         "category": category
#                     }
#                     parking_slots[category].append(slot_info)
                    
#                     # Draw rectangle and label on result image
#                     cv2.rectangle(result_img, (slot['x'], slot['y']), 
#                                 (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                                 color_info['color'], 2)
#                     cv2.putText(result_img, slot_info['label'], 
#                               (slot['x'] + 5, slot['y'] + 20), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_info['color'], 2)
#                     slot_id += 1
        
#         print(f"Found {len(parking_slots[category])} individual {category} slots")
    
#     # Create summary statistics
#     summary = {
#         'total_slots': sum(len(slots) for slots in parking_slots.values()),
#         'counts': {category: len(slots) for category, slots in parking_slots.items()},
#         'slots_by_category': parking_slots
#     }
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(15, 10))
        
#         # Original image
#         plt.subplot(2, 2, 1)
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title("Original Image")
#         plt.axis('off')
        
#         # Final result
#         plt.subplot(2, 2, 2)
#         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#         plt.title(f"Individual Slots Detected (Total: {summary['total_slots']})")
#         plt.axis('off')
        
#         # Show individual category results
#         for i, (category, slots) in enumerate(parking_slots.items()):
#             if slots and i < 2:  # Show first 2 categories with slots
#                 plt.subplot(2, 2, i + 3)
#                 category_img = image.copy()
#                 for slot in slots:
#                     cv2.rectangle(category_img, (slot['x'], slot['y']), 
#                                 (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                                 color_ranges[category]['color'], 2)
#                     cv2.putText(category_img, slot['label'], 
#                               (slot['x'] + 5, slot['y'] + 20), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ranges[category]['color'], 2)
#                 plt.imshow(cv2.cvtColor(category_img, cv2.COLOR_BGR2RGB))
#                 plt.title(f"{category} Slots ({len(slots)})")
#                 plt.axis('off')
        
#         plt.tight_layout()
#         plt.show()
    
#     return summary

# def detect_slots_in_region(roi_gray, roi_color, offset_x, offset_y, region_width, region_height):
#     """
#     Detect individual parking slots within a colored region using various methods
#     """
#     slots = []
    
#     # Method 1: Try to detect rectangular structures using contours
#     # Apply edge detection
#     edges = cv2.Canny(roi_gray, 50, 150)
    
#     # Find contours in the edge image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     detected_any = False
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if 200 < area < 5000:  # Reasonable parking slot size
#             x, y, w, h = cv2.boundingRect(contour)
            
#             # Check if it's roughly rectangular (aspect ratio check)
#             aspect_ratio = w / h if h > 0 else 0
#             if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for parking slots
#                 slots.append({
#                     'x': offset_x + x,
#                     'y': offset_y + y,
#                     'width': w,
#                     'height': h,
#                     'area': area
#                 })
#                 detected_any = True
    
#     # Method 2: If no good slots detected, use grid-based approach
#     if not detected_any or len(slots) < 2:
#         slots = []
        
#         # Estimate number of slots based on region size
#         avg_slot_width = 80  # Average parking slot width
#         avg_slot_height = 60  # Average parking slot height
        
#         # Calculate possible grid dimensions
#         cols = max(1, region_width // avg_slot_width)
#         rows = max(1, region_height // avg_slot_height)
        
#         # If the region is too small for multiple slots, treat as single slot
#         if cols == 1 and rows == 1:
#             slots.append({
#                 'x': offset_x,
#                 'y': offset_y,
#                 'width': region_width,
#                 'height': region_height,
#                 'area': region_width * region_height
#             })
#         else:
#             # Create grid-based slots
#             slot_width = region_width // cols
#             slot_height = region_height // rows
            
#             for row in range(rows):
#                 for col in range(cols):
#                     x = offset_x + col * slot_width
#                     y = offset_y + row * slot_height
                    
#                     slots.append({
#                         'x': x,
#                         'y': y,
#                         'width': slot_width,
#                         'height': slot_height,
#                         'area': slot_width * slot_height
#                     })
    
#     return slots

# def main():
#     # Path to your image with colored parking slots
#     image_path = "Modified_Parking_Lot.png"  # Change this to your image path
    
#     try:
#         # Detect individual parking slots
#         result = detect_individual_parking_slots(image_path, visualize=True)
        
#         # Print summary
#         print(f"\n{'='*60}")
#         print("INDIVIDUAL PARKING SLOT DETECTION SUMMARY")
#         print(f"{'='*60}")
        
#         print(f"Total parking slots detected: {result['total_slots']}")
#         print(f"\nBreakdown by category:")
#         print(f"{'Category':<15} {'Count':<10}")
#         print(f"{'-'*30}")
        
#         for category, count in result['counts'].items():
#             print(f"{category:<15} {count:<10}")
        
#         # Print detailed slot information with coordinates
#         print(f"\n{'='*70}")
#         print("DETAILED INDIVIDUAL SLOT COORDINATES")
#         print(f"{'='*70}")
        
#         for category, slots in result['slots_by_category'].items():
#             if slots:
#                 print(f"\n{category.upper()} PARKING SLOTS ({len(slots)} slots):")
#                 print(f"{'Label':<8} {'Top-Left (x,y)':<15} {'Center (x,y)':<15} {'Size (w×h)':<12} {'Area':<8}")
#                 print(f"{'-'*70}")
                
#                 for slot in slots:
#                     print(f"{slot['label']:<8} "
#                           f"({slot['x']},{slot['y']})      "
#                           f"({slot['center_x']},{slot['center_y']})      "
#                           f"{slot['width']}×{slot['height']}     "
#                           f"{slot['area']:<8.0f}")
        
#         # Export coordinates to a simple format
#         print(f"\n{'='*60}")
#         print("COORDINATES EXPORT FORMAT")
#         print(f"{'='*60}")
        
#         for category, slots in result['slots_by_category'].items():
#             if slots:
#                 print(f"\n{category}:")
#                 for slot in slots:
#                     print(f"  {slot['label']}: ({slot['center_x']}, {slot['center_y']})")
        
#         return result
        
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# if __name__ == "__main__":
#     main()



import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_parking_slots_all_colors(image_path, visualize=True):
    """
    Detect individual parking slots from an image with different colored rectangular outlines
    - Red: Entry slots
    - Purple: Accessible slots  
    - Yellow: Reservation slots
    - Blue: Regular parking slots
    
    Args:
        image_path (str): Path to the image with colored parking slots
        visualize (bool): Whether to visualize the results
    
    Returns:
        dict: Dictionary containing information about each parking slot by color category
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create a copy for visualization
    result_img = image.copy()
    
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for each parking type
    color_configs = {
        'Entry': {
            'lower': [np.array([0, 100, 100]), np.array([170, 100, 100])],  # Two ranges for red
            'upper': [np.array([10, 255, 255]), np.array([180, 255, 255])],
            'color': (0, 0, 255),  # Red for visualization
            'prefix': 'E'
        },
        'Accessible': {
            'lower': [np.array([120, 100, 100])],  # Purple/Magenta
            'upper': [np.array([160, 255, 255])],
            'color': (128, 0, 128),  # Purple for visualization
            'prefix': 'A'
        },
        'Reservation': {
            'lower': [np.array([20, 100, 100])],  # Yellow
            'upper': [np.array([30, 255, 255])],
            'color': (0, 255, 255),  # Yellow for visualization
            'prefix': 'R'
        },
        'Regular': {
            'lower': [np.array([90, 100, 100])],  # Blue
            'upper': [np.array([130, 255, 255])],
            'color': (255, 0, 0),  # Blue for visualization
            'prefix': 'P'
        }
    }
    
    # Dictionary to store all parking slots by category
    all_parking_slots = {
        'Entry': [],
        'Accessible': [],
        'Reservation': [],
        'Regular': []
    }
    
    # Process each color category
    for category, config in color_configs.items():
        print(f"Processing {category} parking slots...")
        
        # Create mask for this color
        masks = []
        for i, lower in enumerate(config['lower']):
            upper = config['upper'][i]
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        # Combine masks if multiple ranges (like for red)
        if len(masks) > 1:
            combined_mask = cv2.bitwise_or(masks[0], masks[1])
        else:
            combined_mask = masks[0]
        
        # Detect individual parking slots for this color
        slots = detect_slots_for_color(combined_mask, category, config, result_img)
        all_parking_slots[category] = slots
        
        print(f"Detected {len(slots)} {category} parking slots")
    
    # Calculate totals
    total_slots = sum(len(slots) for slots in all_parking_slots.values())
    
    # Visualization
    if visualize:
        plt.figure(figsize=(20, 15))
        
        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Final result with all slots
        plt.subplot(3, 3, 2)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"All Detected Slots (Total: {total_slots})")
        plt.axis('off')
        
        # Individual color masks and results
        subplot_idx = 3
        for category, config in color_configs.items():
            if subplot_idx > 9:
                break
                
            # Create mask for visualization
            masks = []
            for i, lower in enumerate(config['lower']):
                upper = config['upper'][i]
                mask = cv2.inRange(hsv, lower, upper)
                masks.append(mask)
            
            if len(masks) > 1:
                combined_mask = cv2.bitwise_or(masks[0], masks[1])
            else:
                combined_mask = masks[0]
            
            # Show mask
            plt.subplot(3, 3, subplot_idx)
            plt.imshow(combined_mask, cmap='gray')
            plt.title(f"{category} Mask")
            plt.axis('off')
            subplot_idx += 1
            
            # Show detected slots for this category
            if subplot_idx <= 9:
                category_img = image.copy()
                slots = all_parking_slots[category]
                for slot in slots:
                    cv2.rectangle(category_img, (slot['x'], slot['y']), 
                                (slot['x'] + slot['width'], slot['y'] + slot['height']), 
                                config['color'], 2)
                    cv2.putText(category_img, slot['label'], 
                              (slot['x'] + 5, slot['y'] + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
                
                plt.subplot(3, 3, subplot_idx)
                plt.imshow(cv2.cvtColor(category_img, cv2.COLOR_BGR2RGB))
                plt.title(f"{category} Slots ({len(slots)})")
                plt.axis('off')
                subplot_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    return all_parking_slots

def detect_slots_for_color(color_mask, category, config, result_img):
    """
    Detect individual parking slots for a specific color using the same approach as blue slots
    """
    parking_slots = []
    
    # Find the region of interest - the entire colored area
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No {category} contours found in the image")
        return parking_slots
    
    # Combine all contours to get the overall area
    all_cnts = np.vstack([cnt for cnt in contours if cv2.contourArea(cnt) > 50])
    if len(all_cnts) == 0:
        return parking_slots
        
    x, y, w, h = cv2.boundingRect(all_cnts)
    print(f"Found overall {category} area: x={x}, y={y}, width={w}, height={h}")
    
    # Method 1: Direct detection of colored rectangles
    contours, _ = cv2.findContours(color_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    slot_id = 1
    rectangle_contours = []
    
    for contour in contours:
        # Get the bounding rectangle
        rx, ry, rw, rh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter by size - exclude very small or very large contours
        if area > 50 and area < 5000:
            rect_area = rw * rh
            # Check if the shape is roughly rectangular
            if area / rect_area > 0.2:
                rectangle_contours.append(contour)
                
                # Add to parking slots
                slot_info = {
                    "id": slot_id,
                    "label": f"{config['prefix']}{slot_id}",
                    "x": rx,
                    "y": ry,
                    "width": rw,
                    "height": rh,
                    "center_x": rx + rw // 2,
                    "center_y": ry + rh // 2,
                    "area": area,
                    "category": category
                }
                parking_slots.append(slot_info)
                slot_id += 1
                
                # Draw rectangle and label on result image
                cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), config['color'], 2)
                cv2.putText(result_img, slot_info['label'], (rx+5, ry+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
    
    # Method 2: If we don't have many slots, try contour hierarchy approach
    if len(parking_slots) < 5:  # Threshold can be adjusted
        print(f"Using hierarchy approach for {category}...")
        
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset if we're trying a different approach
        additional_slots = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 50:
                    continue
                
                rx, ry, rw, rh = cv2.boundingRect(contour)
                
                # Check if this contour has a parent or is a reasonable size
                if hierarchy[i][3] != -1 or cv2.contourArea(contour) > 200:
                    # Avoid duplicates by checking if we already have a slot in this area
                    duplicate = False
                    for existing_slot in parking_slots:
                        if (abs(existing_slot['x'] - rx) < 20 and 
                            abs(existing_slot['y'] - ry) < 20):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        slot_info = {
                            "id": slot_id,
                            "label": f"{config['prefix']}{slot_id}",
                            "x": rx,
                            "y": ry,
                            "width": rw,
                            "height": rh,
                            "center_x": rx + rw // 2,
                            "center_y": ry + rh // 2,
                            "area": cv2.contourArea(contour),
                            "category": category
                        }
                        additional_slots.append(slot_info)
                        slot_id += 1
                        
                        # Draw rectangle and label
                        cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), config['color'], 2)
                        cv2.putText(result_img, slot_info['label'], (rx+5, ry+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
        
        # Add additional slots if they provide more coverage
        if len(additional_slots) > len(parking_slots):
            parking_slots = additional_slots
    
    # Method 3: Grid-based approach if still not enough slots
    if len(parking_slots) < 3 and w > 100 and h > 100:  # Only if the area is significant
        print(f"Using grid approach for {category}...")
        
        # Reset the parking slots list
        parking_slots = []
        slot_id = 1
        
        # Estimate grid dimensions based on area size
        avg_slot_width = 80
        avg_slot_height = 60
        
        num_cols = max(1, w // avg_slot_width)
        num_rows = max(1, h // avg_slot_height)
        
        # Calculate cell dimensions
        cell_width = w // num_cols
        cell_height = h // num_rows
        
        # Create a grid of cells
        for row in range(num_rows):
            for col in range(num_cols):
                rx = x + col * cell_width
                ry = y + row * cell_height
                
                # Check if this area actually contains colored pixels
                roi_mask = color_mask[ry:ry+cell_height, rx:rx+cell_width]
                if np.sum(roi_mask) > cell_width * cell_height * 0.1:  # At least 10% colored
                    slot_info = {
                        "id": slot_id,
                        "label": f"{config['prefix']}{slot_id}",
                        "x": rx,
                        "y": ry,
                        "width": cell_width,
                        "height": cell_height,
                        "center_x": rx + cell_width // 2,
                        "center_y": ry + cell_height // 2,
                        "area": cell_width * cell_height,
                        "category": category,
                        "row": row + 1,
                        "column": col + 1
                    }
                    parking_slots.append(slot_info)
                    slot_id += 1
                    
                    # Draw rectangle and label
                    cv2.rectangle(result_img, (rx, ry), (rx+cell_width, ry+cell_height), config['color'], 2)
                    cv2.putText(result_img, slot_info['label'], (rx+5, ry+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
    
    return parking_slots

def main():
    # Path to your image with colored parking slots
    image_path = "Modified_Parking_Lot.png"  # Change this to your image path
    
    try:
        # Detect parking slots for all colors
        all_slots = detect_parking_slots_all_colors(image_path, visualize=True)
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PARKING SLOT DETECTION SUMMARY")
        print(f"{'='*80}")
        
        total_count = sum(len(slots) for slots in all_slots.values())
        print(f"Total parking slots detected: {total_count}")
        
        print(f"\nBreakdown by category:")
        print(f"{'Category':<15} {'Count':<10} {'Prefix':<10}")
        print(f"{'-'*40}")
        
        for category, slots in all_slots.items():
            count = len(slots)
            prefix = slots[0]['label'][0] if slots else 'N/A'
            print(f"{category:<15} {count:<10} {prefix:<10}")
        
        # Print detailed information for each category
        print(f"\n{'='*100}")
        print("DETAILED SLOT INFORMATION BY CATEGORY")
        print(f"{'='*100}")
        
        for category, slots in all_slots.items():
            if slots:
                print(f"\n{category.upper()} PARKING SLOTS ({len(slots)} slots):")
                print(f"{'Label':<8} {'Position (x,y)':<15} {'Center (x,y)':<15} {'Size (w×h)':<12} {'Area':<8}")
                print(f"{'-'*70}")
                
                for slot in slots:
                    print(f"{slot['label']:<8} "
                          f"({slot['x']},{slot['y']}) {' ':<5} "
                          f"({slot['center_x']},{slot['center_y']}) {' ':<3} "
                          f"{slot['width']}×{slot['height']} {' ':<3} "
                          f"{slot['area']:<8.0f}")
        
        # Export coordinates in a simple format
        print(f"\n{'='*60}")
        print("COORDINATES EXPORT FORMAT")
        print(f"{'='*60}")
        
        for category, slots in all_slots.items():
            if slots:
                print(f"\n{category}:")
                for slot in slots:
                    print(f"  {slot['label']}: Center({slot['center_x']}, {slot['center_y']}) - "
                          f"TopLeft({slot['x']}, {slot['y']}) - Size({slot['width']}×{slot['height']})")
        
        return all_slots
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()


