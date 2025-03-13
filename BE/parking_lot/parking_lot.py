
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import DBSCAN

def detect_parking_slots(image_path, visualize=True):
    """
    Detect parking slots from a satellite image with clearly marked lines
    
    Args:
        image_path (str): Path to the satellite image
        visualize (bool): Whether to visualize the results
    
    Returns:
        list: List of dictionaries containing information about each parking slot
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better line detection across varying lighting conditions
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Use morphological operations to enhance lines and remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Edge detection with optimized parameters
    edges = cv2.Canny(binary, 30, 150)
    
    # Line detection using Hough Transform with more sensitive parameters
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=25
    )
    
    if lines is None or len(lines) < 3:
        print("Not enough lines detected in the image.")
        return []
    
    # Separate lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Skip very short lines
        if length < 30:
            continue
            
        # Calculate angle to determine if line is horizontal or vertical
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:
            # Horizontal line - store with y-coordinate for later processing
            y_avg = (y1 + y2) // 2
            horizontal_lines.append(((x1, y1), (x2, y2), y_avg))
        elif 60 < angle < 120:
            # Vertical line - store with x-coordinate for later processing
            x_avg = (x1 + x2) // 2
            vertical_lines.append(((x1, y1), (x2, y2), x_avg))
    
    print(f"Found {len(horizontal_lines)} horizontal lines and {len(vertical_lines)} vertical lines")
    
    # Enhanced approach - detect parking rows using clustering of vertical lines
    if len(vertical_lines) > 0:
        # Sort by y-coordinate (top to bottom)
        horizontal_lines.sort(key=lambda line: line[2])
        
        # Sort vertical lines by x-coordinate (left to right)
        vertical_lines.sort(key=lambda line: line[2])
        
        # For visualization
        result_img = image.copy()
        
        # Group vertical lines that might form the boundaries of parking rows
        vertical_x_coords = np.array([[line[2]] for line in vertical_lines])
        
        # Use DBSCAN to cluster vertical lines that are close together
        clustering = DBSCAN(eps=width*0.05, min_samples=1).fit(vertical_x_coords)
        labels = clustering.labels_
        
        # Get unique cluster centers (average x-coordinate of each cluster)
        unique_labels = set(labels)
        section_boundaries = []
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            avg_x = np.mean([vertical_lines[i][2] for i in indices])
            # Get the longest line in this cluster
            lengths = [np.sqrt((vertical_lines[i][0][0] - vertical_lines[i][1][0])**2 + 
                              (vertical_lines[i][0][1] - vertical_lines[i][1][1])**2) for i in indices]
            best_idx = indices[np.argmax(lengths)]
            section_boundaries.append((avg_x, vertical_lines[best_idx][0], vertical_lines[best_idx][1]))
        
        # Sort boundaries by x-coordinate
        section_boundaries.sort(key=lambda x: x[0])
        
        # Extract y-ranges from the image
        # Instead of clustering horizontal lines, we'll divide the image into regions
        # This ensures we detect parking slots on both sides of horizontal dividers
        
        # First, find the y-range of the parking area
        all_y_values = []
        for v_line in vertical_lines:
            all_y_values.append(v_line[0][1])
            all_y_values.append(v_line[1][1])
        
        # Get min and max y values (with some margin)
        min_y = max(0, min(all_y_values) - 10)
        max_y = min(height, max(all_y_values) + 10)
        
        # Create horizontal regions by using the horizontal lines as boundaries
        # Add image top and bottom as boundary points
        h_boundaries = [min_y]
        for h_line in horizontal_lines:
            h_boundaries.append(h_line[2])  # y-coordinate
        h_boundaries.append(max_y)
        
        # Sort boundaries from top to bottom
        h_boundaries = sorted(set(h_boundaries))
        
        # Define parking slots using sections and horizontal regions
        all_slots = []
        slot_id = 1
        
        # Create sections based on vertical boundaries (add image edges as boundaries)
        all_v_boundaries = [0] + [b[0] for b in section_boundaries] + [width]
        
        # Loop through all pairs of adjacent vertical boundaries
        for i in range(len(all_v_boundaries) - 1):
            left_bound = all_v_boundaries[i]
            right_bound = all_v_boundaries[i+1]
            
            # Skip if section is too narrow to be a parking row
            if right_bound - left_bound < width * 0.05:  # Minimum width threshold
                continue
                
            row_id = i + 1
            
            # Loop through all pairs of adjacent horizontal regions
            for j in range(len(h_boundaries) - 1):
                top_y = h_boundaries[j]
                bottom_y = h_boundaries[j+1]
                
                # Skip if space is too short to be a parking slot
                if bottom_y - top_y < height * 0.02:  # Minimum height threshold
                    continue
                
                # Calculate center and dimensions of the slot
                center_x = (left_bound + right_bound) / 2
                center_y = (top_y + bottom_y) / 2
                slot_width = right_bound - left_bound
                slot_height = bottom_y - top_y
                
                # Calculate area (in pixels)
                slot_area = slot_width * slot_height
                
                # Check if slot dimensions are reasonable for a parking space
                if slot_width > width * 0.03 and slot_height > height * 0.02:
                    slot_info = {
                        "id": slot_id,
                        "row": row_id,
                        "column": j + 1,
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": slot_width,
                        "height": slot_height,
                        "area": slot_area
                    }
                    all_slots.append(slot_info)
                    slot_id += 1
                    
                    # Draw rectangle and label on result image for visualization
                    if visualize:
                        top_left = (int(left_bound), int(top_y))
                        bottom_right = (int(right_bound), int(bottom_y))
                        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
                        
                        label = f"{slot_id-1}: {int(slot_area)}"
                        cv2.putText(result_img, label, (int(center_x - 40), int(center_y)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Visualization
        if visualize and all_slots:
            # Draw detected lines for better understanding
            for h_line in horizontal_lines:
                x1, y1 = h_line[0]
                x2, y2 = h_line[1]
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for horizontal
                
            for v_line in vertical_lines:
                x1, y1 = v_line[0]
                x2, y2 = v_line[1]
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for vertical
            
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected Parking Slots: {len(all_slots)}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        return all_slots
    
    return []

def main():
    # Path to your satellite image
    image_path = "parking_lot.jpg"
    
    # Detect parking slots
    slots = detect_parking_slots(image_path, visualize=True)
    
    # Print the area of each slot
    print("\nArea of each detected parking slot:")
    print("-" * 50)
    print(f"{'Slot ID':<10}{'Row':<8}{'Column':<8}{'Width':<10}{'Height':<10}{'Area (pixels)':<15}")
    print("-" * 50)
    
    for slot in slots:
        print(f"{slot['id']:<10}{slot['row']:<8}{slot['column']:<8}{slot['width']:<10.1f}{slot['height']:<10.1f}{slot['area']:<15.1f}")
    
    # Calculate some statistics
    if slots:
        total_area = sum(slot['area'] for slot in slots)
        avg_area = total_area / len(slots)
        min_area = min(slot['area'] for slot in slots)
        max_area = max(slot['area'] for slot in slots)
        
        print("\nSummary Statistics:")
        print(f"Total slots detected: {len(slots)}")
        print(f"Total parking area: {total_area:.1f} pixels")
        print(f"Average slot area: {avg_area:.1f} pixels")
        print(f"Minimum slot area: {min_area:.1f} pixels")
        print(f"Maximum slot area: {max_area:.1f} pixels")

if __name__ == "__main__":
    main()



