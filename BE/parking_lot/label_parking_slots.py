# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import argparse
# import glob
# from parking_lot import detect_parking_slots_all_colors

# def label_parking_slots_sequential_improved(slots, image_path, visualize=True):
#     """
#     Assign sequential labels (P1, P2, P3...) to detected parking slots
#     with improved visualization
    
#     Args:
#         slots (list): List of dictionaries containing slot information
#         image_path (str): Path to the original image
#         visualize (bool): Whether to visualize the results
    
#     Returns:
#         numpy.ndarray: Image with labeled parking slots
#     """
#     if not slots:
#         print("No slots to label")
#         return None
    
#     # Load the image for visualization
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     # Create two copies for different visualization options
#     result_img = image.copy()
#     enlarged_img = cv2.resize(image.copy(), (image.shape[1]*2, image.shape[0]*2))
#     schematic_img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255  # White background
    
#     # Group slots by type
#     slot_types = {
#         'Regular': [],
#         'Entry': [],
#         'Reservation': [],
#         'Accessible': []
#     }
    
#     # Categorize slots by type (using the exact keys from your detection function)
#     for slot in slots:
#         slot_type = slot.get('type', 'Regular')
#         if slot_type in slot_types:
#             slot_types[slot_type].append(slot)
#         else:
#             slot_types['Regular'].append(slot)  # Default to Regular if type unknown
    
#     # Sort each type by coordinates (left-to-right, top-to-bottom)
#     for slot_type in slot_types:
#         slot_types[slot_type].sort(key=lambda slot: (slot['y'], slot['x']))
    
#     # Label prefixes for each type
#     prefixes = {
#         'Regular': 'P',
#         'Entry': 'E',
#         'Reservation': 'R',
#         'Accessible': 'A'
#     }
    
#     # Assign sequential labels by type
#     labeled_slots = []
#     for slot_type, slot_list in slot_types.items():
#         prefix = prefixes[slot_type]
#         for i, slot in enumerate(slot_list):
#             # Create a copy of the slot with the sequential label added
#             labeled_slot = slot.copy()
#             labeled_slot['label'] = f"{prefix}{i + 1}"
#             labeled_slots.append(labeled_slot)
            
#             # Method 1: Original image with smaller font
#             text_pos = (slot['x'] + 5, slot['y'] + 20)
#             cv2.putText(result_img, labeled_slot['label'], text_pos,
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#             cv2.rectangle(result_img, 
#                          (slot['x'], slot['y']), 
#                          (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                          (0, 255, 0), 1)
            
#             # Method 2: Enlarged image
#             cv2.putText(enlarged_img, labeled_slot['label'], 
#                         (slot['x']*2 + 10, slot['y']*2 + 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#             cv2.rectangle(enlarged_img, 
#                          (slot['x']*2, slot['y']*2), 
#                          ((slot['x'] + slot['width'])*2, (slot['y'] + slot['height'])*2), 
#                          (0, 255, 0), 2)
            
#             # Method 3: Schematic view with normalized spacing
#             # Scale factor to normalize the parking lot layout
#             scale_x = 1800 / image.shape[1]
#             scale_y = 1300 / image.shape[0]
            
#             sch_x = int(slot['x'] * scale_x) + 100
#             sch_y = int(slot['y'] * scale_y) + 100
#             sch_w = max(int(slot['width'] * scale_x), 40)  # Minimum width
#             sch_h = max(int(slot['height'] * scale_y), 40)  # Minimum height
            
#             cv2.rectangle(schematic_img, 
#                          (sch_x, sch_y), 
#                          (sch_x + sch_w, sch_y + sch_h), 
#                          (0, 0, 0), 2)
#             cv2.putText(schematic_img, labeled_slot['label'], 
#                         (sch_x + sch_w//4, sch_y + sch_h//2),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     # Create a lookup table/index for easier verification
#     lookup_img = np.ones((800, 800, 3), dtype=np.uint8) * 255
#     columns = 5
#     rows = (len(labeled_slots) // columns) + (1 if len(labeled_slots) % columns > 0 else 0)
    
#     for i, slot in enumerate(labeled_slots):
#         row = i // columns
#         col = i % columns
        
#         x = col * 160 + 20
#         y = row * 30 + 40
        
#         text = f"{slot['label']}: (x={slot['x']}, y={slot['y']})"
#         cv2.putText(lookup_img, text, (x, y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(18, 14))
        
#         plt.subplot(2, 2, 1)
#         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#         plt.title(f"Original with Labels ({len(labeled_slots)} slots total)")
#         plt.axis('off')
        
#         plt.subplot(2, 2, 2)
#         plt.imshow(cv2.cvtColor(enlarged_img, cv2.COLOR_BGR2RGB))
#         plt.title("Enlarged View (2x)")
#         plt.axis('off')
        
#         plt.subplot(2, 2, 3)
#         plt.imshow(cv2.cvtColor(schematic_img, cv2.COLOR_BGR2RGB))
#         plt.title("Schematic View")
#         plt.axis('off')
        
#         plt.subplot(2, 2, 4)
#         plt.imshow(cv2.cvtColor(lookup_img, cv2.COLOR_BGR2RGB))
#         plt.title("Label Index")
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.savefig("parking_visualization.png", dpi=300, bbox_inches='tight')
#         plt.show()
    
#     # Return both the labeled image and labeled slots data
#     return result_img, labeled_slots

# def generate_interactive_html(labeled_slots, image_path):
#     """
#     Generate an interactive HTML file to better visualize the parking slots
    
#     Args:
#         labeled_slots (list): List of dictionaries containing labeled slot information
#         image_path (str): Path to the original image
#     """
#     # Load image dimensions
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     img_height, img_width = image.shape[:2]
    
#     # Create HTML content
#     html_content = f'''
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Parking Slot Visualization</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
#             .container {{ display: flex; flex-direction: column; }}
#             .image-container {{ position: relative; margin-bottom: 20px; 
#                               width: {img_width}px; height: {img_height}px; }}
#             .image-container img {{ width: 100%; height: 100%; }}
#             .slot {{ position: absolute; border: 2px solid green; 
#                    display: flex; justify-content: center; align-items: center; }}
#             .slot-label {{ font-weight: bold; color: red; 
#                          background-color: rgba(255,255,255,0.7); 
#                          padding: 2px; border-radius: 3px; }}
#             .controls {{ margin-bottom: 20px; }}
#             .slot-table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
#             .slot-table th, .slot-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
#             .slot-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
#             .slot-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
#             .highlight {{ background-color: yellow !important; }}
#             .search-container {{ margin-bottom: 10px; }}
#         </style>
#     </head>
#     <body>
#         <h1>Parking Slot Visualization</h1>
        
#         <div class="controls">
#             <div class="search-container">
#                 <label for="slot-search">Search for slot: </label>
#                 <input type="text" id="slot-search" placeholder="Enter slot label (e.g. P1)">
#                 <button onclick="searchSlot()">Find</button>
#             </div>
            
#             <label for="label-size">Label Size: </label>
#             <input type="range" id="label-size" min="8" max="24" value="12" 
#                    oninput="updateLabelSize(this.value)">
#             <span id="size-value">12px</span>
            
#             <button onclick="toggleLabels()">Toggle Labels</button>
#         </div>
        
#         <div class="container">
#             <div class="image-container">
#                 <img src="data:image/jpeg;base64,PLACEHOLDER_FOR_BASE64_IMAGE" alt="Parking Lot">
#     '''
    
#     # Add each parking slot as a div
#     for slot in labeled_slots:
#         html_content += f'''
#                 <div class="slot" id="{slot['label']}" 
#                      style="left: {slot['x']}px; top: {slot['y']}px; 
#                             width: {slot['width']}px; height: {slot['height']}px;">
#                     <span class="slot-label">{slot['label']}</span>
#                 </div>
#         '''
    
#     html_content += '''
#             </div>
            
#             <h2>Parking Slot Data</h2>
#             <div class="search-container">
#                 <label for="table-search">Filter table: </label>
#                 <input type="text" id="table-search" placeholder="Filter by any column" 
#                        oninput="filterTable()">
#             </div>
            
#             <table class="slot-table" id="slot-table">
#                 <thead>
#                     <tr>
#                         <th>Label</th>
#                         <th>X Position</th>
#                         <th>Y Position</th>
#                         <th>Width</th>
#                         <th>Height</th>
#                         <th>Area</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#     '''
    
#     # Add table rows for each slot
#     for slot in labeled_slots:
#         html_content += f'''
#                     <tr id="row-{slot['label']}">
#                         <td>{slot['label']}</td>
#                         <td>{slot['x']}</td>
#                         <td>{slot['y']}</td>
#                         <td>{slot['width']}</td>
#                         <td>{slot['height']}</td>
#                         <td>{slot['area']}</td>
#                     </tr>
#         '''
    
#     html_content += '''
#                 </tbody>
#             </table>
#         </div>
        
#         <script>
#             function searchSlot() {
#                 // Reset highlighting
#                 const slots = document.querySelectorAll('.slot');
#                 slots.forEach(slot => {
#                     slot.style.backgroundColor = 'transparent';
#                     slot.style.zIndex = 1;
#                 });
                
#                 const rows = document.querySelectorAll('.slot-table tr');
#                 rows.forEach(row => {
#                     row.classList.remove('highlight');
#                 });
                
#                 // Get search value
#                 const searchValue = document.getElementById('slot-search').value.trim().toUpperCase();
#                 if (!searchValue) return;
                
#                 // Find and highlight the slot
#                 const slot = document.getElementById(searchValue);
#                 if (slot) {
#                     slot.style.backgroundColor = 'rgba(255, 255, 0, 0.5)';
#                     slot.style.zIndex = 100;
#                     slot.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    
#                     // Highlight the table row
#                     const row = document.getElementById('row-' + searchValue);
#                     if (row) {
#                         row.classList.add('highlight');
#                         row.scrollIntoView({ behavior: 'smooth', block: 'center' });
#                     }
#                 } else {
#                     alert('Slot ' + searchValue + ' not found!');
#                 }
#             }
            
#             function updateLabelSize(size) {
#                 document.getElementById('size-value').textContent = size + 'px';
#                 const labels = document.querySelectorAll('.slot-label');
#                 labels.forEach(label => {
#                     label.style.fontSize = size + 'px';
#                 });
#             }
            
#             function toggleLabels() {
#                 const labels = document.querySelectorAll('.slot-label');
#                 labels.forEach(label => {
#                     label.style.display = label.style.display === 'none' ? '' : 'none';
#                 });
#             }
            
#             function filterTable() {
#                 const filter = document.getElementById('table-search').value.toLowerCase();
#                 const rows = document.getElementById('slot-table').getElementsByTagName('tbody')[0].rows;
                
#                 for (let i = 0; i < rows.length; i++) {
#                     let visible = false;
#                     const cells = rows[i].getElementsByTagName('td');
                    
#                     for (let j = 0; j < cells.length; j++) {
#                         const cell = cells[j];
#                         if (cell.textContent.toLowerCase().indexOf(filter) > -1) {
#                             visible = true;
#                             break;
#                         }
#                     }
                    
#                     rows[i].style.display = visible ? '' : 'none';
#                 }
#             }
#         </script>
#     </body>
#     </html>
#     '''
    
#     # Save the HTML file
#     with open("parking_visualization.html", "w") as f:
#         f.write(html_content)
    
#     print("Interactive HTML visualization saved as 'parking_visualization.html'")
#     print("Note: The image placeholder needs to be replaced with the actual base64 encoded image.")

# def normalize_address(address):
#     """
#     Normalize address string for consistent folder/file naming
    
#     Args:
#         address (str): The original address string
        
#     Returns:
#         str: Normalized address string
#     """
#     # Common normalizations for address matching
#     normalized = address.strip()
    
#     # Handle common variations in Canadian addresses
#     normalized = normalized.replace(", ON N2J", ", On N2J")  # Match your file system
#     normalized = normalized.replace(", ON", ", On")  # General ON -> On replacement
    
#     return normalized

# def find_image_in_address_folder(address):
#     """
#     Find an image file in the address folder with flexible matching
    
#     Args:
#         address (str): The address to look for
        
#     Returns:
#         str: Path to the found image file, None if not found
#     """
#     # Base path for address folders
#     # base_path = r"C:\Users\jigsp\Desktop\Slotify\BE\parking_lot\Address"
#     base_path = "Address"

#     # Try multiple variations of the address
#     address_variations = [
#         address,  # Original address
#         normalize_address(address),  # Normalized version
#         address.replace(", ON", ", On"),  # ON -> On
#         address.replace(", On", ", ON"),  # On -> ON (reverse)
#     ]
    
#     # First, try to find a matching folder
#     for addr_variant in address_variations:
#         address_folder = os.path.join(base_path, addr_variant)
#         print(f"Checking folder: {address_folder}")
        
#         if os.path.exists(address_folder):
#             print(f"Found matching folder: {address_folder}")
            
#             # Common image file extensions
#             image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
            
#             # Search for image files in the address folder
#             for extension in image_extensions:
#                 pattern = os.path.join(address_folder, extension)
#                 image_files = glob.glob(pattern, recursive=False)
                
#                 # Also check for uppercase extensions
#                 pattern_upper = os.path.join(address_folder, extension.upper())
#                 image_files.extend(glob.glob(pattern_upper, recursive=False))
                
#                 if image_files:
#                     # Return the first image found
#                     image_path = image_files[0]
#                     print(f"Found image: {image_path}")
#                     return image_path
            
#             print(f"No image files found in folder: {address_folder}")
#             return None
    
#     # If no folder matches, try to find a direct file match
#     print("No matching folder found, trying direct file matching...")
    
#     # Also try to match the file directly (in case it's a file, not a folder)
#     for addr_variant in address_variations:
#         # Try with .png extension
#         direct_file_path = os.path.join(base_path, f"{addr_variant}.png")
#         print(f"Checking direct file: {direct_file_path}")
        
#         if os.path.exists(direct_file_path):
#             print(f"Found direct file: {direct_file_path}")
#             return direct_file_path
    
#     print(f"No image found for address variations: {address_variations}")
#     return None

# def main():
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Process parking lot images with address-based folder lookup')
#     parser.add_argument('--address', type=str, help='Address to look for in the Address folder')
    
#     # Parse arguments
#     args = parser.parse_args()
    
#     # Determine image path based on address parameter
#     if args.address:
#         print(f"Processing address: {args.address}")
#         image_path = find_image_in_address_folder(args.address)
        
#         if image_path is None:
#             print(f"Error: Could not find image for address '{args.address}'")
#             sys.exit(1)
#     else:
#         # Fallback to default image path if no address provided
#         image_path = "Screenshot (4).png"
#         print(f"No address provided, using default image: {image_path}")
        
#         # Check if default image exists
#         if not os.path.exists(image_path):
#             print(f"Error: Default image not found: {image_path}")
#             sys.exit(1)
    
#     try:
#         print(f"Using image: {image_path}")
        
#         # First, detect parking slots using your existing function
#         slots_dict = detect_parking_slots_all_colors(image_path, visualize=False)
        
#         # Debug: Print the structure of what we got back
#         print(f"DEBUG: Type of slots_dict: {type(slots_dict)}")
#         if isinstance(slots_dict, dict):
#             print(f"DEBUG: Keys in slots_dict: {list(slots_dict.keys())}")
        
#         # Extract all slots from the dictionary into a single list, preserving type information
#         all_slots = []
#         if isinstance(slots_dict, dict):
#             for slot_type, slot_list in slots_dict.items():
#                 if isinstance(slot_list, list):
#                     # Add type information to each slot
#                     for slot in slot_list:
#                         slot['type'] = slot_type  # Add the type to each slot
#                     all_slots.extend(slot_list)
#                     print(f"Added {len(slot_list)} {slot_type} slots")
#         else:
#             # If it's already a list, use it directly
#             all_slots = slots_dict if isinstance(slots_dict, list) else []
        
#         if not all_slots:
#             print("No parking slots detected in the image")
#             sys.exit(1)
        
#         print(f"Total slots collected: {len(all_slots)}")
#         slots = all_slots
        
#         # Then, label slots with sequential labels and improved visualization
#         result = label_parking_slots_sequential_improved(slots, image_path, visualize=True)
        
#         if result is not None:
#             labeled_image, labeled_slots = result
            
#             # Generate interactive HTML visualization
#             generate_interactive_html(labeled_slots, image_path)
            
#             # Print information about each labeled slot
#             print(f"\nLabeled {len(labeled_slots)} parking slots")
            
#             # Save slot data to CSV for easy reference
#             import csv
#             csv_filename = f"parking_slots_{args.address if args.address else 'default'}.csv"
#             with open(csv_filename, "w", newline="") as csvfile:
#                 fieldnames = ["label", "type", "id", "x", "y", "width", "height", "area"]
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
#                 writer.writeheader()
#                 for slot in labeled_slots:
#                     writer.writerow({
#                         "label": slot["label"],
#                         "type": slot.get("type", "Regular"),
#                         "id": slot["id"],
#                         "x": slot["x"],
#                         "y": slot["y"],
#                         "width": slot["width"],
#                         "height": slot["height"],
#                         "area": slot["area"]
#                     })
            
#             print(f"Slot data saved to '{csv_filename}'")
#             print("Processing completed successfully!")
            
#             # Return the total number of detected parking spaces
#             total_spaces = len(labeled_slots)
#             print(f"Total detected parking spaces: {total_spaces}")
#             return total_spaces
#         else:
#             print("Failed to process parking slots")
#             sys.exit(1)
            
#     except Exception as e:
#         print(f"Error during processing: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     total_spaces = main()
#     print(f"Final result: {total_spaces} parking spaces detected")


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
from parking_lot import detect_parking_slots_all_colors

def determine_parking_zones(slots, image_shape):
    """
    Divide the parking lot into 4 zones based on slot positions
    
    Args:
        slots (list): List of dictionaries containing slot information
        image_shape (tuple): Shape of the image (height, width, channels)
    
    Returns:
        dict: Dictionary mapping zone names to their boundaries and slots
    """
    if not slots:
        return {}
    
    img_height, img_width = image_shape[:2]
    
    # Calculate the center points for dividing the image
    center_x = img_width // 2
    center_y = img_height // 2
    
    # Define zone boundaries
    zones = {
        'A': {  # Top-left
            'boundary': (0, 0, center_x, center_y),
            'slots': [],
            'color': (255, 0, 0)  # Red
        },
        'B': {  # Bottom-left
            'boundary': (0, center_y, center_x, img_height),
            'slots': [],
            'color': (0, 255, 0)  # Green
        },
        'C': {  # Top-right
            'boundary': (center_x, 0, img_width, center_y),
            'slots': [],
            'color': (0, 0, 255)  # Blue
        },
        'D': {  # Bottom-right
            'boundary': (center_x, center_y, img_width, img_height),
            'slots': [],
            'color': (255, 255, 0)  # Cyan
        }
    }
    
    # Assign slots to zones based on their center position
    for slot in slots:
        slot_center_x = slot['x'] + slot['width'] // 2
        slot_center_y = slot['y'] + slot['height'] // 2
        
        # Determine which zone the slot belongs to
        if slot_center_x < center_x and slot_center_y < center_y:
            zones['A']['slots'].append(slot)
            slot['zone'] = 'A'
        elif slot_center_x < center_x and slot_center_y >= center_y:
            zones['B']['slots'].append(slot)
            slot['zone'] = 'B'
        elif slot_center_x >= center_x and slot_center_y < center_y:
            zones['C']['slots'].append(slot)
            slot['zone'] = 'C'
        else:  # slot_center_x >= center_x and slot_center_y >= center_y
            zones['D']['slots'].append(slot)
            slot['zone'] = 'D'
    
    return zones

def visualize_zones(image, zones):
    """
    Create a visualization showing the zone boundaries and slot distribution
    
    Args:
        image (numpy.ndarray): Original image
        zones (dict): Dictionary containing zone information
    
    Returns:
        numpy.ndarray: Image with zone visualization
    """
    zone_img = image.copy()
    img_height, img_width = image.shape[:2]
    
    # Draw zone boundaries
    center_x = img_width // 2
    center_y = img_height // 2
    
    # Draw vertical line (separating left and right)
    cv2.line(zone_img, (center_x, 0), (center_x, img_height), (255, 255, 255), 3)
    
    # Draw horizontal line (separating top and bottom)
    cv2.line(zone_img, (0, center_y), (img_width, center_y), (255, 255, 255), 3)
    
    # Add zone labels and draw colored overlays
    for zone_name, zone_info in zones.items():
        x1, y1, x2, y2 = zone_info['boundary']
        color = zone_info['color']
        
        # Create semi-transparent overlay for each zone
        overlay = zone_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(zone_img, 0.8, overlay, 0.2, 0, zone_img)
        
        # Add zone label
        label_x = x1 + (x2 - x1) // 2 - 20
        label_y = y1 + (y2 - y1) // 2
        cv2.putText(zone_img, f"Zone {zone_name}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(zone_img, f"({len(zone_info['slots'])} slots)", 
                    (label_x, label_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return zone_img

def label_parking_slots_sequential_improved(slots, image_path, visualize=True):
    """
    Assign sequential labels (P1, P2, P3...) to detected parking slots
    with improved visualization and zone detection
    
    Args:
        slots (list): List of dictionaries containing slot information
        image_path (str): Path to the original image
        visualize (bool): Whether to visualize the results
    
    Returns:
        tuple: (numpy.ndarray, list, dict) - Image with labeled parking slots, labeled slots data, and zones data
    """
    if not slots:
        print("No slots to label")
        return None
    
    # Load the image for visualization
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Determine parking zones
    zones = determine_parking_zones(slots, image.shape)
    
    # Create visualization images
    result_img = image.copy()
    enlarged_img = cv2.resize(image.copy(), (image.shape[1]*2, image.shape[0]*2))
    schematic_img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255  # White background
    zone_img = visualize_zones(image, zones)
    
    # Group slots by type
    slot_types = {
        'Regular': [],
        'Entry': [],
        'Reservation': [],
        'Accessible': []
    }
    
    # Categorize slots by type (using the exact keys from your detection function)
    for slot in slots:
        slot_type = slot.get('type', 'Regular')
        if slot_type in slot_types:
            slot_types[slot_type].append(slot)
        else:
            slot_types['Regular'].append(slot)  # Default to Regular if type unknown
    
    # Sort each type by coordinates (left-to-right, top-to-bottom)
    for slot_type in slot_types:
        slot_types[slot_type].sort(key=lambda slot: (slot['y'], slot['x']))
    
    # Label prefixes for each type
    prefixes = {
        'Regular': 'P',
        'Entry': 'E',
        'Reservation': 'R',
        'Accessible': 'A'
    }
    
    # Assign sequential labels by type
    labeled_slots = []
    for slot_type, slot_list in slot_types.items():
        prefix = prefixes[slot_type]
        for i, slot in enumerate(slot_list):
            # Create a copy of the slot with the sequential label added
            labeled_slot = slot.copy()
            labeled_slot['label'] = f"{prefix}{i + 1}"
            labeled_slot['zone_label'] = f"Zone {slot.get('zone', 'Unknown')}"
            labeled_slots.append(labeled_slot)
            
            # Get zone color for visualization
            zone_color = zones.get(slot.get('zone', 'A'), {}).get('color', (0, 255, 0))
            
            # Method 1: Original image with smaller font and zone color
            text_pos = (slot['x'] + 5, slot['y'] + 20)
            cv2.putText(result_img, labeled_slot['label'], text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.rectangle(result_img, 
                         (slot['x'], slot['y']), 
                         (slot['x'] + slot['width'], slot['y'] + slot['height']), 
                         zone_color, 2)
            
            # Add zone indicator
            zone_text_pos = (slot['x'] + 5, slot['y'] + slot['height'] - 5)
            cv2.putText(result_img, slot.get('zone', ''), zone_text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, zone_color, 1)
            
            # Method 2: Enlarged image
            cv2.putText(enlarged_img, labeled_slot['label'], 
                        (slot['x']*2 + 10, slot['y']*2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(enlarged_img, 
                         (slot['x']*2, slot['y']*2), 
                         ((slot['x'] + slot['width'])*2, (slot['y'] + slot['height'])*2), 
                         zone_color, 3)
            
            # Method 3: Schematic view with normalized spacing
            # Scale factor to normalize the parking lot layout
            scale_x = 1800 / image.shape[1]
            scale_y = 1300 / image.shape[0]
            
            sch_x = int(slot['x'] * scale_x) + 100
            sch_y = int(slot['y'] * scale_y) + 100
            sch_w = max(int(slot['width'] * scale_x), 40)  # Minimum width
            sch_h = max(int(slot['height'] * scale_y), 40)  # Minimum height
            
            cv2.rectangle(schematic_img, 
                         (sch_x, sch_y), 
                         (sch_x + sch_w, sch_y + sch_h), 
                         zone_color, 2)
            cv2.putText(schematic_img, labeled_slot['label'], 
                        (sch_x + sch_w//4, sch_y + sch_h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            cv2.putText(schematic_img, slot.get('zone', ''), 
                        (sch_x + sch_w//4, sch_y + sch_h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, zone_color, 1)
    
    # Create a lookup table/index for easier verification
    lookup_img = np.ones((800, 800, 3), dtype=np.uint8) * 255
    columns = 4
    rows = (len(labeled_slots) // columns) + (1 if len(labeled_slots) % columns > 0 else 0)
    
    for i, slot in enumerate(labeled_slots):
        row = i // columns
        col = i % columns
        
        x = col * 200 + 10
        y = row * 35 + 40
        
        text = f"{slot['label']}: Zone {slot.get('zone', 'N/A')}"
        cv2.putText(lookup_img, text, (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        coord_text = f"(x={slot['x']}, y={slot['y']})"
        cv2.putText(lookup_img, coord_text, (x, y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    # Print zone statistics
    print("\n=== ZONE STATISTICS ===")
    for zone_name, zone_info in zones.items():
        print(f"Zone {zone_name}: {len(zone_info['slots'])} slots")
    
    # Visualization
    if visualize:
        plt.figure(figsize=(20, 16))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original with Labels & Zones ({len(labeled_slots)} slots total)")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(zone_img, cv2.COLOR_BGR2RGB))
        plt.title("Zone Boundaries")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(enlarged_img, cv2.COLOR_BGR2RGB))
        plt.title("Enlarged View (2x)")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(schematic_img, cv2.COLOR_BGR2RGB))
        plt.title("Schematic View with Zones")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(lookup_img, cv2.COLOR_BGR2RGB))
        plt.title("Label Index with Zones")
        plt.axis('off')
        
        # Zone distribution chart
        plt.subplot(2, 3, 6)
        zone_names = list(zones.keys())
        zone_counts = [len(zones[zone]['slots']) for zone in zone_names]
        colors = ['red', 'green', 'blue', 'cyan']
        
        bars = plt.bar(zone_names, zone_counts, color=colors, alpha=0.7)
        plt.title('Slots Distribution by Zone')
        plt.xlabel('Zone')
        plt.ylabel('Number of Slots')
        
        # Add value labels on bars
        for bar, count in zip(bars, zone_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("parking_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Return the labeled image, labeled slots data, and zones data
    return result_img, labeled_slots, zones

def generate_interactive_html(labeled_slots, image_path, zones):
    """
    Generate an interactive HTML file to better visualize the parking slots with zones
    
    Args:
        labeled_slots (list): List of dictionaries containing labeled slot information
        image_path (str): Path to the original image
        zones (dict): Dictionary containing zone information
    """
    # Load image dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    # Create HTML content
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parking Slot Visualization with Zones</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ display: flex; flex-direction: column; }}
            .image-container {{ position: relative; margin-bottom: 20px; 
                              width: {img_width}px; height: {img_height}px; }}
            .image-container img {{ width: 100%; height: 100%; }}
            .slot {{ position: absolute; border: 2px solid green; 
                   display: flex; justify-content: center; align-items: center; }}
            .slot-label {{ font-weight: bold; color: red; 
                         background-color: rgba(255,255,255,0.7); 
                         padding: 2px; border-radius: 3px; }}
            .zone-overlay {{ position: absolute; border: 3px solid white; 
                           background-color: rgba(255,255,255,0.1); }}
            .zone-A {{ border-color: red; background-color: rgba(255,0,0,0.1); }}
            .zone-B {{ border-color: green; background-color: rgba(0,255,0,0.1); }}
            .zone-C {{ border-color: blue; background-color: rgba(0,0,255,0.1); }}
            .zone-D {{ border-color: cyan; background-color: rgba(255,255,0,0.1); }}
            .controls {{ margin-bottom: 20px; }}
            .slot-table {{ border-collapse: collapse; width: 100%; max-width: 1000px; }}
            .slot-table th, .slot-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .slot-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .slot-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
            .highlight {{ background-color: yellow !important; }}
            .search-container {{ margin-bottom: 10px; }}
            .zone-stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .zone-stat {{ padding: 10px; border-radius: 5px; color: white; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Parking Slot Visualization with Zones</h1>
        
        <div class="zone-stats">
            <div class="zone-stat" style="background-color: red;">Zone A: {len(zones.get('A', {}).get('slots', []))} slots</div>
            <div class="zone-stat" style="background-color: green;">Zone B: {len(zones.get('B', {}).get('slots', []))} slots</div>
            <div class="zone-stat" style="background-color: blue;">Zone C: {len(zones.get('C', {}).get('slots', []))} slots</div>
            <div class="zone-stat" style="background-color: #00FFFF; color: black;">Zone D: {len(zones.get('D', {}).get('slots', []))} slots</div>
        </div>
        
        <div class="controls">
            <div class="search-container">
                <label for="slot-search">Search for slot: </label>
                <input type="text" id="slot-search" placeholder="Enter slot label (e.g. P1)">
                <button onclick="searchSlot()">Find</button>
            </div>
            
            <div class="search-container">
                <label for="zone-filter">Filter by zone: </label>
                <select id="zone-filter" onchange="filterByZone()">
                    <option value="">All Zones</option>
                    <option value="A">Zone A</option>
                    <option value="B">Zone B</option>
                    <option value="C">Zone C</option>
                    <option value="D">Zone D</option>
                </select>
            </div>
            
            <label for="label-size">Label Size: </label>
            <input type="range" id="label-size" min="8" max="24" value="12" 
                   oninput="updateLabelSize(this.value)">
            <span id="size-value">12px</span>
            
            <button onclick="toggleLabels()">Toggle Labels</button>
            <button onclick="toggleZones()">Toggle Zone Overlays</button>
        </div>
        
        <div class="container">
            <div class="image-container">
                <img src="data:image/jpeg;base64,PLACEHOLDER_FOR_BASE64_IMAGE" alt="Parking Lot">
    '''
    
    # Add zone overlays
    center_x = img_width // 2
    center_y = img_height // 2
    
    zone_overlays = [
        ('A', 0, 0, center_x, center_y),
        ('B', 0, center_y, center_x, img_height),
        ('C', center_x, 0, img_width, center_y),
        ('D', center_x, center_y, img_width, img_height)
    ]
    
    for zone_name, x1, y1, x2, y2 in zone_overlays:
        html_content += f'''
                <div class="zone-overlay zone-{zone_name}" id="zone-{zone_name}"
                     style="left: {x1}px; top: {y1}px; 
                            width: {x2-x1}px; height: {y2-y1}px;">
                </div>
        '''
    
    # Add each parking slot as a div
    for slot in labeled_slots:
        zone = slot.get('zone', 'A')
        html_content += f'''
                <div class="slot zone-{zone}-slot" id="{slot['label']}" 
                     style="left: {slot['x']}px; top: {slot['y']}px; 
                            width: {slot['width']}px; height: {slot['height']}px;"
                     data-zone="{zone}">
                    <span class="slot-label">{slot['label']}</span>
                </div>
        '''
    
    html_content += '''
            </div>
            
            <h2>Parking Slot Data with Zones</h2>
            <div class="search-container">
                <label for="table-search">Filter table: </label>
                <input type="text" id="table-search" placeholder="Filter by any column" 
                       oninput="filterTable()">
            </div>
            
            <table class="slot-table" id="slot-table">
                <thead>
                    <tr>
                        <th>Label</th>
                        <th>Zone</th>
                        <th>X Position</th>
                        <th>Y Position</th>
                        <th>Width</th>
                        <th>Height</th>
                        <th>Area</th>
                    </tr>
                </thead>
                <tbody>
    '''
    
    # Add table rows for each slot
    for slot in labeled_slots:
        html_content += f'''
                    <tr id="row-{slot['label']}" data-zone="{slot.get('zone', 'A')}">
                        <td>{slot['label']}</td>
                        <td>Zone {slot.get('zone', 'A')}</td>
                        <td>{slot['x']}</td>
                        <td>{slot['y']}</td>
                        <td>{slot['width']}</td>
                        <td>{slot['height']}</td>
                        <td>{slot['area']}</td>
                    </tr>
        '''
    
    html_content += '''
                </tbody>
            </table>
        </div>
        
        <script>
            function searchSlot() {
                // Reset highlighting
                resetHighlighting();
                
                // Get search value
                const searchValue = document.getElementById('slot-search').value.trim().toUpperCase();
                if (!searchValue) return;
                
                // Find and highlight the slot
                const slot = document.getElementById(searchValue);
                if (slot) {
                    slot.style.backgroundColor = 'rgba(255, 255, 0, 0.5)';
                    slot.style.zIndex = 100;
                    slot.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    
                    // Highlight the table row
                    const row = document.getElementById('row-' + searchValue);
                    if (row) {
                        row.classList.add('highlight');
                        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                } else {
                    alert('Slot ' + searchValue + ' not found!');
                }
            }
            
            function filterByZone() {
                const selectedZone = document.getElementById('zone-filter').value;
                const slots = document.querySelectorAll('.slot');
                const rows = document.querySelectorAll('.slot-table tbody tr');
                
                slots.forEach(slot => {
                    if (selectedZone === '' || slot.dataset.zone === selectedZone) {
                        slot.style.display = 'flex';
                    } else {
                        slot.style.display = 'none';
                    }
                });
                
                rows.forEach(row => {
                    if (selectedZone === '' || row.dataset.zone === selectedZone) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }
            
            function resetHighlighting() {
                const slots = document.querySelectorAll('.slot');
                slots.forEach(slot => {
                    slot.style.backgroundColor = 'transparent';
                    slot.style.zIndex = 1;
                });
                
                const rows = document.querySelectorAll('.slot-table tr');
                rows.forEach(row => {
                    row.classList.remove('highlight');
                });
            }
            
            function updateLabelSize(size) {
                document.getElementById('size-value').textContent = size + 'px';
                const labels = document.querySelectorAll('.slot-label');
                labels.forEach(label => {
                    label.style.fontSize = size + 'px';
                });
            }
            
            function toggleLabels() {
                const labels = document.querySelectorAll('.slot-label');
                labels.forEach(label => {
                    label.style.display = label.style.display === 'none' ? '' : 'none';
                });
            }
            
            function toggleZones() {
                const zones = document.querySelectorAll('.zone-overlay');
                zones.forEach(zone => {
                    zone.style.display = zone.style.display === 'none' ? '' : 'none';
                });
            }
            
            function filterTable() {
                const filter = document.getElementById('table-search').value.toLowerCase();
                const rows = document.getElementById('slot-table').getElementsByTagName('tbody')[0].rows;
                
                for (let i = 0; i < rows.length; i++) {
                    let visible = false;
                    const cells = rows[i].getElementsByTagName('td');
                    
                    for (let j = 0; j < cells.length; j++) {
                        const cell = cells[j];
                        if (cell.textContent.toLowerCase().indexOf(filter) > -1) {
                            visible = true;
                            break;
                        }
                    }
                    
                    rows[i].style.display = visible ? '' : 'none';
                }
            }
        </script>
    </body>
    </html>
    '''
    
    # Save the HTML file
    with open("parking_visualization.html", "w") as f:
        f.write(html_content)
    
    print("Interactive HTML visualization with zones saved as 'parking_visualization.html'")
    print("Note: The image placeholder needs to be replaced with the actual base64 encoded image.")

def normalize_address(address):
    """
    Normalize address string for consistent folder/file naming
    
    Args:
        address (str): The original address string
        
    Returns:
        str: Normalized address string
    """
    # Common normalizations for address matching
    normalized = address.strip()
    
    # Handle common variations in Canadian addresses
    normalized = normalized.replace(", ON N2J", ", On N2J")  # Match your file system
    normalized = normalized.replace(", ON", ", On")  # General ON -> On replacement
    
    return normalized

def find_image_in_address_folder(address):
    """
    Find an image file in the address folder with flexible matching
    
    Args:
        address (str): The address to look for
        
    Returns:
        str: Path to the found image file, None if not found
    """
    # Base path for address folders
    # base_path = r"C:\Users\jigsp\Desktop\Slotify\BE\parking_lot\Address"
    base_path = "Address"

    # Try multiple variations of the address
    address_variations = [
        address,  # Original address
        normalize_address(address),  # Normalized version
        address.replace(", ON", ", On"),  # ON -> On
        address.replace(", On", ", ON"),  # On -> ON (reverse)
    ]
    
    # First, try to find a matching folder
    for addr_variant in address_variations:
        address_folder = os.path.join(base_path, addr_variant)
        print(f"Checking folder: {address_folder}")
        
        if os.path.exists(address_folder):
            print(f"Found matching folder: {address_folder}")
            
            # Common image file extensions
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
            
            # Search for image files in the address folder
            for extension in image_extensions:
                pattern = os.path.join(address_folder, extension)
                image_files = glob.glob(pattern, recursive=False)
                
                # Also check for uppercase extensions
                pattern_upper = os.path.join(address_folder, extension.upper())
                image_files.extend(glob.glob(pattern_upper, recursive=False))
                
                if image_files:
                    # Return the first image found
                    image_path = image_files[0]
                    print(f"Found image: {image_path}")
                    return image_path
            
            print(f"No image files found in folder: {address_folder}")
            return None
    
    # If no folder matches, try to find a direct file match
    print("No matching folder found, trying direct file matching...")
    
    # Also try to match the file directly (in case it's a file, not a folder)
    for addr_variant in address_variations:
        # Try with .png extension
        direct_file_path = os.path.join(base_path, f"{addr_variant}.png")
        print(f"Checking direct file: {direct_file_path}")
        
        if os.path.exists(direct_file_path):
            print(f"Found direct file: {direct_file_path}")
            return direct_file_path
    
    print(f"No image found for address variations: {address_variations}")
    return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process parking lot images with address-based folder lookup')
    parser.add_argument('--address', type=str, help='Address to look for in the Address folder')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine image path based on address parameter
    if args.address:
        print(f"Processing address: {args.address}")
        image_path = find_image_in_address_folder(args.address)
        
        if image_path is None:
            print(f"Error: Could not find image for address '{args.address}'")
            sys.exit(1)
    else:
        # Fallback to default image path if no address provided
        image_path = "Modified_Parking_Lot.png"
        print(f"No address provided, using default image: {image_path}")
        
        # Check if default image exists
        if not os.path.exists(image_path):
            print(f"Error: Default image not found: {image_path}")
            sys.exit(1)
    
    try:
        print(f"Using image: {image_path}")
        
        # First, detect parking slots using your existing function
        slots_dict = detect_parking_slots_all_colors(image_path, visualize=False)
        
        # Debug: Print the structure of what we got back
        print(f"DEBUG: Type of slots_dict: {type(slots_dict)}")
        if isinstance(slots_dict, dict):
            print(f"DEBUG: Keys in slots_dict: {list(slots_dict.keys())}")
        
        # Extract all slots from the dictionary into a single list, preserving type information
        all_slots = []
        if isinstance(slots_dict, dict):
            for slot_type, slot_list in slots_dict.items():
                if isinstance(slot_list, list):
                    # Add type information to each slot
                    for slot in slot_list:
                        slot['type'] = slot_type  # Add the type to each slot
                    all_slots.extend(slot_list)
                    print(f"Added {len(slot_list)} {slot_type} slots")
        else:
            # If it's already a list, use it directly
            all_slots = slots_dict if isinstance(slots_dict, list) else []
        
        if not all_slots:
            print("No parking slots detected in the image")
            sys.exit(1)
        
        print(f"Total slots collected: {len(all_slots)}")
        slots = all_slots
        
        # Load the image to get its shape for zone division
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Divide the parking lot into 4 zones (A, B, C, D) based on slot positions
        # This adds zone information to each slot
        zones = determine_parking_zones(slots, image.shape)
        
        # print(f"\n=== ZONE DIVISION COMPLETED ===")
        # print("Zone Layout:")
        # print("┌─────────┬─────────┐")
        # print("│  Zone A │  Zone C │")
        # print("│ (Top-L) │ (Top-R) │")
        # print("├─────────┼─────────┤")
        # print("│  Zone B │  Zone D │")
        # print("│ (Bot-L) │ (Bot-R) │")
        # print("└─────────┴─────────┘")
        # print()
        
        # Print zone statistics with detailed breakdown
        total_detected = 0
        for zone_name, zone_info in zones.items():
            zone_count = len(zone_info['slots'])
            total_detected += zone_count
            print(f"Zone {zone_name}: {zone_count} slots detected")
            
            # Show slot distribution by type within each zone
            if zone_count > 0:
                type_counts = {}
                for slot in zone_info['slots']:
                    slot_type = slot.get('type', 'Regular')
                    type_counts[slot_type] = type_counts.get(slot_type, 0) + 1
                
                type_breakdown = ', '.join([f"{count} {slot_type}" for slot_type, count in type_counts.items()])
                print(f"  └─ Breakdown: {type_breakdown}")
        
        print(f"\nTotal slots with zone assignments: {total_detected}")
        
        # Then, label slots with sequential labels and improved visualization
        # This function now includes zone information in the visualization
        result = label_parking_slots_sequential_improved(slots, image_path, visualize=True)
        
        if result is not None:
            labeled_image, labeled_slots, zones_data = result
            
            # Generate interactive HTML visualization with zone information
            generate_interactive_html(labeled_slots, image_path, zones_data)
            
            # Print information about each labeled slot with zone details
            print(f"\nLabeled {len(labeled_slots)} parking slots with zone assignments")
            
            # Create enhanced CSV with zone information
            import csv
            csv_filename = f"parking_slots_{args.address.replace('/', '_').replace(',', '_') if args.address else 'default'}.csv"
            with open(csv_filename, "w", newline="") as csvfile:
                fieldnames = ["label", "type", "zone", "zone_label", "id", "x", "y", "width", "height", "area"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for slot in labeled_slots:
                    writer.writerow({
                        "label": slot["label"],
                        "type": slot.get("type", "Regular"),
                        "zone": slot.get("zone", "Unknown"),
                        "zone_label": slot.get("zone_label", "Zone Unknown"),
                        "id": slot["id"],
                        "x": slot["x"],
                        "y": slot["y"],
                        "width": slot["width"],
                        "height": slot["height"],
                        "area": slot["area"]
                    })
            
            # Create a zone summary file for navigation purposes
            zone_summary_filename = f"zone_summary_{args.address.replace('/', '_').replace(',', '_') if args.address else 'default'}.txt"
            with open(zone_summary_filename, "w") as zone_file:
                # zone_file.write("PARKING LOT ZONE SUMMARY\n")
                # zone_file.write("========================\n\n")
                # zone_file.write("Zone Layout for Navigation:\n")
                # zone_file.write("┌─────────┬─────────┐\n")
                # zone_file.write("│  Zone A │  Zone C │\n")
                # zone_file.write("│ (Top-L) │ (Top-R) │\n")
                # zone_file.write("├─────────┼─────────┤\n")
                # zone_file.write("│  Zone B │  Zone D │\n")
                # zone_file.write("│ (Bot-L) │ (Bot-R) │\n")
                # zone_file.write("└─────────┴─────────┘\n\n")
                
                for zone_name, zone_info in zones_data.items():
                    zone_file.write(f"Zone {zone_name}: {len(zone_info['slots'])} slots\n")
                    slot_labels = [slot['label'] for slot in zone_info['slots']]
                    zone_file.write(f"Slots: {', '.join(sorted(slot_labels))}\n\n")
            
            print(f"Slot data with zones saved to '{csv_filename}'")
            print(f"Zone summary for navigation saved to '{zone_summary_filename}'")
            print("Processing completed successfully!")
            
            # Print final zone distribution for quick reference
            print(f"\n=== FINAL ZONE DISTRIBUTION ===")
            for zone_name in ['A', 'B', 'C', 'D']:
                zone_slots = zones_data.get(zone_name, {}).get('slots', [])
                if zone_slots:
                    labels = sorted([slot['label'] for slot in zone_slots])
                    print(f"Zone {zone_name} ({len(zone_slots)} slots): {', '.join(labels)}")
            
            # Return the total number of detected parking spaces
            total_spaces = len(labeled_slots)
            print(f"Total detected parking spaces: {total_spaces}")
            return total_spaces
        else:
            print("Failed to process parking slots")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    total_spaces = main()
    print(f"Final result: {total_spaces} parking spaces detected")



