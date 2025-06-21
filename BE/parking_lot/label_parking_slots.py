# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import argparse
# import glob
# from parking_lot import detect_parking_slots_grid

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
    
#     # Extract coordinates for sorting (left-to-right, top-to-bottom)
#     slot_coordinates = [(slot['id'], slot['x'], slot['y']) for slot in slots]
#     slot_coordinates.sort(key=lambda coord: (coord[2], coord[1]))
    
#     # Assign sequential labels
#     labeled_slots = []
#     for i, (slot_id, _, _) in enumerate(slot_coordinates):
#         # Find the original slot by ID
#         for slot in slots:
#             if slot['id'] == slot_id:
#                 # Create a copy of the slot with the sequential label added
#                 labeled_slot = slot.copy()
#                 labeled_slot['label'] = f"P{i + 1}"
#                 labeled_slots.append(labeled_slot)
                
#                 # Method 1: Original image with smaller font
#                 text_pos = (slot['x'] + 5, slot['y'] + 20)
#                 cv2.putText(result_img, labeled_slot['label'], text_pos,
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#                 cv2.rectangle(result_img, 
#                              (slot['x'], slot['y']), 
#                              (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                              (0, 255, 0), 1)
                
#                 # Method 2: Enlarged image
#                 cv2.putText(enlarged_img, labeled_slot['label'], 
#                             (slot['x']*2 + 10, slot['y']*2 + 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#                 cv2.rectangle(enlarged_img, 
#                              (slot['x']*2, slot['y']*2), 
#                              ((slot['x'] + slot['width'])*2, (slot['y'] + slot['height'])*2), 
#                              (0, 255, 0), 2)
                
#                 # Method 3: Schematic view with normalized spacing
#                 # Scale factor to normalize the parking lot layout
#                 scale_x = 1800 / image.shape[1]
#                 scale_y = 1300 / image.shape[0]
                
#                 sch_x = int(slot['x'] * scale_x) + 100
#                 sch_y = int(slot['y'] * scale_y) + 100
#                 sch_w = max(int(slot['width'] * scale_x), 40)  # Minimum width
#                 sch_h = max(int(slot['height'] * scale_y), 40)  # Minimum height
                
#                 cv2.rectangle(schematic_img, 
#                              (sch_x, sch_y), 
#                              (sch_x + sch_w, sch_y + sch_h), 
#                              (0, 0, 0), 2)
#                 cv2.putText(schematic_img, labeled_slot['label'], 
#                             (sch_x + sch_w//4, sch_y + sch_h//2),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 break
    
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
#         plt.title(f"Original with Labels (P1-P{len(labeled_slots)})")
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
    
#     # Return the labeled image instead of the dictionary list
#     return result_img

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
#     base_path = "./Address"



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
#         slots = detect_parking_slots_grid(image_path, visualize=False)
        
#         if not slots:
#             print("No parking slots detected in the image")
#             sys.exit(1)
        
#         # Then, label slots with sequential labels and improved visualization
#         labeled_image = label_parking_slots_sequential_improved(slots, image_path, visualize=True)
        
#         if labeled_image is not None:
#             # Generate interactive HTML visualization
#             # First we need to get the labeled_slots list instead of just the image
#             # Re-run the labeling to get the slots data
#             slots_data = detect_parking_slots_grid(image_path, visualize=False)
            
#             # Extract coordinates for sorting (left-to-right, top-to-bottom)
#             slot_coordinates = [(slot['id'], slot['x'], slot['y']) for slot in slots_data]
#             slot_coordinates.sort(key=lambda coord: (coord[2], coord[1]))
            
#             # Assign sequential labels to create labeled_slots
#             labeled_slots = []
#             for i, (slot_id, _, _) in enumerate(slot_coordinates):
#                 for slot in slots_data:
#                     if slot['id'] == slot_id:
#                         labeled_slot = slot.copy()
#                         labeled_slot['label'] = f"P{i + 1}"
#                         labeled_slots.append(labeled_slot)
#                         break
            
#             generate_interactive_html(labeled_slots, image_path)
            
#             # Print information about each labeled slot
#             print(f"\nLabeled {len(labeled_slots)} parking slots")
            
#             # Save slot data to CSV for easy reference
#             import csv
#             csv_filename = f"parking_slots_{args.address if args.address else 'default'}.csv"
#             with open(csv_filename, "w", newline="") as csvfile:
#                 fieldnames = ["label", "id", "x", "y", "width", "height", "area"]
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
#                 writer.writeheader()
#                 for slot in labeled_slots:
#                     writer.writerow({
#                         "label": slot["label"],
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




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import argparse
# import glob
# import csv

# def detect_parking_slots_all_colors(image_path, visualize=True):
#     """
#     Detect individual parking slots from an image with different colored rectangular outlines
#     - Red: Entry slots
#     - Purple: Accessible slots  
#     - Yellow: Reservation slots
#     - Blue: Regular parking slots
    
#     Args:
#         image_path (str): Path to the image with colored parking slots
#         visualize (bool): Whether to visualize the results
    
#     Returns:
#         dict: Dictionary containing information about each parking slot by color category
#     """
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     # Create a copy for visualization
#     result_img = image.copy()
    
#     # Convert to HSV color space for better color detection
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define color ranges for each parking type
#     color_configs = {
#         'Entry': {
#             'lower': [np.array([0, 100, 100]), np.array([170, 100, 100])],  # Two ranges for red
#             'upper': [np.array([10, 255, 255]), np.array([180, 255, 255])],
#             'color': (0, 0, 255),  # Red for visualization
#             'prefix': 'E'
#         },
#         'Accessible': {
#             'lower': [np.array([120, 100, 100])],  # Purple/Magenta
#             'upper': [np.array([160, 255, 255])],
#             'color': (128, 0, 128),  # Purple for visualization
#             'prefix': 'A'
#         },
#         'Reservation': {
#             'lower': [np.array([20, 100, 100])],  # Yellow
#             'upper': [np.array([30, 255, 255])],
#             'color': (0, 255, 255),  # Yellow for visualization
#             'prefix': 'R'
#         },
#         'Regular': {
#             'lower': [np.array([90, 100, 100])],  # Blue
#             'upper': [np.array([130, 255, 255])],
#             'color': (255, 0, 0),  # Blue for visualization
#             'prefix': 'P'
#         }
#     }
    
#     # Dictionary to store all parking slots by category
#     all_parking_slots = {
#         'Entry': [],
#         'Accessible': [],
#         'Reservation': [],
#         'Regular': []
#     }
    
#     # Process each color category
#     for category, config in color_configs.items():
#         print(f"Processing {category} parking slots...")
        
#         # Create mask for this color
#         masks = []
#         for i, lower in enumerate(config['lower']):
#             upper = config['upper'][i]
#             mask = cv2.inRange(hsv, lower, upper)
#             masks.append(mask)
        
#         # Combine masks if multiple ranges (like for red)
#         if len(masks) > 1:
#             combined_mask = cv2.bitwise_or(masks[0], masks[1])
#         else:
#             combined_mask = masks[0]
        
#         # Detect individual parking slots for this color
#         slots = detect_slots_for_color(combined_mask, category, config, result_img)
#         all_parking_slots[category] = slots
        
#         print(f"Detected {len(slots)} {category} parking slots")
    
#     # Calculate totals
#     total_slots = sum(len(slots) for slots in all_parking_slots.values())
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(20, 15))
        
#         # Original image
#         plt.subplot(3, 3, 1)
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title("Original Image")
#         plt.axis('off')
        
#         # Final result with all slots
#         plt.subplot(3, 3, 2)
#         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#         plt.title(f"All Detected Slots (Total: {total_slots})")
#         plt.axis('off')
        
#         # Individual color masks and results
#         subplot_idx = 3
#         for category, config in color_configs.items():
#             if subplot_idx > 9:
#                 break
                
#             # Create mask for visualization
#             masks = []
#             for i, lower in enumerate(config['lower']):
#                 upper = config['upper'][i]
#                 mask = cv2.inRange(hsv, lower, upper)
#                 masks.append(mask)
            
#             if len(masks) > 1:
#                 combined_mask = cv2.bitwise_or(masks[0], masks[1])
#             else:
#                 combined_mask = masks[0]
            
#             # Show mask
#             plt.subplot(3, 3, subplot_idx)
#             plt.imshow(combined_mask, cmap='gray')
#             plt.title(f"{category} Mask")
#             plt.axis('off')
#             subplot_idx += 1
            
#             # Show detected slots for this category
#             if subplot_idx <= 9:
#                 category_img = image.copy()
#                 slots = all_parking_slots[category]
#                 for slot in slots:
#                     cv2.rectangle(category_img, (slot['x'], slot['y']), 
#                                 (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                                 config['color'], 2)
#                     cv2.putText(category_img, slot['label'], 
#                               (slot['x'] + 5, slot['y'] + 20), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
                
#                 plt.subplot(3, 3, subplot_idx)
#                 plt.imshow(cv2.cvtColor(category_img, cv2.COLOR_BGR2RGB))
#                 plt.title(f"{category} Slots ({len(slots)})")
#                 plt.axis('off')
#                 subplot_idx += 1
        
#         plt.tight_layout()
#         plt.show()
    
#     return all_parking_slots

# def detect_slots_for_color(color_mask, category, config, result_img):
#     """
#     Detect individual parking slots for a specific color using the same approach as blue slots
#     """
#     parking_slots = []
    
#     # Find the region of interest - the entire colored area
#     contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         print(f"No {category} contours found in the image")
#         return parking_slots
    
#     # Combine all contours to get the overall area
#     all_cnts = np.vstack([cnt for cnt in contours if cv2.contourArea(cnt) > 50])
#     if len(all_cnts) == 0:
#         return parking_slots
        
#     x, y, w, h = cv2.boundingRect(all_cnts)
#     print(f"Found overall {category} area: x={x}, y={y}, width={w}, height={h}")
    
#     # Method 1: Direct detection of colored rectangles
#     contours, _ = cv2.findContours(color_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
#     slot_id = 1
#     rectangle_contours = []
    
#     for contour in contours:
#         # Get the bounding rectangle
#         rx, ry, rw, rh = cv2.boundingRect(contour)
#         area = cv2.contourArea(contour)
        
#         # Filter by size - exclude very small or very large contours
#         if area > 50 and area < 5000:
#             rect_area = rw * rh
#             # Check if the shape is roughly rectangular
#             if area / rect_area > 0.2:
#                 rectangle_contours.append(contour)
                
#                 # Add to parking slots
#                 slot_info = {
#                     "id": slot_id,
#                     "label": f"{config['prefix']}{slot_id}",
#                     "x": rx,
#                     "y": ry,
#                     "width": rw,
#                     "height": rh,
#                     "center_x": rx + rw // 2,
#                     "center_y": ry + rh // 2,
#                     "area": area,
#                     "category": category
#                 }
#                 parking_slots.append(slot_info)
#                 slot_id += 1
                
#                 # Draw rectangle and label on result image
#                 cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), config['color'], 2)
#                 cv2.putText(result_img, slot_info['label'], (rx+5, ry+20), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
    
#     # Method 2: If we don't have many slots, try contour hierarchy approach
#     if len(parking_slots) < 5:  # Threshold can be adjusted
#         print(f"Using hierarchy approach for {category}...")
        
#         contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Reset if we're trying a different approach
#         additional_slots = []
        
#         if hierarchy is not None:
#             hierarchy = hierarchy[0]
#             for i, contour in enumerate(contours):
#                 if cv2.contourArea(contour) < 50:
#                     continue
                
#                 rx, ry, rw, rh = cv2.boundingRect(contour)
                
#                 # Check if this contour has a parent or is a reasonable size
#                 if hierarchy[i][3] != -1 or cv2.contourArea(contour) > 200:
#                     # Avoid duplicates by checking if we already have a slot in this area
#                     duplicate = False
#                     for existing_slot in parking_slots:
#                         if (abs(existing_slot['x'] - rx) < 20 and 
#                             abs(existing_slot['y'] - ry) < 20):
#                             duplicate = True
#                             break
                    
#                     if not duplicate:
#                         slot_info = {
#                             "id": slot_id,
#                             "label": f"{config['prefix']}{slot_id}",
#                             "x": rx,
#                             "y": ry,
#                             "width": rw,
#                             "height": rh,
#                             "center_x": rx + rw // 2,
#                             "center_y": ry + rh // 2,
#                             "area": cv2.contourArea(contour),
#                             "category": category
#                         }
#                         additional_slots.append(slot_info)
#                         slot_id += 1
                        
#                         # Draw rectangle and label
#                         cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), config['color'], 2)
#                         cv2.putText(result_img, slot_info['label'], (rx+5, ry+20), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
        
#         # Add additional slots if they provide more coverage
#         if len(additional_slots) > len(parking_slots):
#             parking_slots = additional_slots
    
#     # Method 3: Grid-based approach if still not enough slots
#     if len(parking_slots) < 3 and w > 100 and h > 100:  # Only if the area is significant
#         print(f"Using grid approach for {category}...")
        
#         # Reset the parking slots list
#         parking_slots = []
#         slot_id = 1
        
#         # Estimate grid dimensions based on area size
#         avg_slot_width = 80
#         avg_slot_height = 60
        
#         num_cols = max(1, w // avg_slot_width)
#         num_rows = max(1, h // avg_slot_height)
        
#         # Calculate cell dimensions
#         cell_width = w // num_cols
#         cell_height = h // num_rows
        
#         # Create a grid of cells
#         for row in range(num_rows):
#             for col in range(num_cols):
#                 rx = x + col * cell_width
#                 ry = y + row * cell_height
                
#                 # Check if this area actually contains colored pixels
#                 roi_mask = color_mask[ry:ry+cell_height, rx:rx+cell_width]
#                 if np.sum(roi_mask) > cell_width * cell_height * 0.1:  # At least 10% colored
#                     slot_info = {
#                         "id": slot_id,
#                         "label": f"{config['prefix']}{slot_id}",
#                         "x": rx,
#                         "y": ry,
#                         "width": cell_width,
#                         "height": cell_height,
#                         "center_x": rx + cell_width // 2,
#                         "center_y": ry + cell_height // 2,
#                         "area": cell_width * cell_height,
#                         "category": category,
#                         "row": row + 1,
#                         "column": col + 1
#                     }
#                     parking_slots.append(slot_info)
#                     slot_id += 1
                    
#                     # Draw rectangle and label
#                     cv2.rectangle(result_img, (rx, ry), (rx+cell_width, ry+cell_height), config['color'], 2)
#                     cv2.putText(result_img, slot_info['label'], (rx+5, ry+20), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
    
#     return parking_slots

# def label_parking_slots_sequential_improved(all_slots, image_path, visualize=True):
#     """
#     Assign sequential labels to detected parking slots with improved visualization
    
#     Args:
#         all_slots (dict): Dictionary containing slots by category
#         image_path (str): Path to the original image
#         visualize (bool): Whether to visualize the results
    
#     Returns:
#         tuple: (result_image, flat_slots_list)
#     """
#     # Flatten all slots into a single list
#     flat_slots = []
#     for category, slots in all_slots.items():
#         flat_slots.extend(slots)
    
#     if not flat_slots:
#         print("No slots to label")
#         return None, []
    
#     # Load the image for visualization
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     # Create visualization images
#     result_img = image.copy()
#     enlarged_img = cv2.resize(image.copy(), (image.shape[1]*2, image.shape[0]*2))
#     schematic_img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255  # White background
    
#     # Extract coordinates for sorting (left-to-right, top-to-bottom)
#     slot_coordinates = [(slot['id'], slot['x'], slot['y'], slot['category']) for slot in flat_slots]
#     slot_coordinates.sort(key=lambda coord: (coord[2], coord[1]))
    
#     # Assign sequential labels while preserving category information
#     labeled_slots = []
#     for i, (slot_id, _, _, category) in enumerate(slot_coordinates):
#         # Find the original slot by ID and category
#         for slot in flat_slots:
#             if slot['id'] == slot_id and slot['category'] == category:
#                 # Create a copy of the slot with the sequential label added
#                 labeled_slot = slot.copy()
#                 labeled_slot['sequential_label'] = f"S{i + 1}"  # Sequential label
#                 labeled_slots.append(labeled_slot)
                
#                 # Get color for this category
#                 color_map = {
#                     'Entry': (0, 0, 255),      # Red
#                     'Accessible': (128, 0, 128), # Purple
#                     'Reservation': (0, 255, 255), # Yellow
#                     'Regular': (255, 0, 0)     # Blue
#                 }
#                 color = color_map.get(category, (0, 255, 0))
                
#                 # Method 1: Original image with smaller font
#                 text_pos = (slot['x'] + 5, slot['y'] + 20)
#                 cv2.putText(result_img, f"{labeled_slot['label']}/{labeled_slot['sequential_label']}", text_pos,
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 cv2.rectangle(result_img, 
#                              (slot['x'], slot['y']), 
#                              (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                              color, 1)
                
#                 # Method 2: Enlarged image
#                 cv2.putText(enlarged_img, f"{labeled_slot['label']}/{labeled_slot['sequential_label']}", 
#                             (slot['x']*2 + 10, slot['y']*2 + 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#                 cv2.rectangle(enlarged_img, 
#                              (slot['x']*2, slot['y']*2), 
#                              ((slot['x'] + slot['width'])*2, (slot['y'] + slot['height'])*2), 
#                              color, 2)
                
#                 # Method 3: Schematic view with normalized spacing
#                 scale_x = 1800 / image.shape[1]
#                 scale_y = 1300 / image.shape[0]
                
#                 sch_x = int(slot['x'] * scale_x) + 100
#                 sch_y = int(slot['y'] * scale_y) + 100
#                 sch_w = max(int(slot['width'] * scale_x), 40)
#                 sch_h = max(int(slot['height'] * scale_y), 40)
                
#                 cv2.rectangle(schematic_img, 
#                              (sch_x, sch_y), 
#                              (sch_x + sch_w, sch_y + sch_h), 
#                              color, 2)
#                 cv2.putText(schematic_img, labeled_slot['sequential_label'], 
#                             (sch_x + sch_w//4, sch_y + sch_h//2),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                 break
    
#     # Create a lookup table/index for easier verification
#     lookup_img = np.ones((1000, 1200, 3), dtype=np.uint8) * 255
#     columns = 4
#     rows = (len(labeled_slots) // columns) + (1 if len(labeled_slots) % columns > 0 else 0)
    
#     for i, slot in enumerate(labeled_slots):
#         row = i // columns
#         col = i % columns
        
#         x = col * 300 + 20
#         y = row * 25 + 40
        
#         color_map = {
#             'Entry': (0, 0, 255),
#             'Accessible': (128, 0, 128),
#             'Reservation': (0, 255, 255),
#             'Regular': (255, 0, 0)
#         }
#         color = color_map.get(slot['category'], (0, 0, 0))
        
#         text = f"{slot['sequential_label']}: {slot['label']} ({slot['category']}) at ({slot['x']},{slot['y']})"
#         cv2.putText(lookup_img, text, (x, y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(20, 16))
        
#         plt.subplot(2, 2, 1)
#         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#         plt.title(f"Original with Labels (S1-S{len(labeled_slots)})")
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
#         plt.savefig("parking_visualization_multi_color.png", dpi=300, bbox_inches='tight')
#         plt.show()
    
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
#         <title>Multi-Color Parking Slot Visualization</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
#             .container {{ display: flex; flex-direction: column; }}
#             .image-container {{ position: relative; margin-bottom: 20px; 
#                               width: {img_width}px; height: {img_height}px; }}
#             .image-container img {{ width: 100%; height: 100%; }}
#             .slot {{ position: absolute; display: flex; justify-content: center; align-items: center; }}
#             .slot-label {{ font-weight: bold; background-color: rgba(255,255,255,0.8); 
#                          padding: 2px; border-radius: 3px; font-size: 10px; }}
#             .entry {{ border: 2px solid red; }}
#             .accessible {{ border: 2px solid purple; }}
#             .reservation {{ border: 2px solid #FFD700; }}
#             .regular {{ border: 2px solid blue; }}
#             .controls {{ margin-bottom: 20px; }}
#             .slot-table {{ border-collapse: collapse; width: 100%; max-width: 1200px; }}
#             .slot-table th, .slot-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
#             .slot-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
#             .slot-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
#             .highlight {{ background-color: yellow !important; }}
#             .search-container {{ margin-bottom: 10px; }}
#             .category-filter {{ margin-bottom: 10px; }}
#             .category-stats {{ margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
#         </style>
#     </head>
#     <body>
#         <h1>Multi-Color Parking Slot Visualization</h1>
        
#         <div class="category-stats">
#             <h3>Parking Statistics by Category:</h3>
#     '''
    
#     # Add category statistics
#     category_counts = {}
#     for slot in labeled_slots:
#         category = slot['category']
#         category_counts[category] = category_counts.get(category, 0) + 1
    
#     for category, count in category_counts.items():
#         color_map = {
#             'Entry': 'red',
#             'Accessible': 'purple', 
#             'Reservation': '#FFD700',
#             'Regular': 'blue'
#         }
#         color = color_map.get(category, 'black')
#         html_content += f'<span style="color: {color}; font-weight: bold;">{category}: {count} slots</span> | '
    
#     html_content += f'<span style="font-weight: bold;">Total: {len(labeled_slots)} slots</span>'
#     html_content += '''
#         </div>
        
#         <div class="controls">
#             <div class="search-container">
#                 <label for="slot-search">Search for slot: </label>
#                 <input type="text" id="slot-search" placeholder="Enter slot label (e.g. S1, P1, E1)">
#                 <button onclick="searchSlot()">Find</button>
#             </div>
            
#             <div class="category-filter">
#                 <label>Filter by category: </label>
#                 <select id="category-filter" onchange="filterByCategory()">
#                     <option value="all">All Categories</option>
#                     <option value="Entry">Entry Slots</option>
#                     <option value="Accessible">Accessible Slots</option>
#                     <option value="Reservation">Reservation Slots</option>
#                     <option value="Regular">Regular Slots</option>
#                 </select>
#             </div>
            
#             <label for="label-size">Label Size: </label>
#             <input type="range" id="label-size" min="8" max="20" value="10" 
#                    oninput="updateLabelSize(this.value)">
#             <span id="size-value">10px</span>
            
#             <button onclick="toggleLabels()">Toggle Labels</button>
#         </div>
        
#         <div class="container">
#             <div class="image-container">
#                 <img src="data:image/jpeg;base64,PLACEHOLDER_FOR_BASE64_IMAGE" alt="Parking Lot">
#     '''
    
#     # Add each parking slot as a div
#     for slot in labeled_slots:
#         category_class = slot['category'].lower()
#         html_content += f'''
#                 <div class="slot {category_class}" id="{slot['sequential_label']}" data-category="{slot['category']}"
#                      style="left: {slot['x']}px; top: {slot['y']}px; 
#                             width: {slot['width']}px; height: {slot['height']}px;">
#                     <span class="slot-label">{slot['sequential_label']}<br>{slot['label']}</span>
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
#                         <th>Sequential</th>
#                         <th>Category Label</th>
#                         <th>Category</th>
#                         <th>X Position</th>
#                         <th>Y Position</th>
#                         <th>Width</th>
#                         <th>Height</th>
#                         <th>Area</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#     '''
    
# # Add table rows for each slot
#     for slot in labeled_slots:
#         # Define color mapping outside the f-string
#         color_map = {'Entry': 'red', 'Accessible': 'purple', 'Reservation': '#B8860B', 'Regular': 'blue'}
#         slot_color = color_map.get(slot['category'], 'black')
        
#         html_content += f'''
#                     <tr id="row-{slot['sequential_label']}" data-category="{slot['category']}">
#                         <td>{slot['sequential_label']}</td>
#                         <td>{slot['label']}</td>
#                         <td style="color: {slot_color}">{slot['category']}</td>
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
#                     // Try searching by category label
#                     let found = false;
#                     slots.forEach(slotEl => {
#                         const label = slotEl.querySelector('.slot-label').textContent;
#                         if (label.includes(searchValue)) {
#                             slotEl.style.backgroundColor = 'rgba(255, 255, 0, 0.5)';
#                             slotEl.style.zIndex = 100;
#                             slotEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
#                             found = true;
#                         }
#                     });
                    
#                     if (!found) {
#                         alert('Slot ' + searchValue + ' not found!');
#                     }
#                 }
#             }
            
# function filterByCategory() {
#                 const selectedCategory = document.getElementById('category-filter').value;
#                 const slots = document.querySelectorAll('.slot');
#                 const rows = document.querySelectorAll('.slot-table tbody tr');
                
#                 slots.forEach(slot => {
#                     if (selectedCategory === 'all' || slot.dataset.category === selectedCategory) {
#                         slot.style.display = 'flex';
#                     } else {
#                         slot.style.display = 'none';
#                     }
#                 });
                
#                 rows.forEach(row => {
#                     if (selectedCategory === 'all' || row.dataset.category === selectedCategory) {
#                         row.style.display = '';
#                     } else {
#                         row.style.display = 'none';
#                     }
#                 });
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
            
#             // Add hover effects for better interactivity
#             document.addEventListener('DOMContentLoaded', function() {
#                 const slots = document.querySelectorAll('.slot');
#                 const rows = document.querySelectorAll('.slot-table tbody tr');
                
#                 // Add hover effect to slots
#                 slots.forEach(slot => {
#                     slot.addEventListener('mouseenter', function() {
#                         this.style.backgroundColor = 'rgba(0, 255, 0, 0.3)';
#                         this.style.transform = 'scale(1.05)';
#                         this.style.transition = 'all 0.2s ease';
                        
#                         // Highlight corresponding table row
#                         const slotId = this.id;
#                         const correspondingRow = document.getElementById('row-' + slotId);
#                         if (correspondingRow) {
#                             correspondingRow.style.backgroundColor = '#e6f3ff';
#                         }
#                     });
                    
#                     slot.addEventListener('mouseleave', function() {
#                         if (!this.style.backgroundColor.includes('255, 255, 0')) { // Don't reset if highlighted by search
#                             this.style.backgroundColor = 'transparent';
#                         }
#                         this.style.transform = 'scale(1)';
                        
#                         // Remove highlight from table row
#                         const slotId = this.id;
#                         const correspondingRow = document.getElementById('row-' + slotId);
#                         if (correspondingRow && !correspondingRow.classList.contains('highlight')) {
#                             correspondingRow.style.backgroundColor = '';
#                         }
#                     });
                    
#                     // Click to highlight permanently
#                     slot.addEventListener('click', function() {
#                         // Remove previous permanent highlights
#                         slots.forEach(s => {
#                             if (s !== this) {
#                                 s.style.backgroundColor = 'transparent';
#                                 s.style.zIndex = 1;
#                             }
#                         });
#                         rows.forEach(r => r.classList.remove('highlight'));
                        
#                         // Highlight this slot
#                         this.style.backgroundColor = 'rgba(255, 255, 0, 0.5)';
#                         this.style.zIndex = 100;
                        
#                         // Highlight corresponding table row
#                         const slotId = this.id;
#                         const correspondingRow = document.getElementById('row-' + slotId);
#                         if (correspondingRow) {
#                             correspondingRow.classList.add('highlight');
#                             correspondingRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
#                         }
#                     });
#                 });
                
#                 // Add click effect to table rows
#                 rows.forEach(row => {
#                     row.addEventListener('click', function() {
#                         // Remove previous highlights
#                         slots.forEach(s => {
#                             s.style.backgroundColor = 'transparent';
#                             s.style.zIndex = 1;
#                         });
#                         rows.forEach(r => r.classList.remove('highlight'));
                        
#                         // Highlight this row
#                         this.classList.add('highlight');
                        
#                         // Find and highlight corresponding slot
#                         const rowId = this.id.replace('row-', '');
#                         const correspondingSlot = document.getElementById(rowId);
#                         if (correspondingSlot) {
#                             correspondingSlot.style.backgroundColor = 'rgba(255, 255, 0, 0.5)';
#                             correspondingSlot.style.zIndex = 100;
#                             correspondingSlot.scrollIntoView({ behavior: 'smooth', block: 'center' });
#                         }
#                     });
#                 });
#             });
#         </script>
#     </body>
#     </html>
#     '''
    
#     # Save the HTML file
#     with open("multi_color_parking_visualization.html", "w") as f:
#         f.write(html_content)
    
#     print("Interactive HTML visualization saved as 'multi_color_parking_visualization.html'")
#     print("Note: The image placeholder needs to be replaced with the actual base64 encoded image.")

# def save_slots_to_csv(all_slots, filename="multi_color_parking_slots.csv"):
#     """
#     Save all parking slots data to a CSV file
    
#     Args:
#         all_slots (dict): Dictionary containing slots by category
#         filename (str): Output CSV filename
#     """
#     # Flatten all slots into a single list
#     flat_slots = []
#     for category, slots in all_slots.items():
#         flat_slots.extend(slots)
    
#     if not flat_slots:
#         print("No slots to save")
#         return
    
#     # Sort slots by position (top-to-bottom, left-to-right)
#     flat_slots.sort(key=lambda slot: (slot['y'], slot['x']))
    
#     # Add sequential labels
#     for i, slot in enumerate(flat_slots):
#         slot['sequential_label'] = f"S{i + 1}"
    
#     # Save to CSV
#     fieldnames = ["sequential_label", "category_label", "category", "x", "y", "width", "height", "area", "center_x", "center_y"]
    
#     with open(filename, "w", newline="") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
        
#         for slot in flat_slots:
#             writer.writerow({
#                 "sequential_label": slot.get('sequential_label', ''),
#                 "category_label": slot.get('label', ''),
#                 "category": slot['category'],
#                 "x": slot['x'],
#                 "y": slot['y'],
#                 "width": slot['width'],
#                 "height": slot['height'],
#                 "area": slot['area'],
#                 "center_x": slot['center_x'],
#                 "center_y": slot['center_y']
#             })
    
#     print(f"Slot data saved to '{filename}'")
#     return flat_slots

# def main():
#     """
#     Main function to process multi-color parking slot detection
#     """
#     parser = argparse.ArgumentParser(description='Process parking lot images with multi-color slot detection')
#     parser.add_argument('--image', type=str, help='Path to the parking lot image')
#     parser.add_argument('--address', type=str, help='Address to look for in the Address folder')
    
#     args = parser.parse_args()
    
#     # Determine image path
#     if args.image:
#         image_path = args.image
#     elif args.address:
#         # You can implement address-based lookup here if needed
#         print(f"Address-based lookup not implemented yet for: {args.address}")
#         sys.exit(1)
#     else:
#         # Default image path
#         image_path = "Modified_Parking_Lot.png"
    
#     # Check if image exists
#     if not os.path.exists(image_path):
#         print(f"Error: Image not found: {image_path}")
#         sys.exit(1)
    
#     try:
#         print(f"Processing image: {image_path}")
        
#         # Detect parking slots for all colors
#         all_slots = detect_parking_slots_all_colors(image_path, visualize=True)
        
#         # Calculate total slots
#         total_slots = sum(len(slots) for slots in all_slots.values())
        
#         if total_slots == 0:
#             print("No parking slots detected in the image")
#             sys.exit(1)
        
#         # Label slots with sequential labels and improved visualization
#         result_img, labeled_slots = label_parking_slots_sequential_improved(all_slots, image_path, visualize=True)
        
#         if result_img is not None and labeled_slots:
#             # Generate interactive HTML visualization
#             generate_interactive_html(labeled_slots, image_path)
            
#             # Save slot data to CSV
#             csv_filename = f"parking_slots_{os.path.splitext(os.path.basename(image_path))[0]}.csv"
#             save_slots_to_csv(all_slots, csv_filename)
            
#             # Print summary
#             print("\n" + "="*50)
#             print("PARKING SLOT DETECTION SUMMARY")
#             print("="*50)
            
#             for category, slots in all_slots.items():
#                 if slots:
#                     print(f"{category} slots: {len(slots)}")
            
#             print(f"Total parking slots detected: {total_slots}")
#             print("="*50)
            
#             return total_slots
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

def label_parking_slots_sequential_improved(slots, image_path, visualize=True):
    """
    Assign sequential labels (P1, P2, P3...) to detected parking slots
    with improved visualization
    
    Args:
        slots (list): List of dictionaries containing slot information
        image_path (str): Path to the original image
        visualize (bool): Whether to visualize the results
    
    Returns:
        numpy.ndarray: Image with labeled parking slots
    """
    if not slots:
        print("No slots to label")
        return None
    
    # Load the image for visualization
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create two copies for different visualization options
    result_img = image.copy()
    enlarged_img = cv2.resize(image.copy(), (image.shape[1]*2, image.shape[0]*2))
    schematic_img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255  # White background
    
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
            labeled_slots.append(labeled_slot)
            
            # Method 1: Original image with smaller font
            text_pos = (slot['x'] + 5, slot['y'] + 20)
            cv2.putText(result_img, labeled_slot['label'], text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.rectangle(result_img, 
                         (slot['x'], slot['y']), 
                         (slot['x'] + slot['width'], slot['y'] + slot['height']), 
                         (0, 255, 0), 1)
            
            # Method 2: Enlarged image
            cv2.putText(enlarged_img, labeled_slot['label'], 
                        (slot['x']*2 + 10, slot['y']*2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(enlarged_img, 
                         (slot['x']*2, slot['y']*2), 
                         ((slot['x'] + slot['width'])*2, (slot['y'] + slot['height'])*2), 
                         (0, 255, 0), 2)
            
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
                         (0, 0, 0), 2)
            cv2.putText(schematic_img, labeled_slot['label'], 
                        (sch_x + sch_w//4, sch_y + sch_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Create a lookup table/index for easier verification
    lookup_img = np.ones((800, 800, 3), dtype=np.uint8) * 255
    columns = 5
    rows = (len(labeled_slots) // columns) + (1 if len(labeled_slots) % columns > 0 else 0)
    
    for i, slot in enumerate(labeled_slots):
        row = i // columns
        col = i % columns
        
        x = col * 160 + 20
        y = row * 30 + 40
        
        text = f"{slot['label']}: (x={slot['x']}, y={slot['y']})"
        cv2.putText(lookup_img, text, (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    # Visualization
    if visualize:
        plt.figure(figsize=(18, 14))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original with Labels ({len(labeled_slots)} slots total)")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(enlarged_img, cv2.COLOR_BGR2RGB))
        plt.title("Enlarged View (2x)")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(schematic_img, cv2.COLOR_BGR2RGB))
        plt.title("Schematic View")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(lookup_img, cv2.COLOR_BGR2RGB))
        plt.title("Label Index")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("parking_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Return both the labeled image and labeled slots data
    return result_img, labeled_slots

def generate_interactive_html(labeled_slots, image_path):
    """
    Generate an interactive HTML file to better visualize the parking slots
    
    Args:
        labeled_slots (list): List of dictionaries containing labeled slot information
        image_path (str): Path to the original image
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
        <title>Parking Slot Visualization</title>
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
            .controls {{ margin-bottom: 20px; }}
            .slot-table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
            .slot-table th, .slot-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .slot-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .slot-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
            .highlight {{ background-color: yellow !important; }}
            .search-container {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Parking Slot Visualization</h1>
        
        <div class="controls">
            <div class="search-container">
                <label for="slot-search">Search for slot: </label>
                <input type="text" id="slot-search" placeholder="Enter slot label (e.g. P1)">
                <button onclick="searchSlot()">Find</button>
            </div>
            
            <label for="label-size">Label Size: </label>
            <input type="range" id="label-size" min="8" max="24" value="12" 
                   oninput="updateLabelSize(this.value)">
            <span id="size-value">12px</span>
            
            <button onclick="toggleLabels()">Toggle Labels</button>
        </div>
        
        <div class="container">
            <div class="image-container">
                <img src="data:image/jpeg;base64,PLACEHOLDER_FOR_BASE64_IMAGE" alt="Parking Lot">
    '''
    
    # Add each parking slot as a div
    for slot in labeled_slots:
        html_content += f'''
                <div class="slot" id="{slot['label']}" 
                     style="left: {slot['x']}px; top: {slot['y']}px; 
                            width: {slot['width']}px; height: {slot['height']}px;">
                    <span class="slot-label">{slot['label']}</span>
                </div>
        '''
    
    html_content += '''
            </div>
            
            <h2>Parking Slot Data</h2>
            <div class="search-container">
                <label for="table-search">Filter table: </label>
                <input type="text" id="table-search" placeholder="Filter by any column" 
                       oninput="filterTable()">
            </div>
            
            <table class="slot-table" id="slot-table">
                <thead>
                    <tr>
                        <th>Label</th>
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
                    <tr id="row-{slot['label']}">
                        <td>{slot['label']}</td>
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
                const slots = document.querySelectorAll('.slot');
                slots.forEach(slot => {
                    slot.style.backgroundColor = 'transparent';
                    slot.style.zIndex = 1;
                });
                
                const rows = document.querySelectorAll('.slot-table tr');
                rows.forEach(row => {
                    row.classList.remove('highlight');
                });
                
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
    
    print("Interactive HTML visualization saved as 'parking_visualization.html'")
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
        image_path = "Screenshot (4).png"
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
        
        # Then, label slots with sequential labels and improved visualization
        result = label_parking_slots_sequential_improved(slots, image_path, visualize=True)
        
        if result is not None:
            labeled_image, labeled_slots = result
            
            # Generate interactive HTML visualization
            generate_interactive_html(labeled_slots, image_path)
            
            # Print information about each labeled slot
            print(f"\nLabeled {len(labeled_slots)} parking slots")
            
            # Save slot data to CSV for easy reference
            import csv
            csv_filename = f"parking_slots_{args.address if args.address else 'default'}.csv"
            with open(csv_filename, "w", newline="") as csvfile:
                fieldnames = ["label", "type", "id", "x", "y", "width", "height", "area"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for slot in labeled_slots:
                    writer.writerow({
                        "label": slot["label"],
                        "type": slot.get("type", "Regular"),
                        "id": slot["id"],
                        "x": slot["x"],
                        "y": slot["y"],
                        "width": slot["width"],
                        "height": slot["height"],
                        "area": slot["area"]
                    })
            
            print(f"Slot data saved to '{csv_filename}'")
            print("Processing completed successfully!")
            
            # Return the total number of detected parking spaces
            total_spaces = len(labeled_slots)
            print(f"Total detected parking spaces: {total_spaces}")
            return total_spaces
        else:
            print("Failed to process parking slots")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    total_spaces = main()
    print(f"Final result: {total_spaces} parking spaces detected")