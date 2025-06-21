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



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import csv
from parking_lot import detect_parking_slots_by_color

def label_parking_slots_by_category(slots_data, image_path, visualize=True):
    """
    Create visualizations for categorized parking slots with improved labeling
    
    Args:
        slots_data (dict): Dictionary containing slots by category from detect_parking_slots_by_color
        image_path (str): Path to the original image
        visualize (bool): Whether to visualize the results
    
    Returns:
        tuple: (labeled_image, summary_stats)
    """
    if not slots_data or slots_data['total_slots'] == 0:
        print("No slots to label")
        return None, None
    
    # Load the image for visualization
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create visualization images
    result_img = image.copy()
    enlarged_img = cv2.resize(image.copy(), (image.shape[1]*2, image.shape[0]*2))
    schematic_img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255  # White background
    
    # Color mapping for categories
    category_colors = {
        'Entry': (0, 0, 255),      # Red
        'Accessible': (128, 0, 128), # Purple
        'Reservation': (0, 255, 255), # Yellow
        'Regular': (255, 0, 0)     # Blue
    }
    
    # Process each category
    all_labeled_slots = []
    
    for category, slots in slots_data['slots_by_category'].items():
        if not slots:
            continue
            
        print(f"Processing {len(slots)} {category} slots...")
        
        # Sort slots within category (left-to-right, top-to-bottom)
        slots.sort(key=lambda slot: (slot['y'], slot['x']))
        
        # Re-label slots within category
        for i, slot in enumerate(slots):
            # Update the label to be sequential within category
            prefix = slot['label'][0]  # Keep original prefix (E, A, R, P)
            slot['label'] = f"{prefix}{i + 1}"
            
            color = category_colors[category]
            
            # Method 1: Original image
            text_pos = (slot['x'] + 5, slot['y'] + 20)
            cv2.putText(result_img, slot['label'], text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(result_img, 
                         (slot['x'], slot['y']), 
                         (slot['x'] + slot['width'], slot['y'] + slot['height']), 
                         color, 2)
            
            # Method 2: Enlarged image
            cv2.putText(enlarged_img, slot['label'], 
                        (slot['x']*2 + 10, slot['y']*2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(enlarged_img, 
                         (slot['x']*2, slot['y']*2), 
                         ((slot['x'] + slot['width'])*2, (slot['y'] + slot['height'])*2), 
                         color, 2)
            
            # Method 3: Schematic view
            scale_x = 1800 / image.shape[1]
            scale_y = 1300 / image.shape[0]
            
            sch_x = int(slot['x'] * scale_x) + 100
            sch_y = int(slot['y'] * scale_y) + 100
            sch_w = max(int(slot['width'] * scale_x), 60)
            sch_h = max(int(slot['height'] * scale_y), 40)
            
            cv2.rectangle(schematic_img, 
                         (sch_x, sch_y), 
                         (sch_x + sch_w, sch_y + sch_h), 
                         color, 2)
            cv2.putText(schematic_img, slot['label'], 
                        (sch_x + sch_w//4, sch_y + sch_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            all_labeled_slots.append(slot)
    
    # Create summary statistics image
    summary_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(summary_img, "PARKING SLOT SUMMARY", (200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add category summaries
    y_pos = 100
    total_slots = 0
    
    for category, slots in slots_data['slots_by_category'].items():
        if slots:
            count = len(slots)
            total_slots += count
            color = category_colors[category]
            
            # Category name and count
            text = f"{category}: {count} slots"
            cv2.putText(summary_img, text, (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # List labels
            labels = [slot['label'] for slot in slots]
            label_text = ", ".join(labels)
            
            # Split long label text into multiple lines
            max_chars_per_line = 50
            if len(label_text) > max_chars_per_line:
                words = label_text.split(", ")
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + word + ", ") <= max_chars_per_line:
                        current_line += word + ", "
                    else:
                        if current_line:
                            lines.append(current_line.rstrip(", "))
                        current_line = word + ", "
                
                if current_line:
                    lines.append(current_line.rstrip(", "))
                
                for i, line in enumerate(lines):
                    cv2.putText(summary_img, line, (70, y_pos + 25 + i*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                y_pos += 25 + len(lines) * 20 + 10
            else:
                cv2.putText(summary_img, label_text, (70, y_pos + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                y_pos += 60
    
    # Add total
    cv2.putText(summary_img, f"TOTAL: {total_slots} slots", (50, y_pos + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Visualization
    if visualize:
        plt.figure(figsize=(20, 16))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Labeled Parking Slots (Total: {slots_data['total_slots']})")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(enlarged_img, cv2.COLOR_BGR2RGB))
        plt.title("Enlarged View (2x)")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(schematic_img, cv2.COLOR_BGR2RGB))
        plt.title("Schematic View")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(summary_img, cv2.COLOR_BGR2RGB))
        plt.title("Summary Statistics")
        plt.axis('off')
        
        # Create a detailed breakdown chart
        plt.subplot(2, 3, 5)
        categories = []
        counts = []
        colors_for_plot = []
        
        for category, slots in slots_data['slots_by_category'].items():
            if slots:
                categories.append(category)
                counts.append(len(slots))
                # Convert BGR to RGB for matplotlib
                bgr_color = category_colors[category]
                rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
                colors_for_plot.append(rgb_color)
        
        if categories:
            plt.bar(categories, counts, color=colors_for_plot)
            plt.title("Parking Slots by Category")
            plt.ylabel("Number of Slots")
            plt.xticks(rotation=45)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # Create pie chart
        plt.subplot(2, 3, 6)
        if categories:
            plt.pie(counts, labels=categories, colors=colors_for_plot, autopct='%1.1f%%')
            plt.title("Distribution of Parking Slots")
        
        plt.tight_layout()
        plt.savefig("parking_visualization_categorized.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create summary statistics
    summary_stats = {
        'total_slots': slots_data['total_slots'],
        'categories': slots_data['counts'],
        'all_slots': all_labeled_slots
    }
    
    return result_img, summary_stats

def generate_categorized_html(slots_data, image_path):
    """
    Generate an interactive HTML file for categorized parking slots
    
    Args:
        slots_data (dict): Dictionary containing slots by category
        image_path (str): Path to the original image
    """
    # Load image dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    # Color mapping for HTML
    html_colors = {
        'Entry': '#FF0000',      # Red
        'Accessible': '#800080', # Purple
        'Reservation': '#FFFF00', # Yellow
        'Regular': '#0000FF'     # Blue
    }
    
    # Create HTML content
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Categorized Parking Slot Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ display: flex; flex-direction: column; }}
            .image-container {{ position: relative; margin-bottom: 20px; 
                              width: {img_width}px; height: {img_height}px; }}
            .image-container img {{ width: 100%; height: 100%; }}
            .slot {{ position: absolute; border: 2px solid; 
                   display: flex; justify-content: center; align-items: center; }}
            .slot-label {{ font-weight: bold; 
                         background-color: rgba(255,255,255,0.8); 
                         padding: 2px; border-radius: 3px; font-size: 12px; }}
            .controls {{ margin-bottom: 20px; }}
            .category-controls {{ margin-bottom: 10px; }}
            .category-button {{ margin: 5px; padding: 8px 16px; border: none; 
                              border-radius: 4px; cursor: pointer; font-weight: bold; }}
            .summary-stats {{ background-color: #f5f5f5; padding: 15px; 
                            border-radius: 5px; margin-bottom: 20px; }}
            .slot-table {{ border-collapse: collapse; width: 100%; }}
            .slot-table th, .slot-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .slot-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .slot-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
            .highlight {{ background-color: yellow !important; }}
            .search-container {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Categorized Parking Slot Visualization</h1>
        
        <div class="summary-stats">
            <h3>Summary Statistics</h3>
            <p><strong>Total Parking Slots: {slots_data['total_slots']}</strong></p>
    '''
    
    # Add category statistics
    for category, count in slots_data['counts'].items():
        if count > 0:
            color = html_colors[category]
            slots = slots_data['slots_by_category'][category]
            labels = [slot['label'] for slot in slots]
            html_content += f'''
            <p><span style="color: {color}; font-weight: bold;">{category}:</span> {count} slots ({', '.join(labels)})</p>
            '''
    
    html_content += '''
        </div>
        
        <div class="controls">
            <div class="category-controls">
                <label>Toggle Categories: </label>
    '''
    
    # Add category toggle buttons
    for category, count in slots_data['counts'].items():
        if count > 0:
            color = html_colors[category]
            html_content += f'''
                <button class="category-button" style="background-color: {color}; color: white;" 
                        onclick="toggleCategory('{category}')">{category} ({count})</button>
            '''
    
    html_content += '''
            </div>
            
            <div class="search-container">
                <label for="slot-search">Search for slot: </label>
                <input type="text" id="slot-search" placeholder="Enter slot label (e.g. P1, A1, E1, R1)">
                <button onclick="searchSlot()">Find</button>
            </div>
            
            <label for="label-size">Label Size: </label>
            <input type="range" id="label-size" min="8" max="20" value="12" 
                   oninput="updateLabelSize(this.value)">
            <span id="size-value">12px</span>
            
            <button onclick="toggleLabels()">Toggle All Labels</button>
        </div>
        
        <div class="container">
            <div class="image-container">
                <img src="data:image/jpeg;base64,PLACEHOLDER_FOR_BASE64_IMAGE" alt="Parking Lot">
    '''
    
    # Add each parking slot as a div
    for category, slots in slots_data['slots_by_category'].items():
        if not slots:
            continue
            
        color = html_colors[category]
        for slot in slots:
            html_content += f'''
                <div class="slot {category.lower()}-slot" id="{slot['label']}" 
                     style="left: {slot['x']}px; top: {slot['y']}px; 
                            width: {slot['width']}px; height: {slot['height']}px;
                            border-color: {color};">
                    <span class="slot-label" style="color: {color};">{slot['label']}</span>
                </div>
            '''
    
    html_content += '''
            </div>
            
            <h2>Categorized Parking Slot Data</h2>
            <div class="search-container">
                <label for="table-search">Filter table: </label>
                <input type="text" id="table-search" placeholder="Filter by any column" 
                       oninput="filterTable()">
            </div>
            
            <table class="slot-table" id="slot-table">
                <thead>
                    <tr>
                        <th>Label</th>
                        <th>Category</th>
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
    for category, slots in slots_data['slots_by_category'].items():
        if not slots:
            continue
            
        for slot in slots:
            html_content += f'''
                    <tr id="row-{slot['label']}" class="{category.lower()}-row">
                        <td>{slot['label']}</td>
                        <td style="color: {html_colors[category]}; font-weight: bold;">{category}</td>
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
            let labelsVisible = true;
            let currentHighlight = null;
            
            function toggleCategory(category) {
                const slots = document.querySelectorAll('.' + category.toLowerCase() + '-slot');
                const rows = document.querySelectorAll('.' + category.toLowerCase() + '-row');
                const button = event.target;
                
                slots.forEach(slot => {
                    if (slot.style.display === 'none') {
                        slot.style.display = 'flex';
                        button.style.opacity = '1';
                    } else {
                        slot.style.display = 'none';
                        button.style.opacity = '0.5';
                    }
                });
                
                rows.forEach(row => {
                    if (row.style.display === 'none') {
                        row.style.display = 'table-row';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }
            
            function toggleLabels() {
                const labels = document.querySelectorAll('.slot-label');
                labelsVisible = !labelsVisible;
                
                labels.forEach(label => {
                    label.style.display = labelsVisible ? 'block' : 'none';
                });
            }
            
            function updateLabelSize(size) {
                const labels = document.querySelectorAll('.slot-label');
                document.getElementById('size-value').textContent = size + 'px';
                
                labels.forEach(label => {
                    label.style.fontSize = size + 'px';
                });
            }
            
            function searchSlot() {
                const searchTerm = document.getElementById('slot-search').value.toUpperCase();
                const slot = document.getElementById(searchTerm);
                
                // Remove previous highlight
                if (currentHighlight) {
                    currentHighlight.classList.remove('highlight');
                    const prevRow = document.getElementById('row-' + currentHighlight.id);
                    if (prevRow) prevRow.classList.remove('highlight');
                }
                
                if (slot) {
                    slot.classList.add('highlight');
                    slot.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    currentHighlight = slot;
                    
                    // Highlight corresponding table row
                    const row = document.getElementById('row-' + searchTerm);
                    if (row) row.classList.add('highlight');
                    
                    alert('Found slot: ' + searchTerm);
                } else {
                    alert('Slot not found: ' + searchTerm);
                }
            }
            
            function filterTable() {
                const filter = document.getElementById('table-search').value.toLowerCase();
                const table = document.getElementById('slot-table');
                const rows = table.getElementsByTagName('tr');
                
                for (let i = 1; i < rows.length; i++) {
                    const row = rows[i];
                    const cells = row.getElementsByTagName('td');
                    let found = false;
                    
                    for (let j = 0; j < cells.length; j++) {
                        if (cells[j].textContent.toLowerCase().includes(filter)) {
                            found = true;
                            break;
                        }
                    }
                    
                    row.style.display = found ? '' : 'none';
                }
            }
            
            // Add click event to table rows to highlight corresponding slot
            document.addEventListener('DOMContentLoaded', function() {
                const rows = document.querySelectorAll('#slot-table tbody tr');
                rows.forEach(row => {
                    row.addEventListener('click', function() {
                        const label = this.cells[0].textContent;
                        const slot = document.getElementById(label);
                        
                        if (slot) {
                            // Remove previous highlight
                            if (currentHighlight) {
                                currentHighlight.classList.remove('highlight');
                                const prevRow = document.getElementById('row-' + currentHighlight.id);
                                if (prevRow) prevRow.classList.remove('highlight');
                            }
                            
                            slot.classList.add('highlight');
                            slot.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            currentHighlight = slot;
                            this.classList.add('highlight');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    '''
    
    # Convert image to base64 for embedding
    import base64
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Replace placeholder with actual base64 image
    html_content = html_content.replace('PLACEHOLDER_FOR_BASE64_IMAGE', img_base64)
    
    # Save HTML file
    output_file = "categorized_parking_visualization.html"
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive HTML visualization saved as: {output_file}")

def save_categorized_csv(slots_data, output_file="categorized_parking_slots.csv"):
    """
    Save categorized parking slot data to CSV file
    
    Args:
        slots_data (dict): Dictionary containing slots by category
        output_file (str): Output CSV filename
    """
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Label', 'Category', 'X', 'Y', 'Width', 'Height', 'Area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for category, slots in slots_data['slots_by_category'].items():
            for slot in slots:
                writer.writerow({
                    'Label': slot['label'],
                    'Category': category,
                    'X': slot['x'],
                    'Y': slot['y'],
                    'Width': slot['width'],
                    'Height': slot['height'],
                    'Area': slot['area']
                })
    
    print(f"CSV data saved as: {output_file}")

def main():
    """
    Main function to run the complete categorized parking slot detection and visualization
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect and categorize parking slots by color')
    parser.add_argument('image_path', help='Path to the parking lot image')
    parser.add_argument('--no-visualize', action='store_true', help='Skip matplotlib visualization')
    parser.add_argument('--output-csv', default='parking_slots.csv', help='Output CSV filename')
    parser.add_argument('--output-html', default='parking_visualization.html', help='Output HTML filename')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Detect parking slots by color
        print("Step 1: Detecting parking slots by color...")
        slots_data = detect_parking_slots_by_color(args.image_path, visualize=not args.no_visualize)
        
        if slots_data['total_slots'] == 0:
            print("No parking slots detected. Please check your image and color ranges.")
            return
        
        # Step 2: Create detailed labels and visualizations
        print("Step 2: Creating detailed labels and visualizations...")
        labeled_image, summary_stats = label_parking_slots_by_category(
            slots_data, args.image_path, visualize=not args.no_visualize
        )
        
        # Step 3: Generate interactive HTML
        print("Step 3: Generating interactive HTML visualization...")
        generate_categorized_html(slots_data, args.image_path)
        
        # Step 4: Save CSV data
        print("Step 4: Saving CSV data...")
        save_categorized_csv(slots_data, args.output_csv)
        
        # Step 5: Print final summary
        print(f"\n{'='*70}")
        print("FINAL PARKING SLOT DETECTION SUMMARY")
        print(f"{'='*70}")
        
        print(f"Total parking slots detected: {slots_data['total_slots']}")
        print(f"\nDetailed breakdown by category:")
        print(f"{'Category':<15} {'Count':<8} {'Labels'}")
        print(f"{'-'*60}")
        
        for category, slots in slots_data['slots_by_category'].items():
            if slots:
                count = len(slots)
                labels = [slot['label'] for slot in slots]
                labels_str = ', '.join(labels)
                
                # Handle long label strings
                if len(labels_str) > 40:
                    labels_str = labels_str[:37] + "..."
                
                print(f"{category:<15} {count:<8} {labels_str}")
        
        # Print category-specific details
        print(f"\n{'='*70}")
        print("CATEGORY DETAILS")
        print(f"{'='*70}")
        
        for category, slots in slots_data['slots_by_category'].items():
            if slots:
                print(f"\n{category.upper()} PARKING SLOTS ({len(slots)} total):")
                print(f"{'Label':<8} {'Position':<12} {'Size':<10} {'Area':<8}")
                print(f"{'-'*40}")
                
                for slot in slots:
                    pos = f"({slot['x']},{slot['y']})"
                    size = f"{slot['width']}×{slot['height']}"
                    print(f"{slot['label']:<8} {pos:<12} {size:<10} {slot['area']:<8.0f}")
        
        print(f"\nFiles generated:")
        print(f"- Interactive HTML: categorized_parking_visualization.html")
        print(f"- CSV data: {args.output_csv}")
        print(f"- Visualization image: parking_visualization_categorized.png")
        
        return slots_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # If running without command line arguments, use default image
    import sys
    if len(sys.argv) == 1:
        # Default usage for testing
        image_path = "Modified_Parking_Lot.png"  # Change this to your image path
        
        print("Running with default image path. Use command line arguments for custom paths.")
        print(f"Usage: python {sys.argv[0]} <image_path> [--no-visualize] [--output-csv filename.csv]")
        print(f"Using default image: {image_path}\n")
        
        # Run with default parameters
        slots_data = detect_parking_slots_by_color(image_path, visualize=True)
        
        if slots_data['total_slots'] > 0:
            labeled_image, summary_stats = label_parking_slots_by_category(slots_data, image_path, visualize=True)
            generate_categorized_html(slots_data, image_path)
            save_categorized_csv(slots_data)
            
            # Print summary
            print(f"\nSUMMARY:")
            print(f"Total slots: {slots_data['total_slots']}")
            for category, count in slots_data['counts'].items():
                if count > 0:
                    print(f"{category}: {count} slots")
    else:
        main()