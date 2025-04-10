# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from parking_lot import detect_parking_slots_grid  # Import your existing function

# def label_parking_slots_sequential(slots, image_path, visualize=True):
#     """
#     Assign sequential labels (P1, P2, P3...) to detected parking slots
    
#     Args:
#         slots (list): List of dictionaries containing slot information
#         image_path (str): Path to the original image
#         visualize (bool): Whether to visualize the results
    
#     Returns:
#         list: Updated list of slot dictionaries with sequential labels
#     """
#     if not slots:
#         print("No slots to label")
#         return slots
    
#     # Load the image for visualization
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     # Create a copy for visualization
#     result_img = image.copy()
    
#     # Extract coordinates for sorting (left-to-right, top-to-bottom)
#     slot_coordinates = [(slot['id'], slot['x'], slot['y']) for slot in slots]
    
#     # Sort by y-coordinate first, then by x-coordinate
#     # This creates a reading order (top-to-bottom, left-to-right)
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
                
#                 # Draw the label on the image
#                 text_pos = (slot['x'] + 5, slot['y'] + 20)
#                 cv2.putText(result_img, labeled_slot['label'], text_pos,
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
#                 # Draw rectangle around the slot
#                 cv2.rectangle(result_img, 
#                              (slot['x'], slot['y']), 
#                              (slot['x'] + slot['width'], slot['y'] + slot['height']), 
#                              (0, 255, 0), 2)
#                 break
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(15, 10))
        
#         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#         plt.title(f"Parking Slots with Sequential Labels (P1-P{len(labeled_slots)})")
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.show()
    
#     return labeled_slots

# def main():
#     # Path to your image with blue-marked parking slots
#     image_path = "Screenshot (4).png"  # Change this to your image path
    
#     # First, detect parking slots using your existing function
#     slots = detect_parking_slots_grid(image_path, visualize=False)
    
#     # Then, label slots with sequential labels
#     labeled_slots = label_parking_slots_sequential(slots, image_path, visualize=True)
    
#     # Print information about each labeled slot
#     print(f"\nLabeled {len(labeled_slots)} parking slots")
    
#     if labeled_slots:
#         print("\nSlot Details:")
#         print("-" * 60)
#         print(f"{'ID':<5}{'Label':<10}{'Position (x,y)':<20}{'Size (w×h)':<15}")
#         print("-" * 60)
        
#         for slot in labeled_slots:
#             print(f"{slot['id']:<5}{slot['label']:<10}({slot['x']}, {slot['y']})  {' ':<5}{slot['width']}×{slot['height']}")
        
#         print(f"\nTotal parking slots: {len(labeled_slots)}")
#     else:
#         print("No parking slots detected. Try adjusting parameters.")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from parking_lot import detect_parking_slots_grid

def label_parking_slots_sequential_improved(slots, image_path, visualize=True):
    """
    Assign sequential labels (P1, P2, P3...) to detected parking slots
    with improved visualization
    
    Args:
        slots (list): List of dictionaries containing slot information
        image_path (str): Path to the original image
        visualize (bool): Whether to visualize the results
    
    Returns:
        list: Updated list of slot dictionaries with sequential labels
    """
    if not slots:
        print("No slots to label")
        return slots
    
    # Load the image for visualization
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create two copies for different visualization options
    result_img = image.copy()
    enlarged_img = cv2.resize(image.copy(), (image.shape[1]*2, image.shape[0]*2))
    schematic_img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255  # White background
    
    # Extract coordinates for sorting (left-to-right, top-to-bottom)
    slot_coordinates = [(slot['id'], slot['x'], slot['y']) for slot in slots]
    slot_coordinates.sort(key=lambda coord: (coord[2], coord[1]))
    
    # Assign sequential labels
    labeled_slots = []
    for i, (slot_id, _, _) in enumerate(slot_coordinates):
        # Find the original slot by ID
        for slot in slots:
            if slot['id'] == slot_id:
                # Create a copy of the slot with the sequential label added
                labeled_slot = slot.copy()
                labeled_slot['label'] = f"P{i + 1}"
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
                break
    
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
        plt.title(f"Original with Labels (P1-P{len(labeled_slots)})")
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
    
    return labeled_slots

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

def main():
    # Path to your image with blue-marked parking slots
    image_path = "Screenshot (4).png"  # Change this to your image path
    
    # First, detect parking slots using your existing function
    slots = detect_parking_slots_grid(image_path, visualize=False)
    
    # Then, label slots with sequential labels and improved visualization
    labeled_slots = label_parking_slots_sequential_improved(slots, image_path, visualize=True)
    
    # Generate interactive HTML visualization
    generate_interactive_html(labeled_slots, image_path)
    
    # Print information about each labeled slot
    print(f"\nLabeled {len(labeled_slots)} parking slots")
    
    # Save slot data to CSV for easy reference
    import csv
    with open("parking_slots.csv", "w", newline="") as csvfile:
        fieldnames = ["label", "id", "x", "y", "width", "height", "area"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for slot in labeled_slots:
            writer.writerow({
                "label": slot["label"],
                "id": slot["id"],
                "x": slot["x"],
                "y": slot["y"],
                "width": slot["width"],
                "height": slot["height"],
                "area": slot["area"]
            })
    
    print("Slot data saved to 'parking_slots.csv'")

if __name__ == "__main__":
    main()