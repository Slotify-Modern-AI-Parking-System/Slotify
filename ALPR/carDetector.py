# from ultralytics import YOLO
# import cv2
# import time
# import logging
# import subprocess
# import sys
# from datetime import datetime
# from collections import defaultdict

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('car_detection_monitor.log'),
#         logging.StreamHandler()
#     ]
# )

# class CarTracker:
#     def __init__(self, persistence_threshold=30):
#         self.car_detections = defaultdict(list)  # Track detection times for each car
#         self.persistence_threshold = persistence_threshold  # 30 seconds
#         self.last_cleanup = time.time()
        
#     def add_detection(self, car_id, timestamp):
#         """Add a car detection with timestamp"""
#         self.car_detections[car_id].append(timestamp)
        
#         # Clean old detections every 10 seconds
#         current_time = time.time()
#         if current_time - self.last_cleanup > 10:
#             self.cleanup_old_detections(current_time)
#             self.last_cleanup = current_time
    
#     def cleanup_old_detections(self, current_time):
#         """Remove detections older than threshold + 10 seconds"""
#         cleanup_threshold = self.persistence_threshold + 10
        
#         for car_id in list(self.car_detections.keys()):
#             # Keep only recent detections
#             self.car_detections[car_id] = [
#                 t for t in self.car_detections[car_id] 
#                 if current_time - t <= cleanup_threshold
#             ]
            
#             # Remove empty entries
#             if not self.car_detections[car_id]:
#                 del self.car_detections[car_id]
    
#     def check_persistent_cars(self, current_time):
#         """Check if any car has been present for more than threshold"""
#         persistent_cars = []
        
#         for car_id, timestamps in self.car_detections.items():
#             if len(timestamps) < 5:  # Need at least 5 detections
#                 continue
                
#             # Check if car has been consistently detected
#             earliest_detection = min(timestamps)
#             latest_detection = max(timestamps)
            
#             # Car must have been detected over the threshold period
#             if (current_time - earliest_detection) >= self.persistence_threshold:
#                 # And must have recent detections (within last 5 seconds)
#                 if (current_time - latest_detection) <= 5:
#                     persistent_cars.append({
#                         'car_id': car_id,
#                         'first_seen': earliest_detection,
#                         'duration': current_time - earliest_detection,
#                         'detection_count': len(timestamps)
#                     })
        
#         return persistent_cars

# def detect_cars(frame, car_detector):
#     """Detect cars in the frame"""
#     try:
#         # Detect objects
#         results = car_detector(frame, conf=0.5)[0]
#         cars = []
        
#         for detection in results.boxes.data.tolist():
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
                
#                 # Filter for car-related classes (car, truck, bus, motorcycle)
#                 # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
#                 if int(class_id) in [2, 3, 5, 7] and score > 0.5:
#                     cars.append({
#                         'bbox': (int(x1), int(y1), int(x2), int(y2)),
#                         'confidence': score,
#                         'class_id': int(class_id),
#                         'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                     })
        
#         return cars
        
#     except Exception as e:
#         logging.error(f"Car detection error: {str(e)}")
#         return []

# def get_car_id(car, existing_cars, max_distance=100):
#     """Simple car tracking based on position proximity"""
#     car_center = car['center']
    
#     for existing_id, existing_center in existing_cars.items():
#         # Calculate distance between centers
#         distance = ((car_center[0] - existing_center[0])**2 + 
#                    (car_center[1] - existing_center[1])**2)**0.5
        
#         if distance < max_distance:
#             return existing_id
    
#     # New car - generate new ID
#     return max(existing_cars.keys(), default=0) + 1

# def trigger_main_script(camera_id):
#     """Trigger the main.py script"""
#     try:
#         logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
#         # Call the main.py script
#         script_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/main.py"  # The script to trigger
#         result = subprocess.run([
#             sys.executable, script_path, str(camera_id)
#         ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
#         if result.returncode == 0:
#             logging.info("✅ Main script execution completed successfully")
#             if result.stdout:
#                 print("📋 MAIN SCRIPT RESULT:")
#                 print(result.stdout)
#         else:
#             logging.error(f"❌ Main script execution failed: {result.stderr}")
            
#     except subprocess.TimeoutExpired:
#         logging.error("⏰ Main script execution timed out")
#     except Exception as e:
#         logging.error(f"Error triggering main script: {str(e)}")

# def main():
#     print("🚗 Car Detection Monitor - Script 1")
#     print("=" * 50)
    
#     try:
#         # Get camera ID
#         try:
#             camera_id = int(input("Enter camera ID (usually 0 or 1): ").strip())
#         except ValueError:
#             print("❌ Invalid camera ID! Using default camera 0")
#             camera_id = 0
        
#         # Test camera connection
#         print(f"🔄 Testing camera {camera_id} connection...")
#         cap = cv2.VideoCapture(camera_id)
#         if not cap.isOpened():
#             print(f"❌ Cannot connect to camera {camera_id}")
#             return
        
#         ret, frame = cap.read()
#         if not ret:
#             print(f"❌ Cannot read from camera {camera_id}")
#             cap.release()
#             return
        
#         cap.release()
#         print(f"✅ Camera {camera_id} connection successful")
        
#         # Load YOLO model for car detection (using YOLOv8n for general object detection)
#         print("🔄 Loading YOLO model for car detection...")
#         try:
#             car_detector = YOLO('yolov8n.pt')  # Will download if not present
#             print("✅ Car detection model loaded successfully")
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             return
        
#         # Initialize components
#         car_tracker = CarTracker(persistence_threshold=30)  # 30 seconds threshold
        
#         # Start camera capture
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for car detection
        
#         print(f"\n🚀 Starting car detection on camera {camera_id}")
#         print("Features:")
#         print("  • Continuous car detection")
#         print("  • 30-second persistence tracking")
#         print("  • Auto-trigger main.py script")
#         print("Controls:")
#         print("  • Press 'q' or ESC to quit")
#         print("  • Press 's' to show statistics")
        
#         frame_count = 0
#         existing_cars = {}  # car_id -> center position
#         last_car_update = time.time()
#         triggered_cars = set()  # Track which cars have already triggered main.py
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("Failed to read frame, retrying...")
#                 time.sleep(0.1)
#                 continue
            
#             current_time = time.time()
#             frame_count += 1
            
#             # Process every 3rd frame to reduce load
#             if frame_count % 3 == 0:
#                 # Detect cars
#                 cars = detect_cars(frame, car_detector)
                
#                 # Update car tracking
#                 current_cars = {}
#                 for car in cars:
#                     car_id = get_car_id(car, existing_cars)
#                     current_cars[car_id] = car['center']
                    
#                     # Add detection to tracker
#                     car_tracker.add_detection(car_id, current_time)
                
#                 existing_cars = current_cars
#                 last_car_update = current_time
            
#             # Check for persistent cars every 5 seconds
#             if frame_count % (15 * 5) == 0:  # Every 5 seconds at 15 FPS
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
                    
#                     # Only trigger once per car
#                     if car_id not in triggered_cars:
#                         triggered_cars.add(car_id)
#                         logging.info(f"🎯 Car ID {car_id} persistent for {duration:.1f}s - TRIGGERING MAIN.PY")
                        
#                         # Trigger main.py in separate process
#                         import threading
#                         trigger_thread = threading.Thread(
#                             target=trigger_main_script, 
#                             args=(camera_id,)
#                         )
#                         trigger_thread.daemon = True
#                         trigger_thread.start()
            
#             # Draw detections and info
#             if frame_count % 3 == 0:  # Only redraw when we process
#                 for car in cars:
#                     x1, y1, x2, y2 = car['bbox']
#                     confidence = car['confidence']
                    
#                     # Draw bounding box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
#                     # Draw confidence
#                     label = f"Car {confidence:.2f}"
#                     cv2.putText(frame, label, (x1, y1-10), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             # Draw status info
#             status_text = f"Camera {camera_id} - Cars: {len(existing_cars)} - Frame: {frame_count}"
#             cv2.putText(frame, status_text, (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Draw persistent car info
#             persistent_cars = car_tracker.check_persistent_cars(current_time)
#             if persistent_cars:
#                 y_offset = 60
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
#                     triggered_status = "TRIGGERED" if car_id in triggered_cars else "MONITORING"
                    
#                     text = f"Car {car_id}: {duration:.1f}s - {triggered_status}"
#                     color = (0, 255, 255) if car_id in triggered_cars else (255, 255, 0)
#                     cv2.putText(frame, text, (10, y_offset), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                     y_offset += 25
            
#             # Show frame
#             cv2.imshow(f"Car Detection - Camera {camera_id}", frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:  # 'q' or ESC
#                 break
#             elif key == ord('s'):  # Show statistics
#                 print(f"\n📊 STATISTICS at {datetime.now().strftime('%H:%M:%S')}")
#                 print(f"Active cars being tracked: {len(existing_cars)}")
#                 print(f"Cars that triggered main.py: {len(triggered_cars)}")
                
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
#                 print(f"Currently persistent cars: {len(persistent_cars)}")
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
#                     count = car_info['detection_count']
#                     status = "TRIGGERED" if car_id in triggered_cars else "MONITORING"
#                     print(f"  Car {car_id}: {duration:.1f}s ({count} detections) - {status}")
        
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
        
#         print(f"\n🏁 Car Detection Monitor - Session Complete")
#         print(f"Total cars that triggered main.py: {len(triggered_cars)}")
        
#     except KeyboardInterrupt:
#         print("\n⚡ System interrupted by user")
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"\n❌ Unexpected error: {str(e)}")
#         logging.error(f"Main function error: {str(e)}")
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


from ultralytics import YOLO
import cv2
import time
import logging
import subprocess
import sys
import numpy as np
from datetime import datetime
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_detection_monitor.log'),
        logging.StreamHandler()
    ]
)

class CarTracker:
    def __init__(self, persistence_threshold=15):  # Changed to 15 seconds
        self.car_detections = defaultdict(list)  # Track detection times for each car
        self.persistence_threshold = persistence_threshold  # 15 seconds
        self.last_cleanup = time.time()
        
    def add_detection(self, car_id, timestamp):
        """Add a car detection with timestamp"""
        self.car_detections[car_id].append(timestamp)
        
        # Clean old detections every 5 seconds
        current_time = time.time()
        if current_time - self.last_cleanup > 5:
            self.cleanup_old_detections(current_time)
            self.last_cleanup = current_time
    
    def cleanup_old_detections(self, current_time):
        """Remove detections older than threshold + 5 seconds"""
        cleanup_threshold = self.persistence_threshold + 5
        
        for car_id in list(self.car_detections.keys()):
            # Keep only recent detections
            self.car_detections[car_id] = [
                t for t in self.car_detections[car_id] 
                if current_time - t <= cleanup_threshold
            ]
            
            # Remove empty entries
            if not self.car_detections[car_id]:
                del self.car_detections[car_id]
    
    def check_persistent_cars(self, current_time):
        """Check if any car has been present for more than threshold"""
        persistent_cars = []
        
        for car_id, timestamps in self.car_detections.items():
            if len(timestamps) < 3:  # Need at least 3 detections (reduced from 5)
                continue
                
            # Check if car has been consistently detected
            earliest_detection = min(timestamps)
            latest_detection = max(timestamps)
            
            # Car must have been detected over the threshold period
            if (current_time - earliest_detection) >= self.persistence_threshold:
                # And must have recent detections (within last 3 seconds)
                if (current_time - latest_detection) <= 3:
                    persistent_cars.append({
                        'car_id': car_id,
                        'first_seen': earliest_detection,
                        'duration': current_time - earliest_detection,
                        'detection_count': len(timestamps)
                    })
        
        return persistent_cars

def apply_zoom(frame, zoom_factor=1.0):
    """Apply zoom to frame - zoom_factor < 1.0 zooms out, > 1.0 zooms in"""
    if zoom_factor == 1.0:
        return frame
    
    height, width = frame.shape[:2]
    
    if zoom_factor < 1.0:  # Zoom out
        # Create a larger canvas and place the frame in the center
        new_height = int(height / zoom_factor)
        new_width = int(width / zoom_factor)
        
        # Resize frame to fit in the new canvas
        resized_frame = cv2.resize(frame, (width, height))
        
        # Create black canvas
        canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        # Calculate center position
        y_offset = (new_height - height) // 2
        x_offset = (new_width - width) // 2
        
        # Place frame in center
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = resized_frame
        
        # Resize back to original dimensions
        return cv2.resize(canvas, (width, height))
    
    else:  # Zoom in
        # Calculate crop dimensions
        crop_height = int(height / zoom_factor)
        crop_width = int(width / zoom_factor)
        
        # Calculate center crop
        y_start = (height - crop_height) // 2
        x_start = (width - crop_width) // 2
        
        # Crop and resize
        cropped = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
        return cv2.resize(cropped, (width, height))

def detect_cars(frame, car_detector):
    """Detect cars in the frame with improved detection"""
    try:
        # Apply slight zoom out for better car detection (0.85 = zoom out slightly)
        detection_frame = apply_zoom(frame, zoom_factor=0.85)
        
        # Detect objects with lower confidence threshold for better detection
        results = car_detector(detection_frame, conf=0.3)[0]  # Lowered from 0.5
        cars = []
        
        for detection in results.boxes.data.tolist():
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                
                # Filter for car-related classes with improved criteria
                # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                if int(class_id) in [2, 3, 5, 7] and score > 0.3:  # Lowered threshold
                    # Calculate bounding box area to filter out very small detections
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # Filter out very small detections (likely false positives)
                    if bbox_area > 1000:  # Minimum area threshold
                        cars.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': score,
                            'class_id': int(class_id),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'area': bbox_area
                        })
        
        # Sort by confidence to prioritize better detections
        cars.sort(key=lambda x: x['confidence'], reverse=True)
        
        return cars
        
    except Exception as e:
        logging.error(f"Car detection error: {str(e)}")
        return []

def get_car_id(car, existing_cars, max_distance=150):  # Increased from 100
    """Improved car tracking based on position proximity and size similarity"""
    car_center = car['center']
    car_area = car['area']
    
    best_match_id = None
    best_distance = float('inf')
    
    for existing_id, existing_data in existing_cars.items():
        existing_center = existing_data['center']
        existing_area = existing_data.get('area', car_area)
        
        # Calculate distance between centers
        distance = ((car_center[0] - existing_center[0])**2 + 
                   (car_center[1] - existing_center[1])**2)**0.5
        
        # Calculate area similarity (0 to 1, where 1 is identical)
        area_ratio = min(car_area, existing_area) / max(car_area, existing_area)
        
        # Combined score: distance penalty + area similarity bonus
        score = distance - (area_ratio * 50)  # Area similarity reduces effective distance
        
        if distance < max_distance and score < best_distance:
            best_distance = score
            best_match_id = existing_id
    
    if best_match_id is not None:
        return best_match_id
    
    # New car - generate new ID
    return max(existing_cars.keys(), default=0) + 1

def get_class_name(class_id):
    """Get human-readable class name"""
    class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    return class_names.get(class_id, "Vehicle")

def trigger_main_script(camera_id, zoomed_frame):
    """Trigger the main.py script with zoomed-in frame"""
    try:
        logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
        # Save the zoomed frame for the main script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
        
        # Apply zoom in for better license plate detection (1.3 = zoom in)
        zoomed_in_frame = apply_zoom(zoomed_frame, zoom_factor=1.3)
        cv2.imwrite(frame_path, zoomed_in_frame)
        
        # Call the main.py script with frame path
        script_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/main2.py"
        result = subprocess.run([
            sys.executable, script_path, str(camera_id), frame_path
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logging.info("✅ Main script execution completed successfully")
            if result.stdout:
                print("📋 MAIN SCRIPT RESULT:")
                print(result.stdout)
        else:
            logging.error(f"❌ Main script execution failed: {result.stderr}")
        
        # Clean up temporary frame
        try:
            import os
            os.remove(frame_path)
        except:
            pass
            
    except subprocess.TimeoutExpired:
        logging.error("⏰ Main script execution timed out")
    except Exception as e:
        logging.error(f"Error triggering main script: {str(e)}")

def main():
    print("🚗 Enhanced Car Detection Monitor - Script 1")
    print("=" * 60)
    
    try:
        # Get camera ID
        try:
            camera_id = int(input("Enter camera ID (usually 0 or 1): ").strip())
        except ValueError:
            print("❌ Invalid camera ID! Using default camera 0")
            camera_id = 0
        
        # Test camera connection
        print(f"🔄 Testing camera {camera_id} connection...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot connect to camera {camera_id}")
            return
        
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Cannot read from camera {camera_id}")
            cap.release()
            return
        
        cap.release()
        print(f"✅ Camera {camera_id} connection successful")
        
        # Load YOLO model for car detection
        print("🔄 Loading YOLO model for car detection...")
        try:
            car_detector = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed
            print("✅ Car detection model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Initialize components
        car_tracker = CarTracker(persistence_threshold=15)  # 15 seconds threshold
        
        # Start camera capture with optimized settings
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 20)  # Higher FPS for better tracking
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"\n🚀 Starting enhanced car detection on camera {camera_id}")
        print("Enhanced Features:")
        print("  • Improved car detection with zoom-out")
        print("  • 15-second persistence tracking")
        print("  • Better tracking algorithm")
        print("  • Zoom-in frame for license plate detection")
        print("  • Size-based filtering")
        print("Controls:")
        print("  • Press 'q' or ESC to quit")
        print("  • Press 's' to show statistics")
        print("  • Press 'r' to reset tracking")
        
        frame_count = 0
        existing_cars = {}  # car_id -> {center, area, confidence}
        last_car_update = time.time()
        triggered_cars = set()
        detection_display_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            frame_count += 1
            
            # Process every 2nd frame for better real-time performance
            if frame_count % 2 == 0:
                # Detect cars
                cars = detect_cars(frame, car_detector)
                
                # Update car tracking with improved data
                current_cars = {}
                for car in cars:
                    car_id = get_car_id(car, existing_cars)
                    current_cars[car_id] = {
                        'center': car['center'],
                        'area': car['area'],
                        'confidence': car['confidence'],
                        'class_id': car['class_id']
                    }
                    
                    # Add detection to tracker
                    car_tracker.add_detection(car_id, current_time)
                
                existing_cars = current_cars
                last_car_update = current_time
                detection_display_frame = frame.copy()
                
                # Draw detections on display frame
                for car in cars:
                    x1, y1, x2, y2 = car['bbox']
                    confidence = car['confidence']
                    class_name = get_class_name(car['class_id'])
                    
                    # Draw bounding box with class-specific colors
                    color_map = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 255, 255), 7: (255, 255, 0)}
                    color = color_map.get(car['class_id'], (0, 255, 0))
                    
                    cv2.rectangle(detection_display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label = f"{class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(detection_display_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
                    cv2.putText(detection_display_frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Check for persistent cars every 3 seconds
            if frame_count % (20 * 3) == 0:  # Every 3 seconds at 20 FPS
                persistent_cars = car_tracker.check_persistent_cars(current_time)
                
                for car_info in persistent_cars:
                    car_id = car_info['car_id']
                    duration = car_info['duration']
                    
                    # Only trigger once per car
                    if car_id not in triggered_cars:
                        triggered_cars.add(car_id)
                        logging.info(f"🎯 Car ID {car_id} persistent for {duration:.1f}s - TRIGGERING MAIN.PY")
                        
                        # Trigger main.py in separate thread with current frame
                        import threading
                        trigger_thread = threading.Thread(
                            target=trigger_main_script, 
                            args=(camera_id, frame.copy())
                        )
                        trigger_thread.daemon = True
                        trigger_thread.start()
            
            # Use detection frame if available, otherwise current frame
            display_frame = detection_display_frame if detection_display_frame is not None else frame
            
            # Draw status info
            status_text = f"Camera {camera_id} - Active Cars: {len(existing_cars)} - Frame: {frame_count}"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw timer info
            timer_text = f"Trigger Timer: 15s | Detection: Enhanced"
            cv2.putText(display_frame, timer_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw persistent car info
            persistent_cars = car_tracker.check_persistent_cars(current_time)
            if persistent_cars:
                y_offset = 90
                for car_info in persistent_cars:
                    car_id = car_info['car_id']
                    duration = car_info['duration']
                    triggered_status = "TRIGGERED" if car_id in triggered_cars else f"{15-duration:.1f}s left"
                    
                    text = f"Car {car_id}: {duration:.1f}s - {triggered_status}"
                    color = (0, 255, 255) if car_id in triggered_cars else (255, 255, 0)
                    cv2.putText(display_frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
            
            # Show frame
            cv2.imshow(f"Enhanced Car Detection - Camera {camera_id}", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Show statistics
                print(f"\n📊 ENHANCED STATISTICS at {datetime.now().strftime('%H:%M:%S')}")
                print(f"Active cars being tracked: {len(existing_cars)}")
                print(f"Cars that triggered main.py: {len(triggered_cars)}")
                
                persistent_cars = car_tracker.check_persistent_cars(current_time)
                print(f"Currently persistent cars: {len(persistent_cars)}")
                
                for car_info in persistent_cars:
                    car_id = car_info['car_id']
                    duration = car_info['duration']
                    count = car_info['detection_count']
                    status = "TRIGGERED" if car_id in triggered_cars else "MONITORING"
                    print(f"  Car {car_id}: {duration:.1f}s ({count} detections) - {status}")
                    
            elif key == ord('r'):  # Reset tracking
                print("🔄 Resetting car tracking...")
                car_tracker = CarTracker(persistence_threshold=15)
                existing_cars = {}
                triggered_cars = set()
                print("✅ Tracking reset complete")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🏁 Enhanced Car Detection Monitor - Session Complete")
        print(f"Total cars that triggered main.py: {len(triggered_cars)}")
        
    except KeyboardInterrupt:
        print("\n⚡ System interrupted by user")
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logging.error(f"Main function error: {str(e)}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()