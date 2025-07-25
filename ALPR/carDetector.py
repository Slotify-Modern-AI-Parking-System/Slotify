# from ultralytics import YOLO
# import cv2
# import time
# import logging
# import subprocess
# import sys
# import numpy as np
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

# class OptimizedCarDetector:
#     """Optimized car detection with model caching and performance improvements"""
    
#     def __init__(self, model_path='yolov8n.pt'):
#         # Cache model in memory to avoid repeated loading
#         self.model = None
#         self.model_path = model_path
#         self.load_model()
        
#         # Optimization settings
#         self.frame_skip_counter = 0
#         self.frame_skip_interval = 2  # Process every 2nd frame
#         self.input_size = (640, 360)  # Smaller resolution for faster processing
        
#         # Pre-allocate arrays for better memory management
#         self.resized_frame = None
        
#     def load_model(self):
#         """Load and cache YOLO model in memory"""
#         try:
#             print("🔄 Loading and caching YOLO model...")
#             self.model = YOLO(self.model_path)
            
#             # Warm up the model with a dummy inference for better performance
#             dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
#             self.model(dummy_frame, conf=0.3, verbose=False)
#             print("✅ Model loaded and warmed up successfully")
            
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             raise
    
#     def should_process_frame(self):
#         """Implement frame skipping for better performance"""
#         self.frame_skip_counter += 1
#         should_process = (self.frame_skip_counter % self.frame_skip_interval) == 0
#         return should_process
    
#     def preprocess_frame(self, frame):
#         """Resize frame to smaller resolution for faster processing"""
#         # Resize to smaller resolution for faster inference
#         height, width = frame.shape[:2]
        
#         # Only resize if frame is larger than target size
#         if width > self.input_size[0] or height > self.input_size[1]:
#             self.resized_frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
#         else:
#             self.resized_frame = frame
            
#         return self.resized_frame
    
#     def scale_detections(self, detections, original_shape, processed_shape):
#         """Scale detection coordinates back to original frame size"""
#         orig_h, orig_w = original_shape[:2]
#         proc_h, proc_w = processed_shape[:2]
        
#         scale_x = orig_w / proc_w
#         scale_y = orig_h / proc_h
        
#         scaled_detections = []
#         for detection in detections:
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
                
#                 # Scale coordinates back to original size
#                 x1_scaled = int(x1 * scale_x)
#                 y1_scaled = int(y1 * scale_y)
#                 x2_scaled = int(x2 * scale_x)
#                 y2_scaled = int(y2 * scale_y)
                
#                 scaled_detections.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id])
        
#         return scaled_detections

# class CarTracker:
#     def __init__(self, persistence_threshold=15):
#         self.car_detections = defaultdict(list)
#         self.persistence_threshold = persistence_threshold
#         self.last_cleanup = time.time()
        
#     def add_detection(self, car_id, timestamp):
#         """Add a car detection with timestamp"""
#         self.car_detections[car_id].append(timestamp)
        
#         # Clean old detections every 5 seconds
#         current_time = time.time()
#         if current_time - self.last_cleanup > 5:
#             self.cleanup_old_detections(current_time)
#             self.last_cleanup = current_time
    
#     def cleanup_old_detections(self, current_time):
#         """Remove detections older than threshold + 5 seconds"""
#         cleanup_threshold = self.persistence_threshold + 5
        
#         for car_id in list(self.car_detections.keys()):
#             self.car_detections[car_id] = [
#                 t for t in self.car_detections[car_id] 
#                 if current_time - t <= cleanup_threshold
#             ]
            
#             if not self.car_detections[car_id]:
#                 del self.car_detections[car_id]
    
#     def check_persistent_cars(self, current_time):
#         """Check if any car has been present for more than threshold"""
#         persistent_cars = []
        
#         for car_id, timestamps in self.car_detections.items():
#             if len(timestamps) < 3:
#                 continue
                
#             earliest_detection = min(timestamps)
#             latest_detection = max(timestamps)
            
#             if (current_time - earliest_detection) >= self.persistence_threshold:
#                 if (current_time - latest_detection) <= 3:
#                     persistent_cars.append({
#                         'car_id': car_id,
#                         'first_seen': earliest_detection,
#                         'duration': current_time - earliest_detection,
#                         'detection_count': len(timestamps)
#                     })
        
#         return persistent_cars

# def apply_zoom(frame, zoom_factor=1.0):
#     """Apply zoom to frame - optimized version"""
#     if zoom_factor == 1.0:
#         return frame
    
#     height, width = frame.shape[:2]
    
#     if zoom_factor < 1.0:  # Zoom out
#         new_height = int(height / zoom_factor)
#         new_width = int(width / zoom_factor)
        
#         resized_frame = cv2.resize(frame, (width, height))
#         canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
#         y_offset = (new_height - height) // 2
#         x_offset = (new_width - width) // 2
        
#         canvas[y_offset:y_offset+height, x_offset:x_offset+width] = resized_frame
#         return cv2.resize(canvas, (width, height))
    
#     else:  # Zoom in
#         crop_height = int(height / zoom_factor)
#         crop_width = int(width / zoom_factor)
        
#         y_start = (height - crop_height) // 2
#         x_start = (width - crop_width) // 2
        
#         cropped = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
#         return cv2.resize(cropped, (width, height))

# def detect_cars(frame, car_detector_obj):
#     """Optimized car detection with preprocessing and caching"""
#     try:
#         # Skip frame processing if not needed
#         if not car_detector_obj.should_process_frame():
#             return []
        
#         # Preprocess frame for faster inference
#         processed_frame = car_detector_obj.preprocess_frame(frame)
        
#         # Apply zoom for better detection on processed frame
#         detection_frame = apply_zoom(processed_frame, zoom_factor=0.85)
        
#         # Run inference on smaller frame with cached model
#         results = car_detector_obj.model(detection_frame, conf=0.3, verbose=False)[0]
        
#         # Scale detections back to original frame size
#         scaled_detections = car_detector_obj.scale_detections(
#             results.boxes.data.tolist(), 
#             frame.shape, 
#             detection_frame.shape
#         )
        
#         cars = []
#         for detection in scaled_detections:
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
                
#                 # Filter for car-related classes
#                 if int(class_id) in [2, 3, 5, 7] and score > 0.3:
#                     bbox_area = (x2 - x1) * (y2 - y1)
                    
#                     # Filter out very small detections
#                     if bbox_area > 1000:
#                         cars.append({
#                             'bbox': (int(x1), int(y1), int(x2), int(y2)),
#                             'confidence': score,
#                             'class_id': int(class_id),
#                             'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
#                             'area': bbox_area
#                         })
        
#         # Sort by confidence
#         cars.sort(key=lambda x: x['confidence'], reverse=True)
#         return cars
        
#     except Exception as e:
#         logging.error(f"Car detection error: {str(e)}")
#         return []

# def get_car_id(car, existing_cars, max_distance=150):
#     """Improved car tracking based on position proximity and size similarity"""
#     car_center = car['center']
#     car_area = car['area']
    
#     best_match_id = None
#     best_distance = float('inf')
    
#     for existing_id, existing_data in existing_cars.items():
#         existing_center = existing_data['center']
#         existing_area = existing_data.get('area', car_area)
        
#         distance = ((car_center[0] - existing_center[0])**2 + 
#                    (car_center[1] - existing_center[1])**2)**0.5
        
#         area_ratio = min(car_area, existing_area) / max(car_area, existing_area)
#         score = distance - (area_ratio * 50)
        
#         if distance < max_distance and score < best_distance:
#             best_distance = score
#             best_match_id = existing_id
    
#     if best_match_id is not None:
#         return best_match_id
    
#     return max(existing_cars.keys(), default=0) + 1

# def get_class_name(class_id):
#     """Get human-readable class name"""
#     class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
#     return class_names.get(class_id, "Vehicle")

# def trigger_main_script(camera_id, zoomed_frame):
#     """Trigger the main.py script with zoomed-in frame"""
#     try:
#         logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
        
#         zoomed_in_frame = apply_zoom(zoomed_frame, zoom_factor=1.3)
#         cv2.imwrite(frame_path, zoomed_in_frame)
        
#         script_path = "main2.py"
#         result = subprocess.run([
#             sys.executable, script_path, str(camera_id), frame_path
#         ], capture_output=True, text=True, timeout=120)
        
#         if result.returncode == 0:
#             logging.info("✅ Main script execution completed successfully")
#             if result.stdout:
#                 print("📋 MAIN SCRIPT RESULT:")
#                 print(result.stdout)
#         else:
#             logging.error(f"❌ Main script execution failed: {result.stderr}")
        
#         # Clean up temporary frame
#         try:
#             import os
#             os.remove(frame_path)
#         except:
#             pass
            
#     except subprocess.TimeoutExpired:
#         logging.error("⏰ Main script execution timed out")
#     except Exception as e:
#         logging.error(f"Error triggering main script: {str(e)}")

# def main():
#     print("🚗 OPTIMIZED Enhanced Car Detection Monitor - Script 1")
#     print("=" * 60)
#     print("🚀 Performance Optimizations:")
#     print("  • Cached YOLO model in memory")
#     print("  • Frame skipping (every 2nd frame)")
#     print("  • Smaller input resolution (640x360)")
#     print("  • Pre-allocated memory buffers")
#     print("  • Optimized OpenCV operations")
    
#     try:
#         camera_id = 0
#         print(f"📹 Using camera ID: {camera_id}")
        
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
        
#         # Initialize optimized car detector (model cached in memory)
#         print("🔄 Initializing optimized car detector...")
#         car_detector_obj = OptimizedCarDetector('yolov8n.pt')
        
#         # Initialize components
#         car_tracker = CarTracker(persistence_threshold=15)
        
#         # Start camera capture with optimized settings
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
#         cap.set(cv2.CAP_PROP_FPS, 20)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         print(f"\n🚀 Starting OPTIMIZED car detection on camera {camera_id}")
#         print("Optimization Features:")
#         print("  • Model cached in memory (no reload)")
#         print("  • Processing every 2nd frame only")
#         print("  • Smaller resolution inference (640x360)")
#         print("  • Pre-allocated memory buffers")
#         print("  • Optimized coordinate scaling")
#         print("Controls:")
#         print("  • Press 'q' or ESC to quit")
#         print("  • Press 's' to show statistics")
#         print("  • Press 'r' to reset tracking")
#         print("  • Press 'p' to show performance stats")
        
#         frame_count = 0
#         existing_cars = {}
#         last_car_update = time.time()
#         triggered_cars = set()
#         detection_display_frame = None
        
#         # Performance tracking
#         processing_times = []
#         start_time = time.time()
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("Failed to read frame, retrying...")
#                 time.sleep(0.1)
#                 continue
            
#             current_time = time.time()
#             frame_count += 1
            
#             # Measure processing time for performance monitoring
#             process_start = time.time()
            
#             # Detect cars with optimized function
#             cars = detect_cars(frame, car_detector_obj)
            
#             process_end = time.time()
#             processing_times.append(process_end - process_start)
            
#             # Keep only last 100 processing times for rolling average
#             if len(processing_times) > 100:
#                 processing_times.pop(0)
            
#             # Update car tracking only when we have detections
#             if cars:
#                 current_cars = {}
#                 for car in cars:
#                     car_id = get_car_id(car, existing_cars)
#                     current_cars[car_id] = {
#                         'center': car['center'],
#                         'area': car['area'],
#                         'confidence': car['confidence'],
#                         'class_id': car['class_id']
#                     }
                    
#                     car_tracker.add_detection(car_id, current_time)
                
#                 existing_cars = current_cars
#                 last_car_update = current_time
#                 detection_display_frame = frame.copy()
                
#                 # Draw detections on display frame
#                 for car in cars:
#                     x1, y1, x2, y2 = car['bbox']
#                     confidence = car['confidence']
#                     class_name = get_class_name(car['class_id'])
                    
#                     color_map = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 255, 255), 7: (255, 255, 0)}
#                     color = color_map.get(car['class_id'], (0, 255, 0))
                    
#                     cv2.rectangle(detection_display_frame, (x1, y1), (x2, y2), color, 2)
                    
#                     label = f"{class_name} {confidence:.2f}"
#                     label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#                     cv2.rectangle(detection_display_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
#                     cv2.putText(detection_display_frame, label, (x1, y1-5), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
#             # Check for persistent cars every 3 seconds
#             if frame_count % (20 * 3) == 0:
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
                    
#                     if car_id not in triggered_cars:
#                         triggered_cars.add(car_id)
#                         logging.info(f"🎯 Car ID {car_id} persistent for {duration:.1f}s - TRIGGERING MAIN.PY")
                        
#                         import threading
#                         trigger_thread = threading.Thread(
#                             target=trigger_main_script, 
#                             args=(camera_id, frame.copy())
#                         )
#                         trigger_thread.daemon = True
#                         trigger_thread.start()
            
#             # Use detection frame if available
#             display_frame = detection_display_frame if detection_display_frame is not None else frame
            
#             # Draw status info with performance metrics
#             avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
#             fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
            
#             status_text = f"Camera {camera_id} - Cars: {len(existing_cars)} - FPS: {fps:.1f} - Proc: {avg_process_time*1000:.1f}ms"
#             cv2.putText(display_frame, status_text, (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # Draw optimization info
#             opt_text = f"OPTIMIZED: Skip={car_detector_obj.frame_skip_interval} | Size={car_detector_obj.input_size}"
#             cv2.putText(display_frame, opt_text, (10, 60), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
#             # Draw persistent car info
#             persistent_cars = car_tracker.check_persistent_cars(current_time)
#             if persistent_cars:
#                 y_offset = 90
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
#                     triggered_status = "TRIGGERED" if car_id in triggered_cars else f"{15-duration:.1f}s left"
                    
#                     text = f"Car {car_id}: {duration:.1f}s - {triggered_status}"
#                     color = (0, 255, 255) if car_id in triggered_cars else (255, 255, 0)
#                     cv2.putText(display_frame, text, (10, y_offset), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                     y_offset += 25
            
#             # Show frame
#             cv2.imshow(f"Car Entry OPTIMIZED Car Detection - Camera {camera_id}", display_frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:
#                 break
#             elif key == ord('s'):
#                 print(f"\n📊 OPTIMIZED STATISTICS at {datetime.now().strftime('%H:%M:%S')}")
#                 print(f"Active cars being tracked: {len(existing_cars)}")
#                 print(f"Cars that triggered main.py: {len(triggered_cars)}")
#                 print(f"Average processing time: {avg_process_time*1000:.2f}ms")
#                 print(f"Current FPS: {fps:.2f}")
                
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
#                 print(f"Currently persistent cars: {len(persistent_cars)}")
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
#                     count = car_info['detection_count']
#                     status = "TRIGGERED" if car_id in triggered_cars else "MONITORING"
#                     print(f"  Car {car_id}: {duration:.1f}s ({count} detections) - {status}")
                    
#             elif key == ord('r'):
#                 print("🔄 Resetting car tracking...")
#                 car_tracker = CarTracker(persistence_threshold=15)
#                 existing_cars = {}
#                 triggered_cars = set()
#                 print("✅ Tracking reset complete")
                
#             elif key == ord('p'):
#                 print(f"\n⚡ PERFORMANCE STATISTICS")
#                 print(f"Average processing time: {avg_process_time*1000:.2f}ms")
#                 print(f"Current FPS: {fps:.2f}")
#                 print(f"Frame skip interval: {car_detector_obj.frame_skip_interval}")
#                 print(f"Processing resolution: {car_detector_obj.input_size}")
#                 print(f"Total frames processed: {frame_count}")
        
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
        
#         print(f"\n🏁 OPTIMIZED Car Detection Monitor - Session Complete")
#         print(f"Total cars that triggered main.py: {len(triggered_cars)}")
#         print(f"Average processing time: {avg_process_time*1000:.2f}ms")
#         print(f"Final FPS: {fps:.2f}")
        
#     except KeyboardInterrupt:
#         print("\n⚡ System interrupted by user")
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"\n❌ Unexpected error: {str(e)}")
#         logging.error(f"Main function error: {str(e)}")
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# Code - 2 With Pause
# from ultralytics import YOLO
# import cv2
# import time
# import logging
# import subprocess
# import sys
# import numpy as np
# from datetime import datetime
# from collections import defaultdict
# import threading

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('car_detection_monitor.log'),
#         logging.StreamHandler()
#     ]
# )

# class OptimizedCarDetector:
#     """Optimized car detection with model caching and performance improvements"""
    
#     def __init__(self, model_path='yolov8n.pt'):
#         # Cache model in memory to avoid repeated loading
#         self.model = None
#         self.model_path = model_path
#         self.load_model()
        
#         # Optimization settings
#         self.frame_skip_counter = 0
#         self.frame_skip_interval = 2  # Process every 2nd frame
#         self.input_size = (640, 360)  # Smaller resolution for faster processing
        
#         # Pre-allocate arrays for better memory management
#         self.resized_frame = None
        
#     def load_model(self):
#         """Load and cache YOLO model in memory"""
#         try:
#             print("🔄 Loading and caching YOLO model...")
#             self.model = YOLO(self.model_path)
            
#             # Warm up the model with a dummy inference for better performance
#             dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
#             self.model(dummy_frame, conf=0.3, verbose=False)
#             print("✅ Model loaded and warmed up successfully")
            
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             raise
    
#     def should_process_frame(self):
#         """Implement frame skipping for better performance"""
#         self.frame_skip_counter += 1
#         should_process = (self.frame_skip_counter % self.frame_skip_interval) == 0
#         return should_process
    
#     def preprocess_frame(self, frame):
#         """Resize frame to smaller resolution for faster processing"""
#         # Resize to smaller resolution for faster inference
#         height, width = frame.shape[:2]
        
#         # Only resize if frame is larger than target size
#         if width > self.input_size[0] or height > self.input_size[1]:
#             self.resized_frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
#         else:
#             self.resized_frame = frame
            
#         return self.resized_frame
    
#     def scale_detections(self, detections, original_shape, processed_shape):
#         """Scale detection coordinates back to original frame size"""
#         orig_h, orig_w = original_shape[:2]
#         proc_h, proc_w = processed_shape[:2]
        
#         scale_x = orig_w / proc_w
#         scale_y = orig_h / proc_h
        
#         scaled_detections = []
#         for detection in detections:
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
                
#                 # Scale coordinates back to original size
#                 x1_scaled = int(x1 * scale_x)
#                 y1_scaled = int(y1 * scale_y)
#                 x2_scaled = int(x2 * scale_x)
#                 y2_scaled = int(y2 * scale_y)
                
#                 scaled_detections.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id])
        
#         return scaled_detections

# class CarTracker:
#     def __init__(self, persistence_threshold=15):
#         self.car_detections = defaultdict(list)
#         self.persistence_threshold = persistence_threshold
#         self.last_cleanup = time.time()
        
#     def add_detection(self, car_id, timestamp):
#         """Add a car detection with timestamp"""
#         self.car_detections[car_id].append(timestamp)
        
#         # Clean old detections every 5 seconds
#         current_time = time.time()
#         if current_time - self.last_cleanup > 5:
#             self.cleanup_old_detections(current_time)
#             self.last_cleanup = current_time
    
#     def cleanup_old_detections(self, current_time):
#         """Remove detections older than threshold + 5 seconds"""
#         cleanup_threshold = self.persistence_threshold + 5
        
#         for car_id in list(self.car_detections.keys()):
#             self.car_detections[car_id] = [
#                 t for t in self.car_detections[car_id] 
#                 if current_time - t <= cleanup_threshold
#             ]
            
#             if not self.car_detections[car_id]:
#                 del self.car_detections[car_id]
    
#     def check_persistent_cars(self, current_time):
#         """Check if any car has been present for more than threshold"""
#         persistent_cars = []
        
#         for car_id, timestamps in self.car_detections.items():
#             if len(timestamps) < 3:
#                 continue
                
#             earliest_detection = min(timestamps)
#             latest_detection = max(timestamps)
            
#             if (current_time - earliest_detection) >= self.persistence_threshold:
#                 if (current_time - latest_detection) <= 3:
#                     persistent_cars.append({
#                         'car_id': car_id,
#                         'first_seen': earliest_detection,
#                         'duration': current_time - earliest_detection,
#                         'detection_count': len(timestamps)
#                     })
        
#         return persistent_cars

# def apply_zoom(frame, zoom_factor=1.0):
#     """Apply zoom to frame - optimized version"""
#     if zoom_factor == 1.0:
#         return frame
    
#     height, width = frame.shape[:2]
    
#     if zoom_factor < 1.0:  # Zoom out
#         new_height = int(height / zoom_factor)
#         new_width = int(width / zoom_factor)
        
#         resized_frame = cv2.resize(frame, (width, height))
#         canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
#         y_offset = (new_height - height) // 2
#         x_offset = (new_width - width) // 2
        
#         canvas[y_offset:y_offset+height, x_offset:x_offset+width] = resized_frame
#         return cv2.resize(canvas, (width, height))
    
#     else:  # Zoom in
#         crop_height = int(height / zoom_factor)
#         crop_width = int(width / zoom_factor)
        
#         y_start = (height - crop_height) // 2
#         x_start = (width - crop_width) // 2
        
#         cropped = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
#         return cv2.resize(cropped, (width, height))

# def detect_cars(frame, car_detector_obj):
#     """Optimized car detection with preprocessing and caching"""
#     try:
#         # Skip frame processing if not needed
#         if not car_detector_obj.should_process_frame():
#             return []
        
#         # Preprocess frame for faster inference
#         processed_frame = car_detector_obj.preprocess_frame(frame)
        
#         # Apply zoom for better detection on processed frame
#         detection_frame = apply_zoom(processed_frame, zoom_factor=0.85)
        
#         # Run inference on smaller frame with cached model
#         results = car_detector_obj.model(detection_frame, conf=0.3, verbose=False)[0]
        
#         # Scale detections back to original frame size
#         scaled_detections = car_detector_obj.scale_detections(
#             results.boxes.data.tolist(), 
#             frame.shape, 
#             detection_frame.shape
#         )
        
#         cars = []
#         for detection in scaled_detections:
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
                
#                 # Filter for car-related classes
#                 if int(class_id) in [2, 3, 5, 7] and score > 0.3:
#                     bbox_area = (x2 - x1) * (y2 - y1)
                    
#                     # Filter out very small detections
#                     if bbox_area > 1000:
#                         cars.append({
#                             'bbox': (int(x1), int(y1), int(x2), int(y2)),
#                             'confidence': score,
#                             'class_id': int(class_id),
#                             'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
#                             'area': bbox_area
#                         })
        
#         # Sort by confidence
#         cars.sort(key=lambda x: x['confidence'], reverse=True)
#         return cars
        
#     except Exception as e:
#         logging.error(f"Car detection error: {str(e)}")
#         return []

# def get_car_id(car, existing_cars, max_distance=150):
#     """Improved car tracking based on position proximity and size similarity"""
#     car_center = car['center']
#     car_area = car['area']
    
#     best_match_id = None
#     best_distance = float('inf')
    
#     for existing_id, existing_data in existing_cars.items():
#         existing_center = existing_data['center']
#         existing_area = existing_data.get('area', car_area)
        
#         distance = ((car_center[0] - existing_center[0])**2 + 
#                    (car_center[1] - existing_center[1])**2)**0.5
        
#         area_ratio = min(car_area, existing_area) / max(car_area, existing_area)
#         score = distance - (area_ratio * 50)
        
#         if distance < max_distance and score < best_distance:
#             best_distance = score
#             best_match_id = existing_id
    
#     if best_match_id is not None:
#         return best_match_id
    
#     return max(existing_cars.keys(), default=0) + 1

# def get_class_name(class_id):
#     """Get human-readable class name"""
#     class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
#     return class_names.get(class_id, "Vehicle")

# def schedule_termination():
#     """Schedule script termination after 5 seconds"""
#     time.sleep(5)
#     logging.info("⏰ 5 seconds elapsed - Terminating script as requested")
#     print("🛑 Script terminating after 5 seconds delay...")
#     import os
#     os._exit(0)

# def trigger_main_script(camera_id, zoomed_frame):
#     """Trigger the main.py script with zoomed-in frame and schedule termination"""
#     try:
#         logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
        
#         zoomed_in_frame = apply_zoom(zoomed_frame, zoom_factor=1.3)
#         cv2.imwrite(frame_path, zoomed_in_frame)
        
#         script_path = "main2.py"
        
#         # Start main2.py as independent process
#         process = subprocess.Popen([
#             sys.executable, script_path, str(camera_id), frame_path
#         ])
        
#         logging.info(f"✅ Main script started with PID: {process.pid}")
        
#         # Schedule termination 5 seconds after main2.py starts
#         logging.info("⏰ Scheduling script termination in 5 seconds from main2.py start...")
#         termination_thread = threading.Thread(target=schedule_termination)
#         termination_thread.daemon = True
#         termination_thread.start()
            
#     except Exception as e:
#         logging.error(f"Error triggering main script: {str(e)}")
#         # Still schedule termination even if there's an error
#         termination_thread = threading.Thread(target=schedule_termination)
#         termination_thread.daemon = True
#         termination_thread.start()

# def main():
#     print("🚗 OPTIMIZED Enhanced Car Detection Monitor - Script 1")
#     print("=" * 60)
#     print("🚀 Performance Optimizations:")
#     print("  • Cached YOLO model in memory")
#     print("  • Frame skipping (every 2nd frame)")
#     print("  • Smaller input resolution (640x360)")
#     print("  • Pre-allocated memory buffers")
#     print("  • Optimized OpenCV operations")
#     print("  • Auto-termination after main2.py trigger + 5 seconds")
    
#     try:
#         camera_id = 1
#         print(f"📹 Using camera ID: {camera_id}")
        
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
        
#         # Initialize optimized car detector (model cached in memory)
#         print("🔄 Initializing optimized car detector...")
#         car_detector_obj = OptimizedCarDetector('yolov8n.pt')
        
#         # Initialize components
#         car_tracker = CarTracker(persistence_threshold=15)
        
#         # Start camera capture with optimized settings
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
#         cap.set(cv2.CAP_PROP_FPS, 20)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         print(f"\n🚀 Starting OPTIMIZED car detection on camera {camera_id}")
#         print("Optimization Features:")
#         print("  • Model cached in memory (no reload)")
#         print("  • Processing every 2nd frame only")
#         print("  • Smaller resolution inference (640x360)")
#         print("  • Pre-allocated memory buffers")
#         print("  • Optimized coordinate scaling")
#         print("  • Auto-termination after main2.py trigger + 5 seconds")
#         print("Controls:")
#         print("  • Press 'q' or ESC to quit")
#         print("  • Press 's' to show statistics")
#         print("  • Press 'r' to reset tracking")
#         print("  • Press 'p' to show performance stats")
        
#         frame_count = 0
#         existing_cars = {}
#         last_car_update = time.time()
#         triggered_cars = set()
#         detection_display_frame = None
        
#         # Performance tracking
#         processing_times = []
#         start_time = time.time()
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("Failed to read frame, retrying...")
#                 time.sleep(0.1)
#                 continue
            
#             current_time = time.time()
#             frame_count += 1
            
#             # Measure processing time for performance monitoring
#             process_start = time.time()
            
#             # Detect cars with optimized function
#             cars = detect_cars(frame, car_detector_obj)
            
#             process_end = time.time()
#             processing_times.append(process_end - process_start)
            
#             # Keep only last 100 processing times for rolling average
#             if len(processing_times) > 100:
#                 processing_times.pop(0)
            
#             # Update car tracking only when we have detections
#             if cars:
#                 current_cars = {}
#                 for car in cars:
#                     car_id = get_car_id(car, existing_cars)
#                     current_cars[car_id] = {
#                         'center': car['center'],
#                         'area': car['area'],
#                         'confidence': car['confidence'],
#                         'class_id': car['class_id']
#                     }
                    
#                     car_tracker.add_detection(car_id, current_time)
                
#                 existing_cars = current_cars
#                 last_car_update = current_time
#                 detection_display_frame = frame.copy()
                
#                 # Draw detections on display frame
#                 for car in cars:
#                     x1, y1, x2, y2 = car['bbox']
#                     confidence = car['confidence']
#                     class_name = get_class_name(car['class_id'])
                    
#                     color_map = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 255, 255), 7: (255, 255, 0)}
#                     color = color_map.get(car['class_id'], (0, 255, 0))
                    
#                     cv2.rectangle(detection_display_frame, (x1, y1), (x2, y2), color, 2)
                    
#                     label = f"{class_name} {confidence:.2f}"
#                     label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#                     cv2.rectangle(detection_display_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
#                     cv2.putText(detection_display_frame, label, (x1, y1-5), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
#             # Check for persistent cars every 3 seconds
#             if frame_count % (20 * 3) == 0:
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
                    
#                     if car_id not in triggered_cars:
#                         triggered_cars.add(car_id)
#                         logging.info(f"🎯 Car ID {car_id} persistent for {duration:.1f}s - TRIGGERING MAIN.PY")
                        
#                         trigger_thread = threading.Thread(
#                             target=trigger_main_script, 
#                             args=(camera_id, frame.copy())
#                         )
#                         trigger_thread.daemon = True
#                         trigger_thread.start()
            
#             # Use detection frame if available
#             display_frame = detection_display_frame if detection_display_frame is not None else frame
            
#             # Draw status info with performance metrics
#             avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
#             fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
            
#             status_text = f"Camera {camera_id} - Cars: {len(existing_cars)} - FPS: {fps:.1f} - Proc: {avg_process_time*1000:.1f}ms"
#             cv2.putText(display_frame, status_text, (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # Draw optimization info
#             opt_text = f"OPTIMIZED: Skip={car_detector_obj.frame_skip_interval} | Size={car_detector_obj.input_size} | Auto-Term: ON"
#             cv2.putText(display_frame, opt_text, (10, 60), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
#             # Draw persistent car info
#             persistent_cars = car_tracker.check_persistent_cars(current_time)
#             if persistent_cars:
#                 y_offset = 90
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
#                     triggered_status = "TRIGGERED" if car_id in triggered_cars else f"{15-duration:.1f}s left"
                    
#                     text = f"Car {car_id}: {duration:.1f}s - {triggered_status}"
#                     color = (0, 255, 255) if car_id in triggered_cars else (255, 255, 0)
#                     cv2.putText(display_frame, text, (10, y_offset), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                     y_offset += 25
            
#             # Show frame
#             cv2.imshow(f"Car Entry OPTIMIZED Car Detection - Camera {camera_id}", display_frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:
#                 break
#             elif key == ord('s'):
#                 print(f"\n📊 OPTIMIZED STATISTICS at {datetime.now().strftime('%H:%M:%S')}")
#                 print(f"Active cars being tracked: {len(existing_cars)}")
#                 print(f"Cars that triggered main.py: {len(triggered_cars)}")
#                 print(f"Average processing time: {avg_process_time*1000:.2f}ms")
#                 print(f"Current FPS: {fps:.2f}")
                
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
#                 print(f"Currently persistent cars: {len(persistent_cars)}")
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
#                     duration = car_info['duration']
#                     count = car_info['detection_count']
#                     status = "TRIGGERED" if car_id in triggered_cars else "MONITORING"
#                     print(f"  Car {car_id}: {duration:.1f}s ({count} detections) - {status}")
                    
#             elif key == ord('r'):
#                 print("🔄 Resetting car tracking...")
#                 car_tracker = CarTracker(persistence_threshold=15)
#                 existing_cars = {}
#                 triggered_cars = set()
#                 print("✅ Tracking reset complete")
                
#             elif key == ord('p'):
#                 print(f"\n⚡ PERFORMANCE STATISTICS")
#                 print(f"Average processing time: {avg_process_time*1000:.2f}ms")
#                 print(f"Current FPS: {fps:.2f}")
#                 print(f"Frame skip interval: {car_detector_obj.frame_skip_interval}")
#                 print(f"Processing resolution: {car_detector_obj.input_size}")
#                 print(f"Total frames processed: {frame_count}")
        
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
        
#         print(f"\n🏁 OPTIMIZED Car Detection Monitor - Session Complete")
#         print(f"Total cars that triggered main.py: {len(triggered_cars)}")
#         print(f"Average processing time: {avg_process_time*1000:.2f}ms")
#         print(f"Final FPS: {fps:.2f}")
        
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
import threading
import os
import atexit
import signal
import gc

# Setup logging with unique filename to avoid conflicts
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
log_filename = f'car_detection_monitor_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Global variables for proper cleanup
global_cap = None
global_model = None
script_should_exit = False

def cleanup_resources():
    """Enhanced cleanup function with better error handling"""
    global global_cap, global_model
    
    logging.info("🧹 Starting comprehensive resource cleanup...")
    
    # Force release camera with multiple attempts
    try:
        if global_cap is not None:
            for attempt in range(3):
                try:
                    global_cap.release()
                    time.sleep(0.5)  # Give time for release
                    logging.info(f"📹 Camera release attempt {attempt + 1}/3")
                except:
                    pass
            global_cap = None
            logging.info("📹 Camera released successfully")
    except Exception as e:
        logging.error(f"Error releasing camera: {e}")
    
    # Destroy all OpenCV windows multiple times
    try:
        for attempt in range(3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process window events
            time.sleep(0.2)
        logging.info("🖼️ OpenCV windows destroyed")
    except Exception as e:
        logging.error(f"Error destroying windows: {e}")
    
    # Enhanced memory cleanup
    try:
        if global_model is not None:
            del global_model
        global_model = None
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
        
        logging.info("🗑️ Memory cleaned up")
    except Exception as e:
        logging.error(f"Error cleaning up memory: {e}")
    
    # Wait for system to fully release resources
    time.sleep(1)
    logging.info("✅ Enhanced resource cleanup completed")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global script_should_exit
    logging.info(f"🛑 Received signal {signum} - initiating graceful shutdown")
    script_should_exit = True
    cleanup_resources()

# Register cleanup functions
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class OptimizedCarDetector:
    """Enhanced car detection with improved resource management"""
    
    def __init__(self, model_path='yolov8n.pt'):
        global global_model
        
        # Initialize model with better error handling
        self.model = None
        self.model_path = model_path
        self.load_model()
        
        # Store reference for cleanup
        global_model = self.model
        
        # Optimization settings
        self.frame_skip_counter = 0
        self.frame_skip_interval = 3
        self.input_size = (640, 360)
        
        # Pre-allocate arrays
        self.resized_frame = None
        
    def load_model(self):
        """Enhanced model loading with cleanup and retry logic"""
        try:
            logging.info("🔄 Loading YOLO model with enhanced error handling...")
            
            # Enhanced cleanup of existing model
            if hasattr(self, 'model') and self.model is not None:
                try:
                    del self.model
                    for _ in range(3):
                        gc.collect()
                        time.sleep(0.2)
                    logging.info("🗑️ Previous model cleaned up")
                except:
                    pass
            
            # Load model with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model = YOLO(self.model_path)
                    
                    # Enhanced warm up with error handling
                    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
                    _ = self.model(dummy_frame, conf=0.5, verbose=False)
                    
                    logging.info(f"✅ Model loaded and warmed up successfully (attempt {attempt + 1})")
                    break
                    
                except Exception as e:
                    logging.warning(f"Model loading attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
            
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            raise
    
    def should_process_frame(self):
        """Implement frame skipping"""
        self.frame_skip_counter += 1
        return (self.frame_skip_counter % self.frame_skip_interval) == 0
    
    def preprocess_frame(self, frame):
        """Resize frame for faster processing"""
        height, width = frame.shape[:2]
        
        if width > self.input_size[0] or height > self.input_size[1]:
            self.resized_frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        else:
            self.resized_frame = frame
            
        return self.resized_frame
    
    def scale_detections(self, detections, original_shape, processed_shape):
        """Scale detection coordinates back to original frame size"""
        orig_h, orig_w = original_shape[:2]
        proc_h, proc_w = processed_shape[:2]
        
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h
        
        scaled_detections = []
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                scaled_detections.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id])
        
        return scaled_detections

class CarTracker:
    def __init__(self, persistence_threshold=12):
        self.car_detections = defaultdict(list)
        self.persistence_threshold = persistence_threshold
        self.last_cleanup = time.time()
        self.start_time = time.time()
        
    def reset(self):
        """Reset tracking state"""
        self.car_detections.clear()
        self.start_time = time.time()
        self.last_cleanup = time.time()
        logging.info("🔄 Car tracker reset")
        
    def add_detection(self, car_id, timestamp):
        """Add car detection with timestamp"""
        self.car_detections[car_id].append(timestamp)
        
        current_time = time.time()
        if current_time - self.last_cleanup > 3:
            self.cleanup_old_detections(current_time)
            self.last_cleanup = current_time
    
    def cleanup_old_detections(self, current_time):
        """Remove old detections"""
        cleanup_threshold = self.persistence_threshold + 3
        
        for car_id in list(self.car_detections.keys()):
            self.car_detections[car_id] = [
                t for t in self.car_detections[car_id] 
                if current_time - t <= cleanup_threshold
            ]
            
            if not self.car_detections[car_id]:
                del self.car_detections[car_id]
    
    def check_persistent_cars(self, current_time):
        """Check for persistent cars"""
        persistent_cars = []
        
        for car_id, timestamps in self.car_detections.items():
            if len(timestamps) < 2:
                continue
                
            earliest_detection = min(timestamps)
            latest_detection = max(timestamps)
            
            duration = current_time - earliest_detection
            
            if duration >= self.persistence_threshold:
                if (current_time - latest_detection) <= 2:
                    persistent_cars.append({
                        'car_id': car_id,
                        'first_seen': earliest_detection,
                        'duration': duration,
                        'detection_count': len(timestamps)
                    })
        
        return persistent_cars

def apply_zoom(frame, zoom_factor=1.0):
    """Apply zoom to frame"""
    if zoom_factor == 1.0:
        return frame
    
    height, width = frame.shape[:2]
    
    if zoom_factor < 1.0:
        new_height = int(height / zoom_factor)
        new_width = int(width / zoom_factor)
        
        resized_frame = cv2.resize(frame, (width, height))
        canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        y_offset = (new_height - height) // 2
        x_offset = (new_width - width) // 2
        
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = resized_frame
        return cv2.resize(canvas, (width, height))
    else:
        crop_height = int(height / zoom_factor)
        crop_width = int(width / zoom_factor)
        
        y_start = (height - crop_height) // 2
        x_start = (width - crop_width) // 2
        
        cropped = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
        return cv2.resize(cropped, (width, height))

def detect_cars(frame, car_detector_obj):
    """Optimized car detection with better error handling"""
    try:
        if not car_detector_obj.should_process_frame():
            return []
        
        processed_frame = car_detector_obj.preprocess_frame(frame)
        detection_frame = apply_zoom(processed_frame, zoom_factor=0.85)
        
        results = car_detector_obj.model(detection_frame, conf=0.4, verbose=False)[0]
        
        scaled_detections = car_detector_obj.scale_detections(
            results.boxes.data.tolist(), 
            frame.shape, 
            detection_frame.shape
        )
        
        cars = []
        for detection in scaled_detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                
                if int(class_id) in [2, 3, 5, 7] and score > 0.4:
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    if bbox_area > 800:
                        cars.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': score,
                            'class_id': int(class_id),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'area': bbox_area
                        })
        
        cars.sort(key=lambda x: x['confidence'], reverse=True)
        return cars
        
    except Exception as e:
        logging.error(f"Car detection error: {str(e)}")
        return []

def get_car_id(car, existing_cars, max_distance=100):
    """Improved car tracking"""
    car_center = car['center']
    car_area = car['area']
    
    best_match_id = None
    best_distance = float('inf')
    
    for existing_id, existing_data in existing_cars.items():
        existing_center = existing_data['center']
        existing_area = existing_data.get('area', car_area)
        
        distance = ((car_center[0] - existing_center[0])**2 + 
                   (car_center[1] - existing_center[1])**2)**0.5
        
        area_ratio = min(car_area, existing_area) / max(car_area, existing_area)
        score = distance - (area_ratio * 30)
        
        if distance < max_distance and score < best_distance:
            best_distance = score
            best_match_id = existing_id
    
    if best_match_id is not None:
        return best_match_id
    
    return max(existing_cars.keys(), default=0) + 1

def get_class_name(class_id):
    """Get human-readable class name"""
    class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    return class_names.get(class_id, "Vehicle")

# def trigger_main_script(camera_id, zoomed_frame):
#     """Enhanced main script triggering with better process management"""
#     try:
#         logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
#         frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
        
#         # Save frame with error handling
#         zoomed_in_frame = apply_zoom(zoomed_frame, zoom_factor=1.3)
#         success = cv2.imwrite(frame_path, zoomed_in_frame)
        
#         if not success:
#             logging.error(f"Failed to save frame: {frame_path}")
#             return False
        
#         script_path = "main2.py"
        
#         if not os.path.exists(script_path):
#             logging.error(f"main2.py not found at: {script_path}")
#             return False
        
#         # Enhanced process management with better resource handling
#         try:
#             # Create new environment with proper settings
#             env = os.environ.copy()
#             env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
            
#             # Start main2.py with enhanced process management
#             process = subprocess.Popen([
#                 sys.executable, script_path, str(camera_id), frame_path
#             ], 
#             stdout=subprocess.PIPE, 
#             stderr=subprocess.PIPE,
#             text=True,
#             env=env,
#             start_new_session=True  # Create new process group
#             )
            
#             logging.info(f"✅ Main script started with PID: {process.pid}")
            
#             # Wait longer to ensure main2.py starts properly
#             time.sleep(3)
            
#             # Check if process is still running
#             if process.poll() is None:
#                 logging.info("✅ main2.py is running successfully")
#                 return True
#             else:
#                 stdout, stderr = process.communicate(timeout=5)
#                 logging.error(f"main2.py terminated immediately. STDOUT: {stdout}, STDERR: {stderr}")
#                 return False
                
#         except subprocess.TimeoutExpired:
#             logging.info("✅ main2.py appears to be running (timeout on communication)")
#             return True
#         except Exception as e:
#             logging.error(f"Error starting main2.py subprocess: {str(e)}")
#             return False
            
#     except Exception as e:
#         logging.error(f"Error in trigger_main_script: {str(e)}")
#         return False

def trigger_main_script(camera_id, zoomed_frame):
    """Enhanced main script triggering with proper camera release and process management"""
    try:
        logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
        # CRITICAL: Release camera resources BEFORE triggering ALPR script
        global global_cap
        if global_cap is not None:
            try:
                global_cap.release()
                time.sleep(2)  # Give camera time to fully release
                logging.info("📹 Camera released before triggering ALPR script")
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")
        
        # Destroy OpenCV windows to free resources
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(1)
        except:
            pass
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
        
        # Save frame with error handling
        zoomed_in_frame = apply_zoom(zoomed_frame, zoom_factor=1.3)
        success = cv2.imwrite(frame_path, zoomed_in_frame)
        
        if not success:
            logging.error(f"Failed to save frame: {frame_path}")
            return False
        
        script_path = "main2.py"
        
        if not os.path.exists(script_path):
            logging.error(f"main2.py not found at: {script_path}")
            return False
        
        try:
            # Create new environment with proper settings
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Start main2.py with detached process - don't wait for it
            process = subprocess.Popen([
                sys.executable, script_path, str(camera_id), frame_path
            ], 
            stdout=subprocess.DEVNULL,  # Don't capture output to avoid blocking
            stderr=subprocess.DEVNULL,  # Don't capture stderr either
            env=env,
            start_new_session=True,  # Create new process group
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Unix only
            )
            
            logging.info(f"✅ Main script started with PID: {process.pid}")
            
            # Don't wait for the process - let it run independently
            # Just verify it started successfully
            time.sleep(1)  # Brief pause to let process start
            
            # Check if process started (but don't wait for completion)
            if process.poll() is None:
                logging.info("✅ main2.py started successfully and is running")
                
                # CRITICAL: Exit the car detector script immediately after successful trigger
                # This ensures camera resources are fully released
                logging.info("🏁 Car detector script exiting to free resources for ALPR")
                global script_should_exit
                script_should_exit = True
                
                return True
            else:
                logging.error("❌ main2.py process terminated immediately")
                return False
                
        except Exception as e:
            logging.error(f"Error starting main2.py subprocess: {str(e)}")
            return False
            
    except Exception as e:
        logging.error(f"Error in trigger_main_script: {str(e)}")
        return False

def cleanup_temp_files():
    """Enhanced cleanup of temporary frame files"""
    try:
        import glob
        temp_files = glob.glob("temp_frame_*.jpg")
        for file in temp_files:
            try:
                if os.path.exists(file):
                    # Check if file is older than 30 seconds
                    if time.time() - os.path.getmtime(file) > 30:
                        os.remove(file)
                        logging.info(f"Cleaned up temp file: {file}")
            except Exception as e:
                logging.error(f"Error cleaning up {file}: {e}")
    except Exception as e:
        logging.error(f"Error during temp file cleanup: {e}")

def initialize_camera(camera_id, max_retries=5):
    """Enhanced camera initialization with better retry logic"""
    global global_cap
    
    for attempt in range(max_retries):
        try:
            logging.info(f"🔄 Camera initialization attempt {attempt + 1}/{max_retries}")
            
            # Enhanced cleanup of existing camera
            if global_cap is not None:
                try:
                    global_cap.release()
                    time.sleep(2)  # Longer wait for release
                    logging.info("📹 Previous camera instance released")
                except:
                    pass
            
            # Try different camera initialization methods
            cap = None
            
            # Method 1: Direct initialization
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    logging.info(f"✅ Camera {camera_id} opened with direct method")
                else:
                    cap.release()
                    cap = None
            except:
                cap = None
            
            # Method 2: Try with different backend
            if cap is None:
                try:
                    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                    if cap.isOpened():
                        logging.info(f"✅ Camera {camera_id} opened with V4L2 backend")
                    else:
                        cap.release()
                        cap = None
                except:
                    cap = None
            
            if cap is None:
                logging.warning(f"Camera {camera_id} not opened on attempt {attempt + 1}")
                time.sleep(3)  # Longer wait between attempts
                continue
            
            # Enhanced test read with multiple attempts
            read_attempts = 3
            ret = False
            frame = None
            
            for read_attempt in range(read_attempts):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                time.sleep(0.5)
            
            if not ret or frame is None:
                logging.warning(f"Cannot read from camera {camera_id} on attempt {attempt + 1}")
                cap.release()
                time.sleep(3)
                continue
            
            # Enhanced camera configuration
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                logging.warning("Some camera properties could not be set")
            
            global_cap = cap
            logging.info(f"✅ Camera {camera_id} initialized successfully")
            return cap
            
        except Exception as e:
            logging.error(f"Camera initialization error on attempt {attempt + 1}: {e}")
            time.sleep(3)
    
    logging.error(f"❌ Failed to initialize camera {camera_id} after {max_retries} attempts")
    return None

def main():
    global script_should_exit, global_cap
    
    print("🚗 ENHANCED Car Detection Monitor - Script 1 (RESOURCE-FIXED)")
    print("=" * 70)
    print("🔧 Enhanced Fixes Applied:")
    print("  • Comprehensive resource cleanup on exit")
    print("  • Enhanced camera initialization with multiple backends")
    print("  • Improved main2.py triggering with process groups")
    print("  • Better process management and error handling")
    print("  • Enhanced temporary file cleanup")
    print("  • Advanced memory management improvements")
    print("  • Multiple retry mechanisms")
    
    try:
        # Enhanced cleanup of old temp files
        cleanup_temp_files()
        
        camera_id = 1
        print(f"📹 Using camera ID: {camera_id}")
        
        # Enhanced camera initialization
        cap = initialize_camera(camera_id)
        if cap is None:
            logging.error("❌ Cannot initialize camera - exiting")
            return
        
        # Initialize optimized car detector
        print("🔄 Initializing enhanced car detector...")
        car_detector_obj = OptimizedCarDetector('yolov8n.pt')
        
        # Initialize car tracker
        car_tracker = CarTracker(persistence_threshold=12)
        car_tracker.reset()
        
        print(f"\n🚀 Starting enhanced car detection on camera {camera_id}")
        print("Enhanced Improvements:")
        print("  • Comprehensive resource cleanup")
        print("  • Multi-backend camera initialization")
        print("  • Enhanced main2.py triggering")
        print("  • Process group management")
        print("  • Advanced error handling")
        print("  • Multiple retry mechanisms")
        print("Controls:")
        print("  • Press 'q' or ESC to quit")
        print("  • Press 'r' to reset tracking")
        
        frame_count = 0
        existing_cars = {}
        triggered_cars = set()
        detection_display_frame = None
        
        # Performance tracking
        processing_times = []
        start_time = time.time()
        last_check_time = time.time()
        
        # Reduced max runtime for API calls
        max_runtime = 25
        
        while not script_should_exit:
            # Check for timeout
            current_time = time.time()
            if current_time - start_time > max_runtime:
                logging.info(f"⏰ Maximum runtime ({max_runtime}s) reached")
                break
            
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logging.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process detection
                process_start = time.time()
                cars = detect_cars(frame, car_detector_obj)
                process_end = time.time()
                
                processing_times.append(process_end - process_start)
                if len(processing_times) > 50:
                    processing_times.pop(0)
                
                # Update tracking
                if cars:
                    current_cars = {}
                    for car in cars:
                        car_id = get_car_id(car, existing_cars)
                        current_cars[car_id] = {
                            'center': car['center'],
                            'area': car['area'],
                            'confidence': car['confidence'],
                            'class_id': car['class_id']
                        }
                        
                        car_tracker.add_detection(car_id, current_time)
                    
                    existing_cars = current_cars
                    detection_display_frame = frame.copy()
                    
                    # Draw detections
                    for car in cars:
                        x1, y1, x2, y2 = car['bbox']
                        confidence = car['confidence']
                        class_name = get_class_name(car['class_id'])
                        
                        color_map = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 255, 255), 7: (255, 255, 0)}
                        color = color_map.get(car['class_id'], (0, 255, 0))
                        
                        cv2.rectangle(detection_display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(detection_display_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Check for persistent cars
                if frame_count % 15 == 0 or (current_time - last_check_time) > 1.0:
                    last_check_time = current_time
                    persistent_cars = car_tracker.check_persistent_cars(current_time)
                    
                    for car_info in persistent_cars:
                        car_id = car_info['car_id']
                        duration = car_info['duration']
                        
                        if car_id not in triggered_cars:
                            triggered_cars.add(car_id)
                            logging.info(f"🎯 Car ID {car_id} persistent for {duration:.1f}s - TRIGGERING MAIN2.PY")
                            
                            # Trigger main2.py and check success
                            success = trigger_main_script(camera_id, frame.copy())
                            
                            if success:
                                logging.info("✅ main2.py triggered successfully - preparing to exit")
                                # Give main2.py more time to start, then exit
                                time.sleep(5)  # Increased wait time
                                script_should_exit = True
                                break
                            else:
                                logging.error("❌ Failed to trigger main2.py - continuing detection")
                                triggered_cars.discard(car_id)
                
                # Display frame (for GUI mode)
                try:
                    display_frame = detection_display_frame if detection_display_frame is not None else frame
                    
                    # Add status text
                    avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
                    fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
                    
                    status_text = f"RESOURCE-FIXED: Cars: {len(existing_cars)} - FPS: {fps:.1f} - {avg_process_time*1000:.1f}ms"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show persistent cars
                    persistent_cars = car_tracker.check_persistent_cars(current_time)
                    if persistent_cars:
                        y_offset = 60
                        for car_info in persistent_cars:
                            car_id = car_info['car_id']
                            duration = car_info['duration']
                            triggered_status = "TRIGGERED" if car_id in triggered_cars else f"{12-duration:.1f}s left"
                            
                            text = f"Car {car_id}: {duration:.1f}s - {triggered_status}"
                            color = (0, 255, 255) if car_id in triggered_cars else (255, 255, 0)
                            cv2.putText(display_frame, text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            y_offset += 25
                    
                    cv2.imshow(f"RESOURCE-FIXED Car Detection - Camera {camera_id}", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord('r'):
                        logging.info("🔄 Resetting tracking...")
                        car_tracker.reset()
                        existing_cars = {}
                        triggered_cars = set()
                        
                except cv2.error:
                    # Running in headless mode
                    pass
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(0.1)
                continue
        
        logging.info("🏁 Detection session completed")
        
    except KeyboardInterrupt:
        logging.info("⚡ Interrupted by user")
        
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        
    finally:
        # Enhanced cleanup
        logging.info("🧹 Initiating enhanced cleanup...")
        script_should_exit = True
        cleanup_resources()

if __name__ == "__main__":
    main()