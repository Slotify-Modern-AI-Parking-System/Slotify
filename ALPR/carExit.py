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

class OptimizedCarDetector:
    """Optimized car detection with model caching and performance improvements"""
    
    def __init__(self, model_path='yolov8n.pt'):
        # Cache model in memory to avoid repeated loading
        self.model = None
        self.model_path = model_path
        self.load_model()
        
        # Optimization settings
        self.frame_skip_counter = 0
        self.frame_skip_interval = 2  # Process every 2nd frame
        self.input_size = (640, 360)  # Smaller resolution for faster processing
        
        # Pre-allocate arrays for better memory management
        self.resized_frame = None
        
    def load_model(self):
        """Load and cache YOLO model in memory"""
        try:
            print("🔄 Loading and caching YOLO model...")
            self.model = YOLO(self.model_path)
            
            # Warm up the model with a dummy inference for better performance
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, conf=0.3, verbose=False)
            print("✅ Model loaded and warmed up successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def should_process_frame(self):
        """Implement frame skipping for better performance"""
        self.frame_skip_counter += 1
        should_process = (self.frame_skip_counter % self.frame_skip_interval) == 0
        return should_process
    
    def preprocess_frame(self, frame):
        """Resize frame to smaller resolution for faster processing"""
        # Resize to smaller resolution for faster inference
        height, width = frame.shape[:2]
        
        # Only resize if frame is larger than target size
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
                
                # Scale coordinates back to original size
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                scaled_detections.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id])
        
        return scaled_detections

class CarTracker:
    def __init__(self, persistence_threshold=15):
        self.car_detections = defaultdict(list)
        self.persistence_threshold = persistence_threshold
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
            self.car_detections[car_id] = [
                t for t in self.car_detections[car_id] 
                if current_time - t <= cleanup_threshold
            ]
            
            if not self.car_detections[car_id]:
                del self.car_detections[car_id]
    
    def check_persistent_cars(self, current_time):
        """Check if any car has been present for more than threshold"""
        persistent_cars = []
        
        for car_id, timestamps in self.car_detections.items():
            if len(timestamps) < 3:
                continue
                
            earliest_detection = min(timestamps)
            latest_detection = max(timestamps)
            
            if (current_time - earliest_detection) >= self.persistence_threshold:
                if (current_time - latest_detection) <= 3:
                    persistent_cars.append({
                        'car_id': car_id,
                        'first_seen': earliest_detection,
                        'duration': current_time - earliest_detection,
                        'detection_count': len(timestamps)
                    })
        
        return persistent_cars

def apply_zoom(frame, zoom_factor=1.0):
    """Apply zoom to frame - optimized version"""
    if zoom_factor == 1.0:
        return frame
    
    height, width = frame.shape[:2]
    
    if zoom_factor < 1.0:  # Zoom out
        new_height = int(height / zoom_factor)
        new_width = int(width / zoom_factor)
        
        resized_frame = cv2.resize(frame, (width, height))
        canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        y_offset = (new_height - height) // 2
        x_offset = (new_width - width) // 2
        
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = resized_frame
        return cv2.resize(canvas, (width, height))
    
    else:  # Zoom in
        crop_height = int(height / zoom_factor)
        crop_width = int(width / zoom_factor)
        
        y_start = (height - crop_height) // 2
        x_start = (width - crop_width) // 2
        
        cropped = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
        return cv2.resize(cropped, (width, height))

def detect_cars(frame, car_detector_obj):
    """Optimized car detection with preprocessing and caching"""
    try:
        # Skip frame processing if not needed
        if not car_detector_obj.should_process_frame():
            return []
        
        # Preprocess frame for faster inference
        processed_frame = car_detector_obj.preprocess_frame(frame)
        
        # Apply zoom for better detection on processed frame
        detection_frame = apply_zoom(processed_frame, zoom_factor=0.85)
        
        # Run inference on smaller frame with cached model
        results = car_detector_obj.model(detection_frame, conf=0.3, verbose=False)[0]
        
        # Scale detections back to original frame size
        scaled_detections = car_detector_obj.scale_detections(
            results.boxes.data.tolist(), 
            frame.shape, 
            detection_frame.shape
        )
        
        cars = []
        for detection in scaled_detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                
                # Filter for car-related classes
                if int(class_id) in [2, 3, 5, 7] and score > 0.3:
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # Filter out very small detections
                    if bbox_area > 1000:
                        cars.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': score,
                            'class_id': int(class_id),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'area': bbox_area
                        })
        
        # Sort by confidence
        cars.sort(key=lambda x: x['confidence'], reverse=True)
        return cars
        
    except Exception as e:
        logging.error(f"Car detection error: {str(e)}")
        return []

def get_car_id(car, existing_cars, max_distance=150):
    """Improved car tracking based on position proximity and size similarity"""
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
        score = distance - (area_ratio * 50)
        
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

def trigger_main_script(camera_id, zoomed_frame):
    """Trigger the main.py script with zoomed-in frame"""
    try:
        logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
        
        zoomed_in_frame = apply_zoom(zoomed_frame, zoom_factor=1.3)
        cv2.imwrite(frame_path, zoomed_in_frame)
        
        script_path = "exitDetection.py"
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
    print("🚗 OPTIMIZED Enhanced Car Detection Monitor - Script 1")
    print("=" * 60)
    print("🚀 Performance Optimizations:")
    print("  • Cached YOLO model in memory")
    print("  • Frame skipping (every 2nd frame)")
    print("  • Smaller input resolution (640x360)")
    print("  • Pre-allocated memory buffers")
    print("  • Optimized OpenCV operations")
    
    try:
        camera_id = 2
        print(f"📹 Using camera ID: {camera_id}")
        
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
        
        # Initialize optimized car detector (model cached in memory)
        print("🔄 Initializing optimized car detector...")
        car_detector_obj = OptimizedCarDetector('yolov8n.pt')
        
        # Initialize components
        car_tracker = CarTracker(persistence_threshold=15)
        
        # Start camera capture with optimized settings
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"\n🚀 Starting OPTIMIZED car detection on camera {camera_id}")
        print("Optimization Features:")
        print("  • Model cached in memory (no reload)")
        print("  • Processing every 2nd frame only")
        print("  • Smaller resolution inference (640x360)")
        print("  • Pre-allocated memory buffers")
        print("  • Optimized coordinate scaling")
        print("Controls:")
        print("  • Press 'q' or ESC to quit")
        print("  • Press 's' to show statistics")
        print("  • Press 'r' to reset tracking")
        print("  • Press 'p' to show performance stats")
        
        frame_count = 0
        existing_cars = {}
        last_car_update = time.time()
        triggered_cars = set()
        detection_display_frame = None
        
        # Performance tracking
        processing_times = []
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            frame_count += 1
            
            # Measure processing time for performance monitoring
            process_start = time.time()
            
            # Detect cars with optimized function
            cars = detect_cars(frame, car_detector_obj)
            
            process_end = time.time()
            processing_times.append(process_end - process_start)
            
            # Keep only last 100 processing times for rolling average
            if len(processing_times) > 100:
                processing_times.pop(0)
            
            # Update car tracking only when we have detections
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
                last_car_update = current_time
                detection_display_frame = frame.copy()
                
                # Draw detections on display frame
                for car in cars:
                    x1, y1, x2, y2 = car['bbox']
                    confidence = car['confidence']
                    class_name = get_class_name(car['class_id'])
                    
                    color_map = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 255, 255), 7: (255, 255, 0)}
                    color = color_map.get(car['class_id'], (0, 255, 0))
                    
                    cv2.rectangle(detection_display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(detection_display_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
                    cv2.putText(detection_display_frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Check for persistent cars every 3 seconds
            if frame_count % (20 * 3) == 0:
                persistent_cars = car_tracker.check_persistent_cars(current_time)
                
                for car_info in persistent_cars:
                    car_id = car_info['car_id']
                    duration = car_info['duration']
                    
                    if car_id not in triggered_cars:
                        triggered_cars.add(car_id)
                        logging.info(f"🎯 Car ID {car_id} persistent for {duration:.1f}s - TRIGGERING MAIN.PY")
                        
                        import threading
                        trigger_thread = threading.Thread(
                            target=trigger_main_script, 
                            args=(camera_id, frame.copy())
                        )
                        trigger_thread.daemon = True
                        trigger_thread.start()
            
            # Use detection frame if available
            display_frame = detection_display_frame if detection_display_frame is not None else frame
            
            # Draw status info with performance metrics
            avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
            fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
            
            status_text = f"Camera {camera_id} - Cars: {len(existing_cars)} - FPS: {fps:.1f} - Proc: {avg_process_time*1000:.1f}ms"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw optimization info
            opt_text = f"OPTIMIZED: Skip={car_detector_obj.frame_skip_interval} | Size={car_detector_obj.input_size}"
            cv2.putText(display_frame, opt_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
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
            cv2.imshow(f"OPTIMIZED Car EXIT Detection - Camera {camera_id}", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print(f"\n📊 OPTIMIZED STATISTICS at {datetime.now().strftime('%H:%M:%S')}")
                print(f"Active cars being tracked: {len(existing_cars)}")
                print(f"Cars that triggered main.py: {len(triggered_cars)}")
                print(f"Average processing time: {avg_process_time*1000:.2f}ms")
                print(f"Current FPS: {fps:.2f}")
                
                persistent_cars = car_tracker.check_persistent_cars(current_time)
                print(f"Currently persistent cars: {len(persistent_cars)}")
                
                for car_info in persistent_cars:
                    car_id = car_info['car_id']
                    duration = car_info['duration']
                    count = car_info['detection_count']
                    status = "TRIGGERED" if car_id in triggered_cars else "MONITORING"
                    print(f"  Car {car_id}: {duration:.1f}s ({count} detections) - {status}")
                    
            elif key == ord('r'):
                print("🔄 Resetting car tracking...")
                car_tracker = CarTracker(persistence_threshold=15)
                existing_cars = {}
                triggered_cars = set()
                print("✅ Tracking reset complete")
                
            elif key == ord('p'):
                print(f"\n⚡ PERFORMANCE STATISTICS")
                print(f"Average processing time: {avg_process_time*1000:.2f}ms")
                print(f"Current FPS: {fps:.2f}")
                print(f"Frame skip interval: {car_detector_obj.frame_skip_interval}")
                print(f"Processing resolution: {car_detector_obj.input_size}")
                print(f"Total frames processed: {frame_count}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🏁 OPTIMIZED Car Detection Monitor - Session Complete")
        print(f"Total cars that triggered main.py: {len(triggered_cars)}")
        print(f"Average processing time: {avg_process_time*1000:.2f}ms")
        print(f"Final FPS: {fps:.2f}")
        
    except KeyboardInterrupt:
        print("\n⚡ System interrupted by user")
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logging.error(f"Main function error: {str(e)}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


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
# import queue
# import psutil
# import os

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('car_detection_monitor.log'),
#         logging.StreamHandler()
#     ]
# )

# class UltraOptimizedCarDetector:
#     """Ultra-optimized car detection with adaptive performance tuning"""
    
#     def __init__(self, model_path='yolov8n.pt'):
#         self.model = None
#         self.model_path = model_path
#         self.load_model()
        
#         # Dynamic optimization settings based on system capabilities
#         self.cpu_count = psutil.cpu_count()
#         self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
#         # Adaptive frame skipping based on system specs
#         if self.cpu_count >= 8 and self.ram_gb >= 16:
#             self.frame_skip_interval = 2
#             self.input_size = (640, 360)
#         elif self.cpu_count >= 4 and self.ram_gb >= 8:
#             self.frame_skip_interval = 3
#             self.input_size = (480, 270)
#         else:
#             self.frame_skip_interval = 4
#             self.input_size = (320, 180)
        
#         self.frame_skip_counter = 0
#         self.last_detection_time = 0
#         self.adaptive_threshold = 0.4  # Higher threshold for fewer false positives
        
#         # Pre-allocate memory buffers
#         self.resized_frame = np.zeros((*self.input_size[::-1], 3), dtype=np.uint8)
#         self.detection_cache = {}
#         self.cache_timeout = 1.0  # Cache results for 1 second
        
#         # Performance monitoring
#         self.performance_tracker = {
#             'inference_times': [],
#             'frame_processing_times': [],
#             'cpu_usage': []
#         }
        
#         print(f"🚀 Ultra-Optimized Detector initialized:")
#         print(f"   CPU cores: {self.cpu_count}, RAM: {self.ram_gb:.1f}GB")
#         print(f"   Frame skip: {self.frame_skip_interval}, Size: {self.input_size}")
        
#     def load_model(self):
#         """Load YOLO model with optimization settings"""
#         try:
#             print("🔄 Loading optimized YOLO model...")
            
#             # Set environment variables for optimization
#             os.environ['OMP_NUM_THREADS'] = str(max(1, self.cpu_count // 2))
#             os.environ['OPENBLAS_NUM_THREADS'] = '1'
#             os.environ['MKL_NUM_THREADS'] = '1'
            
#             self.model = YOLO(self.model_path)
            
#             # Configure model for optimization
#             self.model.overrides['verbose'] = False
#             self.model.overrides['device'] = 'cpu'
            
#             # Warm up with minimal dummy inference
#             dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
#             self.model(dummy_frame, conf=0.5, verbose=False, half=False)
            
#             print("✅ Model loaded and optimized")
            
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             raise
    
#     def should_process_frame(self):
#         """Adaptive frame skipping with performance monitoring"""
#         self.frame_skip_counter += 1
        
#         # Dynamic adjustment based on CPU usage
#         cpu_percent = psutil.cpu_percent(interval=None)
#         self.performance_tracker['cpu_usage'].append(cpu_percent)
        
#         # Increase skip interval if CPU usage is high
#         if cpu_percent > 80:
#             effective_skip = self.frame_skip_interval + 1
#         elif cpu_percent > 60:
#             effective_skip = self.frame_skip_interval
#         else:
#             effective_skip = max(1, self.frame_skip_interval - 1)
        
#         should_process = (self.frame_skip_counter % effective_skip) == 0
#         return should_process
    
#     def preprocess_frame_optimized(self, frame):
#         """Ultra-optimized frame preprocessing"""
#         height, width = frame.shape[:2]
#         target_w, target_h = self.input_size
        
#         # Only resize if necessary and use fastest interpolation
#         if width != target_w or height != target_h:
#             # Use cv2.resize with INTER_NEAREST for fastest processing
#             self.resized_frame = cv2.resize(frame, self.input_size, 
#                                           interpolation=cv2.INTER_NEAREST)
#         else:
#             self.resized_frame = frame
            
#         return self.resized_frame
    
#     def detect_cars_cached(self, frame):
#         """Car detection with result caching"""
#         current_time = time.time()
        
#         # Check cache first
#         frame_hash = hash(frame.tobytes())
#         if frame_hash in self.detection_cache:
#             cached_result, cache_time = self.detection_cache[frame_hash]
#             if current_time - cache_time < self.cache_timeout:
#                 return cached_result
        
#         # Process frame
#         start_time = time.time()
#         processed_frame = self.preprocess_frame_optimized(frame)
        
#         # Apply minimal zoom for detection
#         detection_frame = self.apply_zoom_fast(processed_frame, 0.9)
        
#         # Run inference with optimized settings
#         try:
#             results = self.model(detection_frame, 
#                                conf=self.adaptive_threshold,
#                                verbose=False,
#                                half=False,
#                                augment=False,
#                                visualize=False)[0]
            
#             inference_time = time.time() - start_time
#             self.performance_tracker['inference_times'].append(inference_time)
            
#             # Process results efficiently
#             cars = self.process_detections_fast(results, frame.shape, detection_frame.shape)
            
#             # Cache the result
#             self.detection_cache[frame_hash] = (cars, current_time)
            
#             # Cleanup old cache entries
#             self.cleanup_cache(current_time)
            
#             return cars
            
#         except Exception as e:
#             logging.error(f"Detection error: {str(e)}")
#             return []
    
#     def apply_zoom_fast(self, frame, zoom_factor):
#         """Fast zoom implementation"""
#         if zoom_factor == 1.0:
#             return frame
        
#         if zoom_factor < 1.0:
#             return frame  # Skip zoom out for performance
        
#         # Simple crop-based zoom in
#         height, width = frame.shape[:2]
#         crop_height = int(height / zoom_factor)
#         crop_width = int(width / zoom_factor)
        
#         y_start = (height - crop_height) // 2
#         x_start = (width - crop_width) // 2
        
#         cropped = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
#         return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)
    
#     def process_detections_fast(self, results, original_shape, processed_shape):
#         """Fast detection processing"""
#         if not hasattr(results, 'boxes') or results.boxes is None:
#             return []
        
#         detections = results.boxes.data.tolist()
#         if not detections:
#             return []
        
#         # Scale factors
#         orig_h, orig_w = original_shape[:2]
#         proc_h, proc_w = processed_shape[:2]
#         scale_x = orig_w / proc_w
#         scale_y = orig_h / proc_h
        
#         cars = []
#         min_area = max(1000, (orig_w * orig_h) // 1000)  # Adaptive minimum area
        
#         for detection in detections:
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
                
#                 # Filter for vehicles only
#                 if int(class_id) not in [2, 3, 5, 7] or score < self.adaptive_threshold:
#                     continue
                
#                 # Scale coordinates
#                 x1_scaled = int(x1 * scale_x)
#                 y1_scaled = int(y1 * scale_y)
#                 x2_scaled = int(x2 * scale_x)
#                 y2_scaled = int(y2 * scale_y)
                
#                 bbox_area = (x2_scaled - x1_scaled) * (y2_scaled - y1_scaled)
                
#                 # Filter small detections
#                 if bbox_area > min_area:
#                     cars.append({
#                         'bbox': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
#                         'confidence': score,
#                         'class_id': int(class_id),
#                         'center': ((x1_scaled + x2_scaled) // 2, (y1_scaled + y2_scaled) // 2),
#                         'area': bbox_area
#                     })
        
#         # Sort by confidence and keep only top detections
#         cars.sort(key=lambda x: x['confidence'], reverse=True)
#         return cars[:10]  # Limit to top 10 detections
    
#     def cleanup_cache(self, current_time):
#         """Clean up old cache entries"""
#         if len(self.detection_cache) > 50:  # Limit cache size
#             expired_keys = [k for k, (_, t) in self.detection_cache.items() 
#                            if current_time - t > self.cache_timeout]
#             for key in expired_keys:
#                 del self.detection_cache[key]
    
#     def get_performance_stats(self):
#         """Get performance statistics"""
#         stats = {}
#         if self.performance_tracker['inference_times']:
#             stats['avg_inference_time'] = np.mean(self.performance_tracker['inference_times'][-50:])
#         if self.performance_tracker['cpu_usage']:
#             stats['avg_cpu_usage'] = np.mean(self.performance_tracker['cpu_usage'][-50:])
#         return stats

# class OptimizedCarTracker:
#     """Optimized car tracking with memory management"""
    
#     def __init__(self, persistence_threshold=15):
#         self.car_detections = defaultdict(list)
#         self.persistence_threshold = persistence_threshold
#         self.last_cleanup = time.time()
#         self.max_detections_per_car = 20  # Limit memory usage
        
#     def add_detection(self, car_id, timestamp):
#         """Add detection with memory management"""
#         detections = self.car_detections[car_id]
#         detections.append(timestamp)
        
#         # Limit number of detections per car
#         if len(detections) > self.max_detections_per_car:
#             detections.pop(0)
        
#         # Periodic cleanup
#         current_time = time.time()
#         if current_time - self.last_cleanup > 10:  # Cleanup every 10 seconds
#             self.cleanup_old_detections(current_time)
#             self.last_cleanup = current_time
    
#     def cleanup_old_detections(self, current_time):
#         """Optimized cleanup of old detections"""
#         cleanup_threshold = self.persistence_threshold + 10
        
#         # Use list comprehension for faster cleanup
#         cars_to_remove = []
#         for car_id, timestamps in self.car_detections.items():
#             # Filter out old timestamps
#             valid_timestamps = [t for t in timestamps if current_time - t <= cleanup_threshold]
            
#             if valid_timestamps:
#                 self.car_detections[car_id] = valid_timestamps
#             else:
#                 cars_to_remove.append(car_id)
        
#         # Remove empty cars
#         for car_id in cars_to_remove:
#             del self.car_detections[car_id]
    
#     def check_persistent_cars(self, current_time):
#         """Optimized persistent car checking"""
#         persistent_cars = []
        
#         for car_id, timestamps in self.car_detections.items():
#             if len(timestamps) < 5:  # Require more detections for stability
#                 continue
            
#             earliest = min(timestamps)
#             latest = max(timestamps)
#             duration = current_time - earliest
            
#             if (duration >= self.persistence_threshold and 
#                 current_time - latest <= 5):  # Must be recently detected
#                 persistent_cars.append({
#                     'car_id': car_id,
#                     'first_seen': earliest,
#                     'duration': duration,
#                     'detection_count': len(timestamps)
#                 })
        
#         return persistent_cars

# class FrameBuffer:
#     """Thread-safe frame buffer for async processing"""
    
#     def __init__(self, maxsize=2):
#         self.queue = queue.Queue(maxsize=maxsize)
#         self.latest_frame = None
#         self.lock = threading.Lock()
    
#     def put_frame(self, frame):
#         """Add frame to buffer"""
#         with self.lock:
#             self.latest_frame = frame.copy()
            
#         # Non-blocking put
#         try:
#             self.queue.put(frame, block=False)
#         except queue.Full:
#             # Remove old frame and add new one
#             try:
#                 self.queue.get_nowait()
#                 self.queue.put(frame, block=False)
#             except queue.Empty:
#                 pass
    
#     def get_frame(self):
#         """Get frame from buffer"""
#         try:
#             return self.queue.get(timeout=0.1)
#         except queue.Empty:
#             with self.lock:
#                 return self.latest_frame.copy() if self.latest_frame is not None else None

# def get_car_id_optimized(car, existing_cars, max_distance=100):
#     """Optimized car ID assignment"""
#     if not existing_cars:
#         return 1
    
#     car_center = car['center']
#     car_area = car['area']
    
#     best_match_id = None
#     best_score = float('inf')
    
#     # Use numpy for faster distance calculations
#     centers = np.array([data['center'] for data in existing_cars.values()])
#     car_center_np = np.array(car_center)
    
#     # Vectorized distance calculation
#     distances = np.linalg.norm(centers - car_center_np, axis=1)
    
#     for i, (car_id, existing_data) in enumerate(existing_cars.items()):
#         distance = distances[i]
        
#         if distance > max_distance:
#             continue
        
#         existing_area = existing_data.get('area', car_area)
#         area_ratio = min(car_area, existing_area) / max(car_area, existing_area)
#         score = distance - (area_ratio * 30)
        
#         if score < best_score:
#             best_score = score
#             best_match_id = car_id
    
#     return best_match_id if best_match_id is not None else max(existing_cars.keys()) + 1

# def get_class_name(class_id):
#     """Get class name"""
#     class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
#     return class_names.get(class_id, "Vehicle")

# def trigger_main_script_async(camera_id, frame):
#     """Asynchronous main script triggering"""
#     def run_script():
#         try:
#             logging.info(f"🚨 TRIGGERING MAIN SCRIPT for camera {camera_id}")
            
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             frame_path = f"temp_frame_{camera_id}_{timestamp}.jpg"
            
#             # Simple zoom
#             height, width = frame.shape[:2]
#             crop_size = min(height, width) // 2
#             y_start = (height - crop_size) // 2
#             x_start = (width - crop_size) // 2
#             zoomed_frame = frame[y_start:y_start+crop_size, x_start:x_start+crop_size]
#             zoomed_frame = cv2.resize(zoomed_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
#             cv2.imwrite(frame_path, zoomed_frame)
            
#             result = subprocess.run([
#                 sys.executable, "exitDetection.py", str(camera_id), frame_path
#             ], capture_output=True, text=True, timeout=120)
            
#             if result.returncode == 0:
#                 logging.info("✅ Main script completed successfully")
#                 if result.stdout:
#                     print("📋 RESULT:", result.stdout)
#             else:
#                 logging.error(f"❌ Main script failed: {result.stderr}")
            
#             # Cleanup
#             try:
#                 os.remove(frame_path)
#             except:
#                 pass
                
#         except Exception as e:
#             logging.error(f"Script trigger error: {str(e)}")
    
#     # Run in background thread
#     thread = threading.Thread(target=run_script)
#     thread.daemon = True
#     thread.start()

# def main():
#     print("🚀 ULTRA-OPTIMIZED Car Detection Monitor")
#     print("=" * 60)
    
#     # System info
#     cpu_count = psutil.cpu_count()
#     ram_gb = psutil.virtual_memory().total / (1024**3)
#     print(f"💻 System: {cpu_count} CPU cores, {ram_gb:.1f}GB RAM")
    
#     try:
#         camera_id = 1
        
#         # Test camera
#         print(f"📹 Testing camera {camera_id}...")
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
#         print(f"✅ Camera {camera_id} ready")
        
#         # Initialize optimized components
#         print("🔄 Initializing ultra-optimized detector...")
#         car_detector = UltraOptimizedCarDetector('yolov8n.pt')
#         car_tracker = OptimizedCarTracker(persistence_threshold=15)
#         frame_buffer = FrameBuffer(maxsize=2)
        
#         # Camera setup with optimization
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)   # Smaller resolution
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
#         print("\n🚀 Starting ULTRA-OPTIMIZED detection")
#         print("Optimizations:")
#         print(f"  • Adaptive frame skipping: {car_detector.frame_skip_interval}")
#         print(f"  • Processing resolution: {car_detector.input_size}")
#         print(f"  • Result caching: {car_detector.cache_timeout}s")
#         print(f"  • CPU optimization: {os.environ.get('OMP_NUM_THREADS', 'default')} threads")
#         print("  • Memory management: Limited buffers")
#         print("  • Async script execution")
        
#         # Main variables
#         frame_count = 0
#         existing_cars = {}
#         triggered_cars = set()
#         start_time = time.time()
#         last_performance_check = time.time()
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("Frame read failed, retrying...")
#                 time.sleep(0.1)
#                 continue
            
#             current_time = time.time()
#             frame_count += 1
            
#             # Add frame to buffer
#             frame_buffer.put_frame(frame)
            
#             # Process frame if needed
#             cars = []
#             if car_detector.should_process_frame():
#                 processing_frame = frame_buffer.get_frame()
#                 if processing_frame is not None:
#                     cars = car_detector.detect_cars_cached(processing_frame)
            
#             # Update tracking
#             if cars:
#                 current_cars = {}
#                 for car in cars:
#                     car_id = get_car_id_optimized(car, existing_cars)
#                     current_cars[car_id] = {
#                         'center': car['center'],
#                         'area': car['area'],
#                         'confidence': car['confidence'],
#                         'class_id': car['class_id']
#                     }
#                     car_tracker.add_detection(car_id, current_time)
                
#                 existing_cars = current_cars
                
#                 # Draw detections
#                 for car in cars:
#                     x1, y1, x2, y2 = car['bbox']
#                     confidence = car['confidence']
#                     class_name = get_class_name(car['class_id'])
                    
#                     color = (0, 255, 0) if car['class_id'] == 2 else (255, 0, 0)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
#                     label = f"{class_name} {confidence:.2f}"
#                     cv2.putText(frame, label, (x1, y1-10), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             # Check persistent cars (every 5 seconds)
#             if frame_count % (15 * 5) == 0:
#                 persistent_cars = car_tracker.check_persistent_cars(current_time)
                
#                 for car_info in persistent_cars:
#                     car_id = car_info['car_id']
                    
#                     if car_id not in triggered_cars:
#                         triggered_cars.add(car_id)
#                         logging.info(f"🎯 Car {car_id} triggered after {car_info['duration']:.1f}s")
#                         trigger_main_script_async(camera_id, frame)
            
#             # Performance monitoring
#             if current_time - last_performance_check > 10:
#                 stats = car_detector.get_performance_stats()
#                 cpu_usage = stats.get('avg_cpu_usage', 0)
                
#                 if cpu_usage > 70:
#                     car_detector.frame_skip_interval = min(6, car_detector.frame_skip_interval + 1)
#                     car_detector.adaptive_threshold = min(0.6, car_detector.adaptive_threshold + 0.05)
#                     print(f"⚠️  High CPU usage ({cpu_usage:.1f}%), adjusting performance")
                
#                 last_performance_check = current_time
            
#             # Display status
#             fps = frame_count / (current_time - start_time) if current_time > start_time else 0
#             stats = car_detector.get_performance_stats()
            
#             status = f"Camera {camera_id} | Cars: {len(existing_cars)} | FPS: {fps:.1f}"
#             if 'avg_inference_time' in stats:
#                 status += f" | Process: {stats['avg_inference_time']*1000:.0f}ms"
            
#             cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # Show optimizations
#             opt_info = f"ULTRA-OPT: Skip={car_detector.frame_skip_interval} | Thresh={car_detector.adaptive_threshold:.2f}"
#             cv2.putText(frame, opt_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
#             # Show persistent cars
#             persistent_cars = car_tracker.check_persistent_cars(current_time)
#             y_offset = 90
#             for car_info in persistent_cars:
#                 car_id = car_info['car_id']
#                 duration = car_info['duration']
#                 status = "TRIGGERED" if car_id in triggered_cars else f"{15-duration:.1f}s"
                
#                 text = f"Car {car_id}: {duration:.1f}s - {status}"
#                 color = (0, 255, 255) if car_id in triggered_cars else (255, 255, 0)
#                 cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                 y_offset += 20
            
#             # Show frame
#             cv2.imshow(f"Car Exit Ultra-Optimized Detection - Camera {camera_id}", frame)
            
#             # Handle keys
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:
#                 break
#             elif key == ord('s'):
#                 stats = car_detector.get_performance_stats()
#                 print(f"\n📊 PERFORMANCE STATISTICS")
#                 print(f"Active cars: {len(existing_cars)}")
#                 print(f"Triggered cars: {len(triggered_cars)}")
#                 print(f"FPS: {fps:.2f}")
#                 print(f"Frame skip: {car_detector.frame_skip_interval}")
#                 print(f"Threshold: {car_detector.adaptive_threshold:.2f}")
#                 if 'avg_inference_time' in stats:
#                     print(f"Avg inference: {stats['avg_inference_time']*1000:.1f}ms")
#                 if 'avg_cpu_usage' in stats:
#                     print(f"Avg CPU: {stats['avg_cpu_usage']:.1f}%")
#             elif key == ord('r'):
#                 print("🔄 Resetting tracking...")
#                 car_tracker = OptimizedCarTracker(persistence_threshold=15)
#                 existing_cars = {}
#                 triggered_cars = set()
#                 print("✅ Reset complete")
        
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
        
#         print(f"\n🏁 Session Complete")
#         print(f"Total triggered cars: {len(triggered_cars)}")
#         print(f"Final FPS: {fps:.2f}")
        
#     except KeyboardInterrupt:
#         print("\n⚡ Interrupted by user")
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"\n❌ Error: {str(e)}")
#         logging.error(f"Main error: {str(e)}")
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()