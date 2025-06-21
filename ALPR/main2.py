# from ultralytics import YOLO
# import cv2
# import numpy as np
# import pytesseract
# import re
# from collections import defaultdict, Counter
# import easyocr
# import time
# import os
# import logging
# import sys
# from datetime import datetime

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('license_plate_detection.log'),
#         logging.StreamHandler()
#     ]
# )

# class PlateDetection:
#     """Enhanced plate detection with stability scoring and temporal consistency"""
#     def __init__(self, camera_id, detection_duration=60):
#         self.camera_id = camera_id
#         self.camera_name = f"Camera {camera_id}"
#         self.detection_duration = detection_duration
#         self.start_time = None
        
#         # Enhanced tracking with scores and temporal data
#         self.plate_scores = defaultdict(list)
#         self.recent_detections = defaultdict(list)
#         self.active_detections = {}
#         self.confirmed_plates = defaultdict(int)
#         # Add counter for all detections (not just confirmed ones)
#         self.all_detections_count = defaultdict(int)
#         self.plate_history = Counter()
#         self.all_plates = Counter()  # Track all detected plates across the entire session
        
#         # Enhanced parameters for better stability
#         self.frame_window = 30  # Using value from reference code
#         self.confidence_threshold = 0.6
#         self.min_scores = 6
#         self.min_recent = 4
#         self.recent_time_window = 8  # seconds
#         self.active_detection_timeout = 5  # seconds
        
#     def update_scores(self, plate_number, confidence, text_size):
#         """Update scores using logic from reference code"""
#         score = confidence * 0.6 + text_size * 0.4  # Using reference code formula
#         self.plate_scores[plate_number].append(score)
#         self.plate_history[plate_number] += 1
#         self.all_plates[plate_number] += 1  # Increment the count for this plate
        
#         # Keep only the last window of scores for real-time processing
#         if len(self.plate_scores[plate_number]) > self.frame_window:
#             self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
        
#         current_time = time.time()
        
#         if self.start_time is None:
#             self.start_time = current_time
        
#         # Update recent detections
#         self.recent_detections[plate_number].append(current_time)
        
#         # Count all detections
#         self.all_detections_count[plate_number] += 1
        
#         # Clean old data
#         self._cleanup_old_data(current_time)
        
#     def add_detection(self, plate_number, confidence, text_size, bbox):
#         """Add a license plate detection with enhanced scoring"""
#         current_time = time.time()
        
#         if self.start_time is None:
#             self.start_time = current_time
        
#         # Calculate composite score using reference code logic
#         score = confidence * 0.6 + text_size * 0.4
        
#         # Update score tracking
#         self.plate_scores[plate_number].append(score)
#         self.recent_detections[plate_number].append(current_time)
#         self.plate_history[plate_number] += 1
#         self.all_plates[plate_number] += 1
        
#         # Count all detections
#         self.all_detections_count[plate_number] += 1
        
#         # Store active detection with bounding box
#         self.active_detections[plate_number] = {
#             'bbox': bbox,
#             'timestamp': current_time,
#             'confidence': confidence,
#             'score': score
#         }
        
#         # Clean old data
#         self._cleanup_old_data(current_time)
        
#         # Keep only recent frame window
#         if len(self.plate_scores[plate_number]) > self.frame_window:
#             self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
    
#     def _cleanup_old_data(self, current_time):
#         """Clean up old detection data"""
#         # Clean recent detections (keep only last 8 seconds)
#         cutoff_time = current_time - self.recent_time_window
#         for plate in list(self.recent_detections.keys()):
#             self.recent_detections[plate] = [
#                 t for t in self.recent_detections[plate] if t > cutoff_time
#             ]
#             if not self.recent_detections[plate]:
#                 del self.recent_detections[plate]
        
#         # Clean old active detections
#         plates_to_remove = []
#         for plate, detection in self.active_detections.items():
#             if current_time - detection['timestamp'] > self.active_detection_timeout:
#                 plates_to_remove.append(plate)
        
#         for plate in plates_to_remove:
#             del self.active_detections[plate]

#     def get_best_plate(self):
#         """Get best plate using reference code logic"""
#         if not self.plate_scores:
#             return None, 0, 0
#         best_plate = None
#         best_score = 0

#         for plate, scores in self.plate_scores.items():
#             if not scores:
#                 continue
#             avg_score = sum(scores) / len(scores)
#             frequency = len(scores) / self.frame_window
#             final_score = avg_score * 0.7 + frequency * 0.3

#             if final_score > best_score:
#                 best_score = final_score
#                 best_plate = plate

#         if best_plate:
#             frequency = len(self.plate_scores[best_plate]) / self.frame_window
#             return best_plate, best_score, frequency
#         return None, 0, 0
    
#     def get_top_plates(self, n=5):
#         """Return the top n most frequently detected plates"""
#         return self.all_plates.most_common(n)
    
#     def get_stable_plates(self):
#         """Return plates that have been consistently detected and are stable"""
#         stable_plates = []
#         current_time = time.time()
        
#         for plate, scores in self.plate_scores.items():
#             recent_count = len(self.recent_detections.get(plate, []))
            
#             # Check stability criteria
#             if len(scores) >= self.min_scores and recent_count >= self.min_recent:
#                 avg_score = sum(scores[-self.min_scores:]) / self.min_scores
                
#                 if avg_score > self.confidence_threshold:
#                     # Check if detection is recent
#                     if plate in self.recent_detections:
#                         last_detection = max(self.recent_detections[plate])
#                         if current_time - last_detection < 2:
#                             stable_plates.append((plate, avg_score))
#                             # Clear recent detections to avoid repeated confirmations
#                             self.recent_detections[plate] = []
#                             # Mark as confirmed
#                             self.confirmed_plates[plate] += 1
        
#         return stable_plates
    
#     def get_active_detections(self):
#         """Return currently active detections for drawing"""
#         return self.active_detections.copy()
    
#     def get_detection_summary(self):
#         """Get comprehensive summary of all detections"""
#         summary = {}
#         for plate in self.all_detections_count:
#             scores = self.plate_scores.get(plate, [])
#             if scores:
#                 summary[plate] = {
#                     'count': self.all_detections_count[plate],
#                     'confirmed_count': self.confirmed_plates[plate],
#                     'avg_confidence': sum(scores) / len(scores),
#                     'max_confidence': max(scores),
#                     'total_detections': len(scores)
#                 }
#         return summary
    
#     def get_most_detected_plate(self):
#         """Return the most frequently detected license plate (by total count)"""
#         if not self.all_detections_count:
#             return None, 0
        
#         # Find plate with highest total detection count
#         best_plate = max(self.all_detections_count.items(), key=lambda x: x[1])
#         return best_plate[0], best_plate[1]
    
#     def is_detection_complete(self):
#         """Check if detection period is complete"""
#         if self.start_time is None:
#             return False
#         return (time.time() - self.start_time) >= self.detection_duration
    
#     def get_remaining_time(self):
#         """Get remaining detection time in seconds"""
#         if self.start_time is None:
#             return self.detection_duration
#         elapsed = time.time() - self.start_time
#         return max(0, self.detection_duration - elapsed)

# def extract_text_with_tesseract(image):
#     """Enhanced tesseract configuration with more parameters from reference code"""
#     custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_do_invert=0'
#     try:
#         text = pytesseract.image_to_string(image, config=custom_config)
#         return text.strip()
#     except Exception as e:
#         return ""

# def extract_text_with_easyocr(reader, image):
#     """Extract text using EasyOCR with optimized settings from reference code"""
#     try:
#         # Lower confidence threshold to catch more potential plates
#         results = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=20)
#         cleaned_texts = []
#         for (bbox, text, confidence) in results:
#             # Only keep alphanumeric characters
#             text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
#             # More lenient length check
#             if 2 <= len(text) <= 10:
#                 cleaned_texts.append((text, confidence))
#         return cleaned_texts
#     except Exception as e:
#         return []

# def enhance_plate_region(plate_region):
#     """Apply multiple image enhancement techniques from reference code"""
#     # Resize for better OCR performance if too small
#     min_height = 40  # Increased from 30 as in reference code
#     if plate_region.shape[0] < min_height:
#         scale = min_height / plate_region.shape[0]
#         width = int(plate_region.shape[1] * scale)
#         plate_region = cv2.resize(plate_region, (width, min_height), interpolation=cv2.INTER_CUBIC)

#     # Convert to grayscale
#     if len(plate_region.shape) == 3:
#         gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = plate_region

#     # Apply bilateral filter to remove noise while preserving edges
#     filtered = cv2.bilateralFilter(gray, 11, 17, 17)

#     # Try different thresholding techniques
#     thresh_methods = []

#     # Method 1: Adaptive thresholding
#     adaptive_thresh = cv2.adaptiveThreshold(
#         filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 11, 2
#     )
#     thresh_methods.append(adaptive_thresh)

#     # Method 2: Otsu's thresholding
#     _, otsu_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     thresh_methods.append(otsu_thresh)

#     # Method 3: Simple binary threshold
#     _, simple_thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
#     thresh_methods.append(simple_thresh)

#     # Apply morphological closing to all thresholded images
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Using reference code kernel size
#     processed_images = []
#     for thresh in thresh_methods:
#         closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#         processed_images.append(closed)

#     # Also add the original grayscale and filtered images
#     processed_images.append(gray)
#     processed_images.append(filtered)

#     return processed_images, plate_region

# def detect_license_plates(frame, license_plate_detector, reader):
#     """Enhanced license plate detection using reference code logic"""
#     try:
#         # Make a copy of the frame to avoid modifying the original
#         processed_frame = frame.copy()
        
#         # Apply a series of image enhancements from reference code
#         # 1. Histogram equalization for better contrast
#         gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.equalizeHist(gray)
        
#         # 2. Apply gamma correction to adjust brightness for night videos
#         gamma = 1.5  # Increased gamma for night videos as in reference code
#         lookUpTable = np.empty((1, 256), np.uint8)
#         for i in range(256):
#             lookUpTable[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
#         processed_frame = cv2.LUT(processed_frame, lookUpTable)
        
#         # 3. Increase contrast
#         alpha = 1.3  # Contrast control (1.0 means no change)
#         beta = 10    # Brightness control (0 means no change)
#         processed_frame = cv2.convertScaleAbs(processed_frame, alpha=alpha, beta=beta)

#         # Apply detection with lower confidence threshold from reference code
#         conf_threshold = 0.35  # Lower confidence threshold
#         detections = license_plate_detector(processed_frame, conf=conf_threshold)[0]
#         frame_plates = []

#         for detection in detections.boxes.data.tolist():
#             if len(detection) >= 6:
#                 x1, y1, x2, y2, score, class_id = detection
#                 if score < conf_threshold:
#                     continue
                    
#                 x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
#                 # Add padding around the license plate region
#                 padding = 5  # Using reference code padding
#                 y1 = max(0, y1 - padding)
#                 y2 = min(frame.shape[0], y2 + padding)
#                 x1 = max(0, x1 - padding)
#                 x2 = min(frame.shape[1], x2 + padding)
                
#                 if x1 >= x2 or y1 >= y2:
#                     continue
                    
#                 plate_region = processed_frame[y1:y2, x1:x2]
                
#                 if plate_region.size == 0:
#                     continue
                
#                 try:
#                     # Apply multiple image processing techniques from reference code
#                     processed_images, resized_plate = enhance_plate_region(plate_region)
                    
#                     all_texts = []
                    
#                     # Try OCR on all processed images
#                     for img in processed_images:
#                         # Try with Tesseract
#                         tesseract_text = extract_text_with_tesseract(img)
#                         if tesseract_text:
#                             all_texts.append((tesseract_text, 0.8))
                    
#                     # Try with EasyOCR on the original and resized plate
#                     easyocr_texts = extract_text_with_easyocr(reader, resized_plate)
#                     all_texts.extend(easyocr_texts)
                    
#                     # Process all detected texts
#                     for text, confidence in all_texts:
#                         if not text.strip():
#                             continue
                        
#                         text_height = resized_plate.shape[0]
#                         relative_text_size = text_height / (y2 - y1)
                        
#                         # Clean text - keep only alphanumeric characters
#                         cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                        
#                         # More lenient length check for night videos
#                         if 2 <= len(cleaned_text) <= 10:
#                             frame_plates.append({
#                                 'plate_number': cleaned_text,
#                                 'confidence': confidence * score,
#                                 'text_size': relative_text_size,
#                                 'bbox': (x1, y1, x2, y2)
#                             })
#                 except Exception as e:
#                     logging.warning(f"OCR processing error: {str(e)}")
        
#         return frame_plates
        
#     except Exception as e:
#         logging.error(f"Detection error: {str(e)}")
#         return []

# def detect_license_plates_for_duration(camera_id, duration=60):
#     """Main function to detect license plates for specified duration with enhanced stability"""
#     print(f"🔍 Enhanced License Plate Detection on Camera {camera_id}")
#     print(f"⏱️  Detection Duration: {duration} seconds")
#     print("🔧 Features: Stability scoring, temporal consistency, enhanced OCR")
#     print("=" * 60)
    
#     try:
#         # Test camera connection
#         cap = cv2.VideoCapture(camera_id)
#         if not cap.isOpened():
#             logging.error(f"Cannot connect to camera {camera_id}")
#             return None
        
#         ret, frame = cap.read()
#         if not ret:
#             logging.error(f"Cannot read from camera {camera_id}")
#             cap.release()
#             return None
        
#         cap.release()
#         logging.info(f"Camera {camera_id} connection verified")
        
#         # Load license plate detection model
#         license_plate_model = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/license_plate_detector.pt"
        
#         if not os.path.isfile(license_plate_model):
#             logging.error(f"Model file not found at: {license_plate_model}")
#             return None
        
#         logging.info("Loading license plate detection model...")
#         license_plate_detector = YOLO(license_plate_model)
#         logging.info("✅ License plate model loaded")
        
#         # Initialize EasyOCR
#         logging.info("Initializing EasyOCR...")
#         try:
#             reader = easyocr.Reader(['en'], gpu=True)
#             logging.info("✅ EasyOCR initialized with GPU")
#         except Exception as e:
#             logging.warning(f"EasyOCR GPU failed: {str(e)}, trying CPU...")
#             try:
#                 reader = easyocr.Reader(['en'], gpu=False)
#                 logging.info("✅ EasyOCR initialized with CPU")
#             except Exception as e2:
#                 logging.error(f"EasyOCR initialization failed: {str(e2)}")
#                 return None
        
#         # Initialize enhanced plate detector
#         plate_detector = PlateDetection(camera_id, duration)
        
#         # Start camera capture
#         cap = cv2.VideoCapture(camera_id)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         cap.set(cv2.CAP_PROP_FPS, 25)  # Consistent FPS
        
#         # Try to set resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         logging.info(f"🚀 Starting enhanced {duration}-second license plate detection...")
        
#         frame_count = 0
#         last_plate_check = time.time()
#         confirmed_detections = []
        
#         while not plate_detector.is_detection_complete():
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("Failed to read frame")
#                 time.sleep(0.1)
#                 continue
            
#             frame_count += 1
#             current_time = time.time()
            
#             # Process every 2nd frame to speed up analysis (from reference code)
#             process_interval = 2
#             if frame_count % process_interval == 0:
#                 plates = detect_license_plates(frame, license_plate_detector, reader)
                
#                 for plate in plates:
#                     plate_number = plate['plate_number']
#                     confidence = plate['confidence']
#                     text_size = plate['text_size']
#                     bbox = plate['bbox']
                    
#                     # Add to enhanced detector with scoring
#                     plate_detector.add_detection(plate_number, confidence, text_size, bbox)
            
#             # Draw all active detections with enhanced visualization
#             active_detections = plate_detector.get_active_detections()
#             for plate_number, detection in active_detections.items():
#                 x1, y1, x2, y2 = detection['bbox']
#                 confidence = detection['confidence']
#                 score = detection['score']
                
#                 # Color coding based on confidence
#                 if confidence > 0.8:
#                     color = (0, 255, 0)  # Green for high confidence
#                 elif confidence > 0.6:
#                     color = (0, 255, 255)  # Yellow for medium confidence
#                 else:
#                     color = (0, 165, 255)  # Orange for lower confidence
                
#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
#                 # Draw detailed label
#                 label = f"{plate_number} (C:{confidence:.2f}, S:{score:.2f})"
#                 label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#                 cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
#                 cv2.putText(frame, label, (x1, y1-5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
#             # Check for stable plates at intervals
#             check_interval = 2.0
#             if current_time - last_plate_check > check_interval:
#                 stable_plates = plate_detector.get_stable_plates()
#                 for plate_number, avg_score in stable_plates:
#                     confirmed_detections.append({
#                         'plate': plate_number,
#                         'score': avg_score,
#                         'timestamp': current_time
#                     })
                    
#                     # Show confirmation on screen
#                     cv2.putText(frame, f"CONFIRMED: {plate_number}", (10, 60), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#                     logging.info(f"🎯 CONFIRMED DETECTION: {plate_number} (Score: {avg_score:.3f})")
                
#                 last_plate_check = current_time
            
#             # Enhanced status display
#             remaining = plate_detector.get_remaining_time()
#             cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
#             # Show current leading detection (by total count)
#             best_plate, best_count = plate_detector.get_most_detected_plate()
#             if best_plate:
#                 cv2.putText(frame, f"Leading: {best_plate} ({best_count}x)", (10, 90), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
#             # Show top plates from reference code logic
#             top_plates = plate_detector.get_top_plates(3)
#             if top_plates:
#                 cv2.putText(frame, f"Top 3:", (10, 120), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                 for i, (plate, count) in enumerate(top_plates[:3]):
#                     cv2.putText(frame, f"{i+1}. {plate} ({count})", (10, 150 + i*25), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
#             # Show confirmed count
#             cv2.putText(frame, f"Confirmed: {len(confirmed_detections)}", (10, 240), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             # Display frame
#             cv2.imshow(f"Enhanced License Plate Detection - Camera {camera_id}", frame)
            
#             # Allow early exit
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or key == 27:
#                 logging.info("Early exit requested")
#                 break
        
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
        
#         # Get final results - return the plate with maximum detections
#         most_detected_plate, detection_count = plate_detector.get_most_detected_plate()
        
#         # Print comprehensive results
#         print(f"\n🏁 ENHANCED DETECTION COMPLETE")
#         print("=" * 60)
        
#         if most_detected_plate:
#             print(f"🎯 MOST DETECTED PLATE: {most_detected_plate}")
#             print(f"📊 TOTAL DETECTION COUNT: {detection_count}")
#             print(f"✅ TOTAL CONFIRMATIONS: {len(confirmed_detections)}")
            
#             # Show top plates summary
#             top_plates = plate_detector.get_top_plates(5)
#             print(f"\n📋 TOP 5 DETECTED PLATES:")
#             for i, (plate, count) in enumerate(top_plates):
#                 print(f"  {i+1}. {plate} - Detected {count} times")
            
#             # Show detailed summary
#             summary = plate_detector.get_detection_summary()
#             print(f"\n📋 DETAILED DETECTION SUMMARY:")
#             for plate, stats in sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True):
#                 count = stats['count']
#                 confirmed_count = stats['confirmed_count']
#                 avg_conf = stats['avg_confidence']
#                 max_conf = stats['max_confidence']
#                 print(f"  {plate}: {count} total detections, {confirmed_count} confirmations")
#                 print(f"    Avg confidence: {avg_conf:.3f}, Max: {max_conf:.3f}")
            
#             # Show chronological confirmations
#             if confirmed_detections:
#                 print(f"\n🕒 CHRONOLOGICAL CONFIRMATIONS:")
#                 for i, detection in enumerate(confirmed_detections, 1):
#                     plate = detection['plate']
#                     score = detection['score']
#                     timestamp = detection['timestamp']
#                     rel_time = timestamp - plate_detector.start_time
#                     print(f"  {i}. {plate} at {rel_time:.1f}s (Score: {score:.3f})")
#         else:
#             print("❌ NO LICENSE PLATES DETECTED")
#             most_detected_plate = None
        
#         return most_detected_plate
        
#     except Exception as e:
#         logging.error(f"Enhanced detection error: {str(e)}")
#         return None

# def main():
#     """Enhanced main function with improved user interaction"""
#     if len(sys.argv) > 1:
#         # Called from another script with camera ID
#         try:
#             camera_id = int(sys.argv[1])
#         except ValueError:
#             logging.error("Invalid camera ID provided as argument")
#             return None
            
#         # Call the enhanced detection function
#         result = detect_license_plates_for_duration(camera_id, duration=60)
        
#         if result:
#             print(f"\n🎯 FINAL RESULT: {result}")
#             return result
#         else:
#             print("\n❌ No license plate detected")
#             return None
#     else:
#         # Interactive mode with enhanced interface
#         print("🚀 Enhanced License Plate Detection System")
#         print("Features: Stability scoring, temporal consistency, multi-OCR")
#         print("=" * 60)
        
#         try:
#             camera_id = int(input("Enter camera ID: ").strip())
#             duration = input("Enter detection duration (60s default): ").strip()
#             duration = int(duration) if duration else 60
#         except ValueError:
#             logging.error("Invalid input")
#             return None

#         # Call the enhanced detection function
#         result = detect_license_plates_for_duration(camera_id, duration=duration)
        
#         if result:
#             print(f"\n🎯 FINAL RESULT: {result}")
#             return result
#         else:
#             print("\n❌ No license plate detected")
#             return None

# if __name__ == "__main__":
#     main()



from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import re
from collections import defaultdict, Counter
import easyocr
import time
import os
import logging
import sys
from datetime import datetime
import requests
import subprocess
import webbrowser
import socket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('license_plate_detection.log'),
        logging.StreamHandler()
    ]
)

class PlateDetection:
    """Enhanced plate detection with stability scoring and temporal consistency"""
    def __init__(self, camera_id, detection_duration=60):
        self.camera_id = camera_id
        self.camera_name = f"Camera {camera_id}"
        self.detection_duration = detection_duration
        self.start_time = None
        
        # Enhanced tracking with scores and temporal data
        self.plate_scores = defaultdict(list)
        self.recent_detections = defaultdict(list)
        self.active_detections = {}
        self.confirmed_plates = defaultdict(int)
        # Add counter for all detections (not just confirmed ones)
        self.all_detections_count = defaultdict(int)
        self.plate_history = Counter()
        self.all_plates = Counter()  # Track all detected plates across the entire session
        
        # Enhanced parameters for better stability
        self.frame_window = 30  # Using value from reference code
        self.confidence_threshold = 0.6
        self.min_scores = 6
        self.min_recent = 4
        self.recent_time_window = 8  # seconds
        self.active_detection_timeout = 5  # seconds
        
    def update_scores(self, plate_number, confidence, text_size):
        """Update scores using logic from reference code"""
        score = confidence * 0.6 + text_size * 0.4  # Using reference code formula
        self.plate_scores[plate_number].append(score)
        self.plate_history[plate_number] += 1
        self.all_plates[plate_number] += 1  # Increment the count for this plate
        
        # Keep only the last window of scores for real-time processing
        if len(self.plate_scores[plate_number]) > self.frame_window:
            self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
        
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        # Update recent detections
        self.recent_detections[plate_number].append(current_time)
        
        # Count all detections
        self.all_detections_count[plate_number] += 1
        
        # Clean old data
        self._cleanup_old_data(current_time)
        
    def add_detection(self, plate_number, confidence, text_size, bbox):
        """Add a license plate detection with enhanced scoring"""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        # Calculate composite score using reference code logic
        score = confidence * 0.6 + text_size * 0.4
        
        # Update score tracking
        self.plate_scores[plate_number].append(score)
        self.recent_detections[plate_number].append(current_time)
        self.plate_history[plate_number] += 1
        self.all_plates[plate_number] += 1
        
        # Count all detections
        self.all_detections_count[plate_number] += 1
        
        # Store active detection with bounding box
        self.active_detections[plate_number] = {
            'bbox': bbox,
            'timestamp': current_time,
            'confidence': confidence,
            'score': score
        }
        
        # Clean old data
        self._cleanup_old_data(current_time)
        
        # Keep only recent frame window
        if len(self.plate_scores[plate_number]) > self.frame_window:
            self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
    
    def _cleanup_old_data(self, current_time):
        """Clean up old detection data"""
        # Clean recent detections (keep only last 8 seconds)
        cutoff_time = current_time - self.recent_time_window
        for plate in list(self.recent_detections.keys()):
            self.recent_detections[plate] = [
                t for t in self.recent_detections[plate] if t > cutoff_time
            ]
            if not self.recent_detections[plate]:
                del self.recent_detections[plate]
        
        # Clean old active detections
        plates_to_remove = []
        for plate, detection in self.active_detections.items():
            if current_time - detection['timestamp'] > self.active_detection_timeout:
                plates_to_remove.append(plate)
        
        for plate in plates_to_remove:
            del self.active_detections[plate]

    def get_best_plate(self):
        """Get best plate using reference code logic"""
        if not self.plate_scores:
            return None, 0, 0
        best_plate = None
        best_score = 0

        for plate, scores in self.plate_scores.items():
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            frequency = len(scores) / self.frame_window
            final_score = avg_score * 0.7 + frequency * 0.3

            if final_score > best_score:
                best_score = final_score
                best_plate = plate

        if best_plate:
            frequency = len(self.plate_scores[best_plate]) / self.frame_window
            return best_plate, best_score, frequency
        return None, 0, 0
    
    def get_top_plates(self, n=5):
        """Return the top n most frequently detected plates"""
        return self.all_plates.most_common(n)
    
    def get_stable_plates(self):
        """Return plates that have been consistently detected and are stable"""
        stable_plates = []
        current_time = time.time()
        
        for plate, scores in self.plate_scores.items():
            recent_count = len(self.recent_detections.get(plate, []))
            
            # Check stability criteria
            if len(scores) >= self.min_scores and recent_count >= self.min_recent:
                avg_score = sum(scores[-self.min_scores:]) / self.min_scores
                
                if avg_score > self.confidence_threshold:
                    # Check if detection is recent
                    if plate in self.recent_detections:
                        last_detection = max(self.recent_detections[plate])
                        if current_time - last_detection < 2:
                            stable_plates.append((plate, avg_score))
                            # Clear recent detections to avoid repeated confirmations
                            self.recent_detections[plate] = []
                            # Mark as confirmed
                            self.confirmed_plates[plate] += 1
        
        return stable_plates
    
    def get_active_detections(self):
        """Return currently active detections for drawing"""
        return self.active_detections.copy()
    
    def get_detection_summary(self):
        """Get comprehensive summary of all detections"""
        summary = {}
        for plate in self.all_detections_count:
            scores = self.plate_scores.get(plate, [])
            if scores:
                summary[plate] = {
                    'count': self.all_detections_count[plate],
                    'confirmed_count': self.confirmed_plates[plate],
                    'avg_confidence': sum(scores) / len(scores),
                    'max_confidence': max(scores),
                    'total_detections': len(scores)
                }
        return summary
    
    def get_most_detected_plate(self):
        """Return the most frequently detected license plate (by total count)"""
        if not self.all_detections_count:
            return None, 0
        
        # Find plate with highest total detection count
        best_plate = max(self.all_detections_count.items(), key=lambda x: x[1])
        return best_plate[0], best_plate[1]
    
    def is_detection_complete(self):
        """Check if detection period is complete"""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.detection_duration
    
    def get_remaining_time(self):
        """Get remaining detection time in seconds"""
        if self.start_time is None:
            return self.detection_duration
        elapsed = time.time() - self.start_time
        return max(0, self.detection_duration - elapsed)

def extract_text_with_tesseract(image):
    """Enhanced tesseract configuration with more parameters from reference code"""
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_do_invert=0'
    try:
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        return ""

def extract_text_with_easyocr(reader, image):
    """Extract text using EasyOCR with optimized settings from reference code"""
    try:
        # Lower confidence threshold to catch more potential plates
        results = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=20)
        cleaned_texts = []
        for (bbox, text, confidence) in results:
            # Only keep alphanumeric characters
            text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
            # More lenient length check
            if 2 <= len(text) <= 10:
                cleaned_texts.append((text, confidence))
        return cleaned_texts
    except Exception as e:
        return []

def enhance_plate_region(plate_region):
    """Apply multiple image enhancement techniques from reference code"""
    # Resize for better OCR performance if too small
    min_height = 40  # Increased from 30 as in reference code
    if plate_region.shape[0] < min_height:
        scale = min_height / plate_region.shape[0]
        width = int(plate_region.shape[1] * scale)
        plate_region = cv2.resize(plate_region, (width, min_height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    if len(plate_region.shape) == 3:
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_region

    # Apply bilateral filter to remove noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Try different thresholding techniques
    thresh_methods = []

    # Method 1: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh_methods.append(adaptive_thresh)

    # Method 2: Otsu's thresholding
    _, otsu_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_methods.append(otsu_thresh)

    # Method 3: Simple binary threshold
    _, simple_thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    thresh_methods.append(simple_thresh)

    # Apply morphological closing to all thresholded images
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Using reference code kernel size
    processed_images = []
    for thresh in thresh_methods:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(closed)

    # Also add the original grayscale and filtered images
    processed_images.append(gray)
    processed_images.append(filtered)

    return processed_images, plate_region

def detect_license_plates(frame, license_plate_detector, reader):
    """Enhanced license plate detection using reference code logic"""
    try:
        # Make a copy of the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Apply a series of image enhancements from reference code
        # 1. Histogram equalization for better contrast
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # 2. Apply gamma correction to adjust brightness for night videos
        gamma = 1.5  # Increased gamma for night videos as in reference code
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
        processed_frame = cv2.LUT(processed_frame, lookUpTable)
        
        # 3. Increase contrast
        alpha = 1.3  # Contrast control (1.0 means no change)
        beta = 10    # Brightness control (0 means no change)
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=alpha, beta=beta)

        # Apply detection with lower confidence threshold from reference code
        conf_threshold = 0.35  # Lower confidence threshold
        detections = license_plate_detector(processed_frame, conf=conf_threshold)[0]
        frame_plates = []

        for detection in detections.boxes.data.tolist():
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                if score < conf_threshold:
                    continue
                    
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
                # Add padding around the license plate region
                padding = 5  # Using reference code padding
                y1 = max(0, y1 - padding)
                y2 = min(frame.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                plate_region = processed_frame[y1:y2, x1:x2]
                
                if plate_region.size == 0:
                    continue
                
                try:
                    # Apply multiple image processing techniques from reference code
                    processed_images, resized_plate = enhance_plate_region(plate_region)
                    
                    all_texts = []
                    
                    # Try OCR on all processed images
                    for img in processed_images:
                        # Try with Tesseract
                        tesseract_text = extract_text_with_tesseract(img)
                        if tesseract_text:
                            all_texts.append((tesseract_text, 0.8))
                    
                    # Try with EasyOCR on the original and resized plate
                    easyocr_texts = extract_text_with_easyocr(reader, resized_plate)
                    all_texts.extend(easyocr_texts)
                    
                    # Process all detected texts
                    for text, confidence in all_texts:
                        if not text.strip():
                            continue
                        
                        text_height = resized_plate.shape[0]
                        relative_text_size = text_height / (y2 - y1)
                        
                        # Clean text - keep only alphanumeric characters
                        cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                        
                        # More lenient length check for night videos
                        if 2 <= len(cleaned_text) <= 10:
                            frame_plates.append({
                                'plate_number': cleaned_text,
                                'confidence': confidence * score,
                                'text_size': relative_text_size,
                                'bbox': (x1, y1, x2, y2)
                            })
                except Exception as e:
                    logging.warning(f"OCR processing error: {str(e)}")
        
        return frame_plates
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return []

def write_plate_to_file(plate_number):
    """Write the detected license plate to the specified file"""
    file_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/detection.txt"
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write only the plate number to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(plate_number)
            
        logging.info(f"✅ License plate '{plate_number}' written to {file_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Error writing to file {file_path}: {str(e)}")
        return False

def open_dashboard_page():
    """Open the dashboard page in the default web browser"""
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    # Assuming the Django server is running on port 8000 on the same machine
    dashboard_url = f"http://{ip}:8000/dashboard/"

    try:
        import webbrowser
        webbrowser.open(dashboard_url)
        logging.info(f"✅ Opening dashboard page: {dashboard_url}")
        return True
    except Exception as e:
        logging.error(f"❌ Error opening dashboard page: {str(e)}")
        return False

# def start_django_server():
#     """Start the Django server if it's not running"""
#     django_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/BE/User"
    
#     try:
#         # Check if server is already running
#         response = requests.get("http://127.0.0.1:8000/api/license-plate/", timeout=5)
#         logging.info("✅ Django server is already running")
#         return True
#     except:
#         logging.info("🚀 Starting Django server...")
#         try:
#             # Change to Django directory and start server in background
#             subprocess.Popen(
#                 ["python", "manage.py", "runserver"],
#                 cwd=django_path,
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.DEVNULL
#             )
#             # Wait a moment for server to start
#             time.sleep(3)
#             logging.info("✅ Django server started")
#             return True
#         except Exception as e:
#             logging.error(f"❌ Failed to start Django server: {str(e)}")
#             return False

def detect_license_plates_for_duration(camera_id, duration=60):
    """Main function to detect license plates for specified duration with enhanced stability"""
    print(f"🔍 Enhanced License Plate Detection on Camera {camera_id}")
    print(f"⏱️  Detection Duration: {duration} seconds")
    print("🔧 Features: Stability scoring, temporal consistency, enhanced OCR")
    print("=" * 60)
    
    try:
        # Test camera connection
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Cannot connect to camera {camera_id}")
            return None
        
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Cannot read from camera {camera_id}")
            cap.release()
            return None
        
        cap.release()
        logging.info(f"Camera {camera_id} connection verified")
        
        # Load license plate detection model
        license_plate_model = "license_plate_detector.pt"
        
        if not os.path.isfile(license_plate_model):
            logging.error(f"Model file not found at: {license_plate_model}")
            return None
        
        logging.info("Loading license plate detection model...")
        license_plate_detector = YOLO(license_plate_model)
        logging.info("✅ License plate model loaded")
        
        # Initialize EasyOCR
        logging.info("Initializing EasyOCR...")
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            logging.info("✅ EasyOCR initialized with GPU")
        except Exception as e:
            logging.warning(f"EasyOCR GPU failed: {str(e)}, trying CPU...")
            try:
                reader = easyocr.Reader(['en'], gpu=False)
                logging.info("✅ EasyOCR initialized with CPU")
            except Exception as e2:
                logging.error(f"EasyOCR initialization failed: {str(e2)}")
                return None
        
        # Initialize enhanced plate detector
        plate_detector = PlateDetection(camera_id, duration)
        
        # Start camera capture
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 25)  # Consistent FPS
        
        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        logging.info(f"🚀 Starting enhanced {duration}-second license plate detection...")
        
        frame_count = 0
        last_plate_check = time.time()
        confirmed_detections = []
        
        while not plate_detector.is_detection_complete():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Process every 2nd frame to speed up analysis (from reference code)
            process_interval = 2
            if frame_count % process_interval == 0:
                plates = detect_license_plates(frame, license_plate_detector, reader)
                
                for plate in plates:
                    plate_number = plate['plate_number']
                    confidence = plate['confidence']
                    text_size = plate['text_size']
                    bbox = plate['bbox']
                    
                    # Add to enhanced detector with scoring
                    plate_detector.add_detection(plate_number, confidence, text_size, bbox)
            
            # Draw all active detections with enhanced visualization
            active_detections = plate_detector.get_active_detections()
            for plate_number, detection in active_detections.items():
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                score = detection['score']
                
                # Color coding based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for lower confidence
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw detailed label
                label = f"{plate_number} (C:{confidence:.2f}, S:{score:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Check for stable plates at intervals
            check_interval = 2.0
            if current_time - last_plate_check > check_interval:
                stable_plates = plate_detector.get_stable_plates()
                for plate_number, avg_score in stable_plates:
                    confirmed_detections.append({
                        'plate': plate_number,
                        'score': avg_score,
                        'timestamp': current_time
                    })
                    
                    # Show confirmation on screen
                    cv2.putText(frame, f"CONFIRMED: {plate_number}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    logging.info(f"🎯 CONFIRMED DETECTION: {plate_number} (Score: {avg_score:.3f})")
                
                last_plate_check = current_time
            
            # Enhanced status display
            remaining = plate_detector.get_remaining_time()
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show current leading detection (by total count)
            best_plate, best_count = plate_detector.get_most_detected_plate()
            if best_plate:
                cv2.putText(frame, f"Leading: {best_plate} ({best_count}x)", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show top plates from reference code logic
            top_plates = plate_detector.get_top_plates(3)
            if top_plates:
                cv2.putText(frame, f"Top 3:", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                for i, (plate, count) in enumerate(top_plates[:3]):
                    cv2.putText(frame, f"{i+1}. {plate} ({count})", (10, 150 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show confirmed count
            cv2.putText(frame, f"Confirmed: {len(confirmed_detections)}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(f"Enhanced License Plate Detection - Camera {camera_id}", frame)
            
            # Allow early exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logging.info("Early exit requested")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Get final results - return the plate with maximum detections
        most_detected_plate, detection_count = plate_detector.get_most_detected_plate()
        
        # Print comprehensive results
        print(f"\n🏁 ENHANCED DETECTION COMPLETE")
        print("=" * 60)
        
        if most_detected_plate:
            print(f"🎯 MOST DETECTED PLATE: {most_detected_plate}")
            print(f"📊 TOTAL DETECTION COUNT: {detection_count}")
            print(f"✅ TOTAL CONFIRMATIONS: {len(confirmed_detections)}")
            
            # Write the most detected plate to file
            if write_plate_to_file(most_detected_plate):
                print(f"💾 License plate saved to detection.txt")
                
                # Open dashboard page
                print(f"🌐 Opening dashboard page...")
                if open_dashboard_page():
                    print(f"✅ Dashboard page opened successfully")
                else:
                    print(f"❌ Failed to open dashboard page")
            else:
                print(f"❌ Failed to save license plate to file")
            
            # Show top plates summary
            top_plates = plate_detector.get_top_plates(5)
            print(f"\n📋 TOP 5 DETECTED PLATES:")
            for i, (plate, count) in enumerate(top_plates):
                print(f"  {i+1}. {plate} - Detected {count} times")
            
            # Show detailed summary
            summary = plate_detector.get_detection_summary()
            print(f"\n📋 DETAILED DETECTION SUMMARY:")
            for plate, stats in sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True):
                count = stats['count']
                confirmed_count = stats['confirmed_count']
                avg_conf = stats['avg_confidence']
                max_conf = stats['max_confidence']
                print(f"  {plate}: {count} total detections, {confirmed_count} confirmations")
                print(f"    Avg confidence: {avg_conf:.3f}, Max: {max_conf:.3f}")
            
            # Show chronological confirmations
            if confirmed_detections:
                print(f"\n🕒 CHRONOLOGICAL CONFIRMATIONS:")
                for i, detection in enumerate(confirmed_detections, 1):
                    plate = detection['plate']
                    score = detection['score']
                    timestamp = detection['timestamp']
                    rel_time = timestamp - plate_detector.start_time
                    print(f"  {i}. {plate} at {rel_time:.1f}s (Score: {score:.3f})")
        else:
            print("❌ NO LICENSE PLATES DETECTED")
            most_detected_plate = None
        
        return most_detected_plate
        
    except Exception as e:
        logging.error(f"Enhanced detection error: {str(e)}")
        return None

def main():
    """Main function to run license plate detection"""
    try:
        # Default parameters
        camera_id = 0  # Default camera
        detection_duration = 60  # Default 60 seconds
        
        # Check if Django server is running and start if needed
        logging.info("🔧 Checking Django server status...")
        # start_django_server()
        
        # Parse command line arguments if provided
        if len(sys.argv) > 1:
            try:
                camera_id = int(sys.argv[1])
            except ValueError:
                logging.warning(f"Invalid camera ID: {sys.argv[1]}, using default: {camera_id}")
        
        if len(sys.argv) > 2:
            try:
                detection_duration = int(sys.argv[2])
            except ValueError:
                logging.warning(f"Invalid duration: {sys.argv[2]}, using default: {detection_duration}")
        
        # Run the enhanced detection
        result = detect_license_plates_for_duration(camera_id, detection_duration)
        
        if result:
            print(f"\n🎉 FINAL RESULT: {result}")
            logging.info(f"Detection completed successfully. Final result: {result}")
        else:
            print(f"\n❌ No license plate detected in {detection_duration} seconds")
            logging.info("Detection completed with no results")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Detection interrupted by user")
        logging.info("Detection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logging.error(f"Main function error: {str(e)}")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        print(f"🔚 Program terminated")

if __name__ == "__main__":
    main()