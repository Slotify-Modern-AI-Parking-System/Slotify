#Use Input Video File 1

# from ultralytics import YOLO
# import cv2
# import numpy as np
# import pytesseract
# import re
# from collections import defaultdict, Counter
# import easyocr
# import time
# import os
 
# class PlateDetection:
#     def __init__(self):
#         self.plate_scores = defaultdict(list)
#         self.plate_history = Counter()
#         self.frame_window = 30
#         self.all_plates = Counter()  # Track all detected plates across the entire video
 
#     def update_scores(self, plate_number, confidence, text_size):
#         score = confidence * 0.6 + text_size * 0.4
#         self.plate_scores[plate_number].append(score)
#         self.plate_history[plate_number] += 1
#         self.all_plates[plate_number] += 1  # Increment the count for this plate
        
#         # Keep only the last window of scores for real-time processing
#         if len(self.plate_scores[plate_number]) > self.frame_window:
#             self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
 
#     def get_best_plate(self):
#         if not self.plate_scores:
#             return None, 0, 0
#         best_plate = None
#         best_score = 0
 
#         for plate, scores in self.plate_scores.items():
#             avg_score = sum(scores) / len(scores)
#             frequency = len(scores) / self.frame_window
#             final_score = avg_score * 0.7 + frequency * 0.3
 
#             if final_score > best_score:
#                 best_score = final_score
#                 best_plate = plate
 
#         frequency = len(self.plate_scores[best_plate]) / self.frame_window
#         return best_plate, best_score, frequency
    
#     def get_top_plates(self, n=5):
#         """Return the top n most frequently detected plates"""
#         return self.all_plates.most_common(n)
 
# def extract_text_with_tesseract(image):
#     # Enhanced tesseract configuration with more parameters
#     custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_do_invert=0'
#     try:
#         text = pytesseract.image_to_string(image, config=custom_config)
#         return text.strip()
#     except Exception as e:
#         print(f"Tesseract error: {str(e)}")
#         return ""
 
# def extract_text_with_easyocr(reader, image):
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
#         print(f"EasyOCR error: {str(e)}")
#         return []
 
# def enhance_plate_region(plate_region):
#     """Apply multiple image enhancement techniques to improve plate readability"""
#     # Resize for better OCR performance if too small
#     min_height = 40 # Increased from 30
#     if plate_region.shape[0] < min_height:
#         scale = min_height / plate_region.shape[0]
#         width = int(plate_region.shape[1] * scale)
#         plate_region = cv2.resize(plate_region, (width, min_height))
 
#     # Convert to grayscale
#     gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
 
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
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     processed_images = []
#     for thresh in thresh_methods:
#         closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#         processed_images.append(closed)
 
#     # Also add the original grayscale and filtered images
#     processed_images.append(gray)
#     processed_images.append(filtered)
 
#     return processed_images, plate_region
 
# def detect_license_plates(frame, license_plate_detector, reader):
#     # Make a copy of the frame to avoid modifying the original
#     processed_frame = frame.copy()
 
#     # Apply a series of image enhancements
#     # 1. Histogram equalization for better contrast
#     gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
 
#     # 2. Apply gamma correction to adjust brightness for night videos
#     gamma = 1.5 # Increased gamma for night videos
#     lookUpTable = np.empty((1, 256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
#     processed_frame = cv2.LUT(processed_frame, lookUpTable)
 
#     # 3. Increase contrast
#     alpha = 1.3 # Contrast control (1.0 means no change)
#     beta = 10 # Brightness control (0 means no change)
#     processed_frame = cv2.convertScaleAbs(processed_frame, alpha=alpha, beta=beta)
 
#     # Apply detection with lower confidence threshold
#     detections = license_plate_detector(processed_frame, conf=0.35)[0] # Lower confidence threshold
#     frame_plates = []
 
#     for detection in detections.boxes.data.tolist():
#         if len(detection) >= 6:
#             x1, y1, x2, y2, score, class_id = detection
#             if score < 0.35: # Lower threshold to catch more plates
#                 continue
 
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
 
#             # Add padding around the license plate region
#             padding = 5
#             y1 = max(0, y1 - padding)
#             y2 = min(frame.shape[0], y2 + padding)
#             x1 = max(0, x1 - padding)
#             x2 = min(frame.shape[1], x2 + padding)
 
#             if x1 >= x2 or y1 >= y2:
#                 continue
 
#             plate_region = processed_frame[y1:y2, x1:x2]
 
#             if plate_region.size == 0:
#                 continue
 
#             try:
#                 # Apply multiple image processing techniques
#                 processed_images, resized_plate = enhance_plate_region(plate_region)
 
#                 all_texts = []
 
#                 # Try OCR on all processed images
#                 for img in processed_images:
#                     # Try with Tesseract
#                     tesseract_text = extract_text_with_tesseract(img)
#                     if tesseract_text:
#                         all_texts.append((tesseract_text, 0.8))
 
#                 # Try with EasyOCR on the original and resized plate
#                 easyocr_texts = extract_text_with_easyocr(reader, resized_plate)
#                 all_texts.extend(easyocr_texts)
 
#                 # Process all detected texts
#                 for text, confidence in all_texts:
#                     if not text.strip():
#                         continue
 
#                     text_height = resized_plate.shape[0]
#                     relative_text_size = text_height / (y2 - y1)
 
#                     # Clean text - keep only alphanumeric characters
#                     cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
 
#                     # More lenient length check for night videos
#                     if 2 <= len(cleaned_text) <= 10:
#                         frame_plates.append({
#                             'plate_number': cleaned_text,
#                             'confidence': confidence * score,
#                             'text_size': relative_text_size,
#                             'bbox': (x1, y1, x2, y2),
#                             'image': resized_plate.copy()  # Save the plate image for display later
#                         })
#             except Exception as e:
#                 print(f"OCR processing error: {str(e)}")
 
#     return frame_plates
 
# def process_video(video_path, license_plate_detector, reader):
#     print(f"Starting to process video: {video_path}")
    
#     # Initialize video capture
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return None
    
#     # Get video info
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Video contains {total_frames} frames at {fps} FPS")
    
#     # Initialize plate tracker
#     plate_tracker = PlateDetection()
    
#     # Process frames
#     frame_count = 0
#     plate_images = {}  # Store the best image for each plate
#     plate_confidences = defaultdict(float)  # Track the best confidence for each plate
    
#     print("Processing frames...")
    
#     # Create a progress indicator
#     progress_interval = max(1, total_frames // 20)  # Show progress at 5% intervals
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_count += 1
        
#         # Show progress
#         if frame_count % progress_interval == 0:
#             progress = (frame_count / total_frames) * 100
#             print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
#         # Process every 2nd frame to speed up analysis
#         if frame_count % 2 == 0:
#             plates = detect_license_plates(frame, license_plate_detector, reader)
            
#             for plate in plates:
#                 plate_number = plate['plate_number']
#                 confidence = plate['confidence']
#                 text_size = plate['text_size']
                
#                 # Update tracker
#                 plate_tracker.update_scores(plate_number, confidence, text_size)
                
#                 # Store the best quality image for each plate
#                 combined_score = confidence * 0.6 + text_size * 0.4
#                 if combined_score > plate_confidences[plate_number]:
#                     plate_confidences[plate_number] = combined_score
#                     plate_images[plate_number] = plate['image']
    
#     # Release resources
#     cap.release()
    
#     # Get top plates
#     top_plates = plate_tracker.get_top_plates(5)
#     print(f"Processing complete. Found {len(plate_tracker.all_plates)} unique license plates.")
    
#     return top_plates, plate_images
 
# def create_result_display(top_plates, plate_images):
#     # Create a blank image for displaying results
#     result_img_height = 800
#     result_img_width = 600
#     result_img = np.ones((result_img_height, result_img_width, 3), dtype=np.uint8) * 255  # White background
    
#     # Add title
#     title = "Top 5 Detected License Plates"
#     cv2.putText(result_img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
#     # Display each plate
#     y_offset = 100
#     for i, (plate, count) in enumerate(top_plates):
#         # Display number and count
#         text = f"{i+1}. {plate} - Detected {count} times"
#         cv2.putText(result_img, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
#         # Display plate image if available
#         if plate in plate_images:
#             img = plate_images[plate]
            
#             # Resize to a standard width while maintaining aspect ratio
#             target_width = 400
#             aspect_ratio = img.shape[1] / img.shape[0]
#             target_height = int(target_width / aspect_ratio)
#             resized_img = cv2.resize(img, (target_width, target_height))
            
#             # Add to result image
#             y_pos = y_offset + 20
            
#             # Make sure the image fits
#             if y_pos + resized_img.shape[0] < result_img_height and resized_img.shape[1] < result_img_width-40:
#                 # Convert grayscale to BGR if needed
#                 if len(resized_img.shape) == 2:
#                     resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
                
#                 # Place the image at the right position
#                 result_img[y_pos:y_pos+resized_img.shape[0], 20:20+resized_img.shape[1]] = resized_img
                
#                 # Add a border around the image
#                 cv2.rectangle(result_img, (19, y_pos-1), (21+resized_img.shape[1], y_pos+resized_img.shape[0]+1), (0, 0, 0), 1)
                
#                 y_offset = y_pos + resized_img.shape[0] + 40
#             else:
#                 y_offset += 60  # Skip the image if it doesn't fit
#         else:
#             y_offset += 60
    
#     return result_img
 
# def main():
#     try:
#         # Specify your video file path
#         video_file_path = input("Enter the path to the video file: ")
        
#         # Check if the video file exists
#         if not os.path.isfile(video_file_path):
#             print(f"Error: Video file {video_file_path} not found.")
#             return
        
#         # Load model paths
#         license_plate_model = input("Enter the path to the license plate detector model: ")
        
#         if not os.path.isfile(license_plate_model):
#             print(f"Error: Model file not found at specified path")
#             return
        
#         # Load models
#         print("Loading detection models...")
#         license_plate_detector = YOLO(license_plate_model)
        
#         print("Initializing OCR reader (this might take a minute)...")
#         try:
#             reader = easyocr.Reader(['en'], gpu=True)
#             print("EasyOCR initialized with GPU support")
#         except Exception as e:
#             print(f"Error initializing EasyOCR with GPU: {str(e)}")
#             print("Falling back to CPU mode")
#             reader = easyocr.Reader(['en'], gpu=False)
        
#         # Process the video without showing it
#         start_time = time.time()
#         top_plates, plate_images = process_video(video_file_path, license_plate_detector, reader)
#         processing_time = time.time() - start_time
        
#         if top_plates:
#             print(f"\nVideo processing completed in {processing_time:.2f} seconds")
#             print("\nTop 5 most frequently detected license plates:")
#             for i, (plate, count) in enumerate(top_plates):
#                 print(f"{i+1}. {plate} - Detected {count} times")
            
#             # Create and display result image
#             result_img = create_result_display(top_plates, plate_images)
            
#             # Show the results
#             cv2.namedWindow("License Plate Detection Results", cv2.WINDOW_NORMAL)
#             cv2.imshow("License Plate Detection Results", result_img)
            
#             # Save the results
#             results_path = "license_plate_results.jpg"
#             cv2.imwrite(results_path, result_img)
#             print(f"\nResults saved to {results_path}")
            
#             print("\nPress any key to exit...")
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         else:
#             print("No license plates detected in the video.")
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         cv2.destroyAllWindows()
 
# if __name__ == "__main__":
#     main()


#Uses Live Camera Feed 1

# from ultralytics import YOLO
# import cv2
# import numpy as np
# from sort.sort import *
# import pytesseract
# import re
# from collections import defaultdict, Counter
# import easyocr

# class PlateDetection:
#     def __init__(self):
#         self.plate_scores = defaultdict(list)
#         self.plate_history = Counter()
#         self.frame_window = 30

#     def update_scores(self, plate_number, confidence, text_size):
#         score = confidence * 0.6 + text_size * 0.4
#         self.plate_scores[plate_number].append(score)
#         self.plate_history[plate_number] += 1
#         if len(self.plate_scores[plate_number]) > self.frame_window:
#             self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
            
#     def get_best_plate(self):
#         if not self.plate_scores:
#             return None, 0, 0
#         best_plate = None
#         best_score = 0
        
#         for plate, scores in self.plate_scores.items():
#             avg_score = sum(scores) / len(scores)
#             frequency = len(scores) / self.frame_window
#             final_score = avg_score * 0.7 + frequency * 0.3
            
#             if final_score > best_score:
#                 best_score = final_score
#                 best_plate = plate
                
#         frequency = len(self.plate_scores[best_plate]) / self.frame_window
#         return best_plate, best_score, frequency

# def extract_text_with_tesseract(image):
#     custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#     return pytesseract.image_to_string(image, config=custom_config)

# def extract_text_with_easyocr(reader, image):
#     results = reader.readtext(image)
#     cleaned_texts = []
#     for (bbox, text, confidence) in results:
#         text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
#         if 2 <= len(text) <= 8:
#             cleaned_texts.append((text, confidence))
#     return cleaned_texts

# def detect_license_plates(frame, license_plate_detector, reader):
#     # Apply histogram equalization to improve contrast
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)  # Histogram equalization
    
#     # Apply gamma correction to adjust brightness
#     gamma = 1.2  # Adjust this value to control the correction
#     lookUpTable = np.empty((1, 256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
#     frame = cv2.LUT(frame, lookUpTable)
    
#     # Apply detection
#     detections = license_plate_detector(frame)[0]
#     frame_plates = []
    
#     for detection in detections.boxes.data.tolist():
#         if len(detection) >= 6:
#             x1, y1, x2, y2, score, class_id = detection
#             if score < 0.5:
#                 continue
            
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
#             if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
#                 continue
            
#             plate_region = frame[y1:y2, x1:x2]
            
#             if plate_region.size == 0:
#                 continue
            
#             try:
#                 min_height = 30
#                 if plate_region.shape[0] < min_height:
#                     scale = min_height / plate_region.shape[0]
#                     width = int(plate_region.shape[1] * scale)
#                     plate_region = cv2.resize(plate_region, (width, min_height))
                
#                 gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
#                 thresh = cv2.adaptiveThreshold(
#                     gray_plate, 255, 
#                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                     cv2.THRESH_BINARY, 11, 2
#                 )

#                 # Apply morphological closing to fill gaps
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#                 thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
#                 # Extract text from Tesseract and EasyOCR
#                 tesseract_text = extract_text_with_tesseract(thresh)
#                 easyocr_texts = extract_text_with_easyocr(reader, plate_region)
                
#                 # Combine results from both OCRs
#                 all_texts = [(tesseract_text.strip(), 0.8)] + easyocr_texts
                
#                 for text, confidence in all_texts:
#                     if not text.strip():
#                         continue
                    
#                     text_height = plate_region.shape[0]
#                     relative_text_size = text_height / (y2 - y1)
                    
#                     cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
#                     if 2 <= len(cleaned_text) <= 8:
#                         frame_plates.append({
#                             'plate_number': cleaned_text,
#                             'confidence': confidence * score,
#                             'text_size': relative_text_size,
#                             'bbox': (x1, y1, x2, y2)
#                         })
#             except Exception as e:
#                 print(f"OCR processing error: {str(e)}")
    
#     return frame_plates

# def main():
#     try:
#         license_plate_detector = YOLO('/Users/jainamdoshi/Desktop/ALPR/license_plate_detector.pt')
#         car_detector = YOLO('/Users/jainamdoshi/Desktop/ALPR/yolo11n.pt')  
#         cap = cv2.VideoCapture(1)
#         reader = easyocr.Reader(['en'], gpu=True)
        
#         if not cap.isOpened():
#             raise Exception("Error opening video file")
        
#         plate_tracker = PlateDetection()
#         car_tracker = Sort()  # Initialize SORT tracker
#         car_id_map = {}  # Dictionary to track car IDs
#         car_plates_map = {}  # Dictionary to track car ID and corresponding license plate
#         previous_car_ids = set()  # Track car IDs from the previous frame
#         car_counter = 0  # Unique car count

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             plates = detect_license_plates(frame, license_plate_detector, reader)
            
#             # Detect cars using YOLO
#             car_detections = car_detector(frame)[0]
#             car_bboxes = []

#             for car in car_detections.boxes.data.tolist():
#                 if len(car) >= 6:
#                     x1, y1, x2, y2, score, class_id = car
#                     if score < 0.5:
#                         continue
                    
#                     car_bboxes.append([x1, y1, x2, y2, score])

#             # Convert list to NumPy array for SORT tracking
#             car_bboxes = np.array(car_bboxes).reshape(-1, 5)  # Ensures it always has shape (n, 5)

#             # Update tracker with current frame detections
#             tracked_cars = car_tracker.update(car_bboxes)

#             current_car_ids = set()  # Track current frame car IDs

#             for car in tracked_cars:
#                 x1, y1, x2, y2, track_id = map(int, car)

#                 current_car_ids.add(track_id)  # Add to current frame's set

#                 if track_id not in car_id_map:
#                     car_counter += 1
#                     car_id_map[track_id] = f"Car{car_counter}"

#                 car_label = car_id_map[track_id]

#                 # Draw bounding box and label for car
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
#                 cv2.putText(frame, car_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#             # Track and associate license plates with cars
#             for plate in plates:
#                 plate_tracker.update_scores(plate['plate_number'], plate['confidence'], plate['text_size'])

#                 # Associate license plate with the nearest tracked car
#                 best_plate, score, frequency = plate_tracker.get_best_plate()
#                 closest_car_id = None
#                 min_distance = float('inf')

#                 for car in tracked_cars:
#                     x1, y1, x2, y2, track_id = map(int, car)

#                     plate_x, plate_y, _, _ = plate['bbox']
#                     car_center_x, car_center_y = (x1 + x2) // 2, (y1 + y2) // 2
#                     distance = np.sqrt((car_center_x - plate_x) ** 2 + (car_center_y - plate_y) ** 2)

#                     if distance < min_distance:
#                         min_distance = distance
#                         closest_car_id = track_id

#                 if closest_car_id is not None:
#                     car_plates_map[closest_car_id] = plate['plate_number']  # Map carId to plate number

#             # Detect cars that have exited the frame
#             exited_cars = previous_car_ids - current_car_ids  # Cars in previous frame but not in current frame
#             for exited_car_id in exited_cars:
#                 if exited_car_id in car_plates_map:
#                     print(f"Car ID: {car_id_map[exited_car_id]}, License Plate: {car_plates_map[exited_car_id]} exited the frame.")
#                     del car_plates_map[exited_car_id]  # Remove from tracking

#             previous_car_ids = current_car_ids  # Update for next frame

#             # Display the frame
#             cv2.imshow('License Plate and Car Detection', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     except Exception as e:
#         print(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()



#Uses Live Camera Feed 2

# from ultralytics import YOLO
# import cv2
# import numpy as np
# from sort.sort import *
# import pytesseract
# import re
# from collections import defaultdict, Counter
# import easyocr
# import threading
# import queue
# import time
# import os

# class PlateDetection:
#     def __init__(self):
#         self.plate_scores = defaultdict(list)
#         self.plate_history = Counter()
#         self.frame_window = 30

#     def update_scores(self, plate_number, confidence, text_size):
#         score = confidence * 0.6 + text_size * 0.4
#         self.plate_scores[plate_number].append(score)
#         self.plate_history[plate_number] += 1
#         if len(self.plate_scores[plate_number]) > self.frame_window:
#             self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
            
#     def get_best_plate(self):
#         if not self.plate_scores:
#             return None, 0, 0
#         best_plate = None
#         best_score = 0
        
#         for plate, scores in self.plate_scores.items():
#             avg_score = sum(scores) / len(scores)
#             frequency = len(scores) / self.frame_window
#             final_score = avg_score * 0.7 + frequency * 0.3
            
#             if final_score > best_score:
#                 best_score = final_score
#                 best_plate = plate
                
#         frequency = len(self.plate_scores[best_plate]) / self.frame_window
#         return best_plate, best_score, frequency

# # Enhanced Shared data structure for communicating between cameras
# class SharedCarData:
#     def __init__(self):
#         self.lock = threading.Lock()
#         self.known_cars = {}  # Map license plates to car IDs
#         self.car_data = {}    # Store additional car data
#         self.global_car_id_counter = 0  # Global counter for consistent IDs across cameras

#     def get_next_car_id(self):
#         with self.lock:
#             self.global_car_id_counter += 1
#             return f"Car{self.global_car_id_counter}"

#     def add_car(self, license_plate, camera_id):
#         with self.lock:
#             # Check if this license plate is already known
#             if license_plate in self.known_cars:
#                 car_id = self.known_cars[license_plate]
#                 # Update the car data to show it was seen by this camera
#                 if camera_id not in self.car_data[car_id]['seen_by_cameras']:
#                     self.car_data[car_id]['seen_by_cameras'].append(camera_id)
#                 self.car_data[car_id]['last_seen'] = time.time()
#                 return car_id
            
#             # If it's a new car, create a new entry
#             car_id = self.get_next_car_id()
#             self.known_cars[license_plate] = car_id
#             self.car_data[car_id] = {
#                 'license_plate': license_plate,
#                 'first_seen': time.time(),
#                 'last_seen': time.time(),
#                 'seen_by_cameras': [camera_id],
#                 'first_camera': camera_id
#             }
#             return car_id
    
#     def update_car(self, license_plate, camera_id):
#         with self.lock:
#             if license_plate in self.known_cars:
#                 car_id = self.known_cars[license_plate]
#                 self.car_data[car_id]['last_seen'] = time.time()
#                 if camera_id not in self.car_data[car_id]['seen_by_cameras']:
#                     self.car_data[car_id]['seen_by_cameras'].append(camera_id)
#                 return car_id, self.car_data[car_id]['first_camera'] == 1  # Return True if first seen by camera 1
#             return None, False
    
#     def get_car_id_by_plate(self, license_plate):
#         with self.lock:
#             return self.known_cars.get(license_plate)
    
#     def get_car_data(self, car_id):
#         with self.lock:
#             return self.car_data.get(car_id, {})

# def extract_text_with_tesseract(image):
#     custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#     return pytesseract.image_to_string(image, config=custom_config)

# def extract_text_with_easyocr(reader, image):
#     results = reader.readtext(image)
#     cleaned_texts = []
#     for (bbox, text, confidence) in results:
#         text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
#         if 2 <= len(text) <= 8:
#             cleaned_texts.append((text, confidence))
#     return cleaned_texts

# def detect_license_plates(frame, license_plate_detector, reader):
#     # Apply histogram equalization to improve contrast
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)  # Histogram equalization
    
#     # Apply gamma correction to adjust brightness
#     gamma = 1.2  # Adjust this value to control the correction
#     lookUpTable = np.empty((1, 256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
#     frame = cv2.LUT(frame, lookUpTable)
    
#     # Apply detection
#     detections = license_plate_detector(frame)[0]
#     frame_plates = []
    
#     for detection in detections.boxes.data.tolist():
#         if len(detection) >= 6:
#             x1, y1, x2, y2, score, class_id = detection
#             if score < 0.5:
#                 continue
            
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
#             if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
#                 continue
            
#             plate_region = frame[y1:y2, x1:x2]
            
#             if plate_region.size == 0:
#                 continue
            
#             try:
#                 min_height = 30
#                 if plate_region.shape[0] < min_height:
#                     scale = min_height / plate_region.shape[0]
#                     width = int(plate_region.shape[1] * scale)
#                     plate_region = cv2.resize(plate_region, (width, min_height))
                
#                 gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
#                 thresh = cv2.adaptiveThreshold(
#                     gray_plate, 255, 
#                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                     cv2.THRESH_BINARY, 11, 2
#                 )

#                 # Apply morphological closing to fill gaps
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#                 thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
#                 # Extract text from Tesseract and EasyOCR
#                 tesseract_text = extract_text_with_tesseract(thresh)
#                 easyocr_texts = extract_text_with_easyocr(reader, plate_region)
                
#                 # Combine results from both OCRs
#                 all_texts = [(tesseract_text.strip(), 0.8)] + easyocr_texts
                
#                 for text, confidence in all_texts:
#                     if not text.strip():
#                         continue
                    
#                     text_height = plate_region.shape[0]
#                     relative_text_size = text_height / (y2 - y1)
                    
#                     cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
#                     if 2 <= len(cleaned_text) <= 8:
#                         frame_plates.append({
#                             'plate_number': cleaned_text,
#                             'confidence': confidence * score,
#                             'text_size': relative_text_size,
#                             'bbox': (x1, y1, x2, y2)
#                         })
#             except Exception as e:
#                 print(f"OCR processing error: {str(e)}")
    
#     return frame_plates

# # Add this function to detect and show available cameras
# def identify_cameras(max_to_check=5):
#     for i in range(max_to_check):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 # Display a preview with camera index
#                 cv2.namedWindow(f"Camera Index {i}", cv2.WINDOW_NORMAL)
#                 cv2.imshow(f"Camera Index {i}", frame)
#                 cv2.waitKey(1000)  # Display for 1 second
#             cap.release()
#     cv2.waitKey(0)  # Wait for key press
#     cv2.destroyAllWindows()
    
#     # Ask user to choose camera indices
#     camera1_index = int(input("Enter index for iPhone connected directly (Camera 1): "))
#     camera2_index = int(input("Enter index for iPhone with Epoch Cam (Camera 2): "))
#     return camera1_index, camera2_index

# # In main(), you could call this before starting the camera threads:
# # camera1_source, camera2_source = identify_cameras()

# def run_camera(camera_id, video_source, shared_data, license_plate_detector, car_detector, reader, result_queue):
#     # Display a message about the source
#     source_type = "Video File" if isinstance(video_source, str) and os.path.isfile(video_source) else "Network Camera" if isinstance(video_source, str) else "Webcam"
#     print(f"Camera {camera_id} using {source_type}: {video_source}")
    
#     # Add retry logic for network cameras
#     max_retries = 5
#     retry_count = 0
    
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(video_source)
        
#         if not cap.isOpened():
#             print(f"Error opening video source {video_source} for camera {camera_id}. Retry {retry_count+1}/{max_retries}")
#             retry_count += 1
#             time.sleep(2)  # Wait before retrying
#             continue
            
#         plate_tracker = PlateDetection()
#         car_tracker = Sort()  # Initialize SORT tracker
#         car_id_map = {}  # Dictionary to track car IDs
#         car_plates_map = {}  # Dictionary to track car ID and corresponding license plate
#         previous_car_ids = set()  # Track car IDs from the previous frame
        
#         window_name = f"Camera {camera_id}"
#         frame_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             frame_count += 1
            
#             if not ret:
#                 # For video files, loop back to the beginning
#                 if isinstance(video_source, str) and os.path.isfile(video_source):
#                     print(f"Restarting video file for Camera {camera_id}")
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
#                     ret, frame = cap.read()
#                     if not ret:
#                         break  # If still can't read, then exit
#                 else:
#                     print(f"Camera {camera_id} disconnected. Attempting to reconnect...")
#                     break  # Break inner loop to try reconnecting
            
#             plates = detect_license_plates(frame, license_plate_detector, reader)
            
#             # Detect cars using YOLO
#             car_detections = car_detector(frame)[0]
#             car_bboxes = []

#             for car in car_detections.boxes.data.tolist():
#                 if len(car) >= 6:
#                     x1, y1, x2, y2, score, class_id = car
#                     if score < 0.5:
#                         continue
                    
#                     car_bboxes.append([x1, y1, x2, y2, score])

#             # Convert list to NumPy array for SORT tracking
#             if len(car_bboxes) > 0:
#                 car_bboxes = np.array(car_bboxes).reshape(-1, 5)
#                 # Update tracker with current frame detections
#                 tracked_cars = car_tracker.update(car_bboxes)
#             else:
#                 tracked_cars = np.array([])

#             current_car_ids = set()  # Track current frame car IDs

#             for car in tracked_cars:
#                 x1, y1, x2, y2, track_id = map(int, car)
#                 current_car_ids.add(track_id)  # Add to current frame's set

#                 # Get an existing global ID for this local track_id, or create a new one
#                 if track_id not in car_id_map:
#                     # This is a new car for this camera's tracker
#                     # We'll associate it with a license plate later if possible
#                     car_id_map[track_id] = f"Cam{camera_id}_Track{track_id}"  # Temporary ID

#                 car_label = car_id_map[track_id]
#                 car_color = (255, 0, 0)  # Default blue box

#                 # Draw bounding box and label for car
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), car_color, 2)
#                 cv2.putText(frame, car_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, car_color, 2)

#             # Track and associate license plates with cars
#             for plate in plates:
#                 plate_tracker.update_scores(plate['plate_number'], plate['confidence'], plate['text_size'])
                
#                 # Draw plate bounding box
#                 x1, y1, x2, y2 = plate['bbox']
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for plates
                
#                 # Associate license plate with the nearest tracked car
#                 closest_car_id = None
#                 min_distance = float('inf')

#                 for car in tracked_cars:
#                     car_x1, car_y1, car_x2, car_y2, track_id = map(int, car)

#                     plate_x, plate_y, _, _ = plate['bbox']
#                     car_center_x, car_center_y = (car_x1 + car_x2) // 2, (car_y1 + car_y2) // 2
#                     distance = np.sqrt((car_center_x - plate_x) ** 2 + (car_center_y - plate_y) ** 2)

#                     if distance < min_distance and distance < 200:  # Add a distance threshold
#                         min_distance = distance
#                         closest_car_id = track_id

#                 if closest_car_id is not None:
#                     local_car_id = car_id_map[closest_car_id]
                    
#                     # Update the shared data with this plate, get the global car ID
#                     global_car_id, from_primary = shared_data.update_car(plate['plate_number'], camera_id)
                    
#                     if global_car_id is None:
#                         # This is a new car, register it
#                         global_car_id = shared_data.add_car(plate['plate_number'], camera_id)
#                         is_new = True
#                     else:
#                         is_new = False
                    
#                     # Update our local tracking to use the global ID
#                     car_id_map[closest_car_id] = global_car_id
#                     car_plates_map[closest_car_id] = plate['plate_number']
                    
#                     # Display information based on where the car was first seen
#                     if camera_id == 1:
#                         # Primary camera
#                         if is_new:
#                             plate_text = f"{plate['plate_number']} (New)"
#                             cv2.putText(frame, plate_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                         else:
#                             plate_text = f"{plate['plate_number']} (Known)"
#                             cv2.putText(frame, plate_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     else:
#                         # Secondary camera - check if this car was first seen by the primary camera
#                         if from_primary:
#                             # This car was first seen by camera 1 (primary)
#                             plate_text = f"{plate['plate_number']} (From Primary: {global_car_id})"
#                             # Draw red box for cars identified from primary camera
#                             for car in tracked_cars:
#                                 car_x1, car_y1, car_x2, car_y2, track_id = map(int, car)
#                                 if track_id == closest_car_id:
#                                     cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 0, 255), 3)  # Red for recognized cars
#                             cv2.putText(frame, plate_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                             print(f"Camera {camera_id} identified car from primary: {plate['plate_number']} as {global_car_id}")
#                         else:
#                             # This car was first seen by this or another non-primary camera
#                             plate_text = f"{plate['plate_number']} (First seen here)"
#                             cv2.putText(frame, plate_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow

#             # Detect cars that have exited the frame
#             exited_cars = previous_car_ids - current_car_ids  # Cars in previous frame but not in current frame
#             for exited_car_id in exited_cars:
#                 if exited_car_id in car_plates_map:
#                     print(f"Camera {camera_id} - Car ID: {car_id_map[exited_car_id]}, License Plate: {car_plates_map[exited_car_id]} exited the frame.")
#                     # Don't delete from tracking in case it reappears

#             previous_car_ids = current_car_ids  # Update for next frame

#             # Add camera identifier with source type
#             source_label = f"Camera {camera_id} - {source_type}"
#             cv2.putText(frame, source_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             # Instead of displaying here, send to the main thread
#             result_queue.put((camera_id, frame))

#             # For video files, add a delay to slow down playback
#             if isinstance(video_source, str) and os.path.isfile(video_source):
#                 time.sleep(0.05)  # Adjust this value to control playback speed
            
#             # Check if we should reconnect to improve reliability
#             if frame_count % 300 == 0 and not isinstance(video_source, str):  # Every 300 frames for live feed
#                 print(f"Camera {camera_id} - Checking connection health")

#         # If we reach here, the camera was disconnected or video ended
#         cap.release()
#         if not isinstance(video_source, str) or not os.path.isfile(video_source):
#             print(f"Camera {camera_id} disconnected. Attempting to reconnect in 3 seconds...")
#             time.sleep(3)
#             retry_count += 1
#         else:
#             # This was a video file that ended
#             print(f"Video file for Camera {camera_id} has ended.")
#             break  # Exit reconnection loop for video files
            
#     print(f"Camera {camera_id} thread exiting after {retry_count} reconnection attempts")

# def main():
#     try:
#         # Path to the video file for the first camera
#         # Replace this with your video file path
#         video_file_path = "/Users/jainamdoshi/Desktop/ALPR/Videos/seltos2.MOV"  # CHANGE THIS TO YOUR REAL VIDEO FILE
        
       
        
#         # Check if the video file exists
#         if not os.path.isfile(video_file_path):
#             print(f"Warning: Video file {video_file_path} not found.")
#             print("Please specify a valid video file path for testing.")
#             print("For now, using webcam as a fallback for both cameras.")
#             video_file_path = 0  # Fallback to webcam
        
#         # Load models once to be shared between threads
#         models_base_path = os.path.expanduser('~/Desktop/ALPR')  # Default path
        
#         # Check if model files exist in the default path
#         license_plate_model = os.path.join(models_base_path, 'license_plate_detector.pt')
#         car_model = os.path.join(models_base_path, 'yolo11n.pt')
        
#         if not os.path.isfile(license_plate_model) or not os.path.isfile(car_model):
#             print(f"Warning: Model files not found at {models_base_path}")
#             # Ask user for custom model paths
#             license_plate_model = input("Enter the path to the license plate detector model: ")
#             car_model = input("Enter the path to the car detector model: ")
        
#         # Load models
#         print("Loading detection models...")
#         license_plate_detector = YOLO(license_plate_model)
#         car_detector = YOLO(car_model)
        
#         print("Initializing OCR reader (this might take a minute)...")
#         try:
#             reader = easyocr.Reader(['en'], gpu=True)
#             print("EasyOCR initialized with GPU support")
#         except Exception as e:
#             print(f"Error initializing EasyOCR with GPU: {str(e)}")
#             print("Falling back to CPU mode")
#             reader = easyocr.Reader(['en'], gpu=False)
        
#         # Shared data structure for both cameras
#         shared_data = SharedCarData()
        
#         # Create windows in the main thread
#         cv2.namedWindow("Camera 1 - Primary (Video File)", cv2.WINDOW_NORMAL)
#         cv2.namedWindow("Camera 2 - Secondary (Live Feed)", cv2.WINDOW_NORMAL)
        
#         # Queue for communicating frames from threads to main thread
#         result_queue = queue.Queue()
        
#         # Camera 1 uses the video file, Camera 2 uses webcam
#         camera1_source = "/Users/jainamdoshi/Desktop/ALPR/Videos/mazdaNight.mov"
#         camera2_source = 1 # Built-in webcam
        
#         print(f"Starting Camera 1 (Primary) with video file: {camera1_source}")
#         print(f"Starting Camera 2 (Secondary) with webcam ID: {camera2_source}")
        
#         # Create and start threads for each camera
#         camera1_thread = threading.Thread(
#             target=run_camera, 
#             args=(1, camera1_source, shared_data, license_plate_detector, car_detector, reader, result_queue),
#             daemon=True
#         )
        
#         camera2_thread = threading.Thread(
#             target=run_camera, 
#             args=(2, camera2_source, shared_data, license_plate_detector, car_detector, reader, result_queue),
#             daemon=True
#         )
        
#         camera1_thread.start()
#         camera2_thread.start()
        
#         print("Both camera threads started. Press 'q' to exit.")
        
#         # Main loop to display frames from both cameras
#         while True:
#             try:
#                 # Non-blocking get from queue with timeout
#                 camera_id, frame = result_queue.get(timeout=0.1)
#                 window_name = f"Camera {camera_id} - {'Primary (Video File)' if camera_id == 1 else 'Secondary (Live Feed)'}"
#                 cv2.imshow(window_name, frame)
                
#                 # Check for quit key
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     print("Quit key pressed. Exiting...")
#                     break
                    
#             except queue.Empty:
#                 # No frames in queue, just continue
#                 continue
#             except Exception as e:
#                 print(f"Error in display loop: {str(e)}")
#                 break
        
#         # Clean up
#         cv2.destroyAllWindows()
#         print("Exiting program")

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import re
from collections import defaultdict, Counter
import time
import os
import threading
import logging
from datetime import datetime
import json
import requests
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iphone_vehicle_tracking.log'),
        logging.StreamHandler()
    ]
)

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.plate_to_id = {}
        self.next_id = 1
        self.lock = threading.Lock()
        
    def add_vehicle(self, plate_number, camera_id, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.lock:
            if plate_number in self.plate_to_id:
                vehicle_id = self.plate_to_id[plate_number]
                self.vehicles[vehicle_id]['last_seen'] = timestamp
                if camera_id not in self.vehicles[vehicle_id]['camera_detections']:
                    self.vehicles[vehicle_id]['camera_detections'][camera_id] = []
                self.vehicles[vehicle_id]['camera_detections'][camera_id].append(timestamp)
                
                # Check if this is a cross-camera detection
                cameras_seen = list(self.vehicles[vehicle_id]['camera_detections'].keys())
                if len(cameras_seen) > 1:
                    logging.info(f"🎯 CROSS-CAMERA MATCH! Vehicle ID {vehicle_id} (Plate: {plate_number}) now seen on cameras: {cameras_seen}")
                else:
                    logging.info(f"Vehicle ID {vehicle_id} (Plate: {plate_number}) detected again on iPhone {camera_id}")
            else:
                vehicle_id = self.next_id
                self.next_id += 1
                self.plate_to_id[plate_number] = vehicle_id
                self.vehicles[vehicle_id] = {
                    'plate': plate_number,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'camera_detections': {camera_id: [timestamp]}
                }
                logging.info(f"🆕 NEW VEHICLE: ID {vehicle_id} (Plate: {plate_number}) registered on iPhone {camera_id}")
                
            return vehicle_id

    def get_all_vehicles(self):
        with self.lock:
            return self.vehicles.copy()

    def save_tracking_data(self, filename):
        with self.lock:
            data = {}
            for vehicle_id, vehicle_data in self.vehicles.items():
                data[vehicle_id] = {
                    'plate': vehicle_data['plate'],
                    'first_seen': vehicle_data['first_seen'].isoformat(),
                    'last_seen': vehicle_data['last_seen'].isoformat(),
                    'camera_detections': {
                        str(cam_id): [ts.isoformat() for ts in timestamps]
                        for cam_id, timestamps in vehicle_data['camera_detections'].items()
                    }
                }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

class PlateDetection:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.plate_scores = defaultdict(list)
        self.recent_detections = defaultdict(list)
        self.frame_window = 10  # Reduced for iPhone streaming
        self.confidence_threshold = 0.4  # Adjusted for iPhone cameras
        
    def update_scores(self, plate_number, confidence, text_size):
        current_time = time.time()
        score = confidence * 0.7 + text_size * 0.3
        
        self.plate_scores[plate_number].append(score)
        self.recent_detections[plate_number].append(current_time)
        
        # Keep only recent detections (last 8 seconds)
        cutoff_time = current_time - 8
        self.recent_detections[plate_number] = [
            t for t in self.recent_detections[plate_number] if t > cutoff_time
        ]
        
        # Keep only the last window of scores
        if len(self.plate_scores[plate_number]) > self.frame_window:
            self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
    
    def get_stable_plates(self):
        """Return plates that have been consistently detected"""
        stable_plates = []
        current_time = time.time()
        
        for plate, scores in self.plate_scores.items():
            recent_count = len(self.recent_detections[plate])
            
            if len(scores) >= 4 and recent_count >= 2:  # Relaxed for iPhone streaming
                avg_score = sum(scores[-4:]) / 4
                if avg_score > self.confidence_threshold:
                    last_detection = max(self.recent_detections[plate])
                    if current_time - last_detection < 2:
                        stable_plates.append((plate, avg_score))
                        self.recent_detections[plate] = []
        
        return stable_plates

def enhanced_ocr(image_region):
    """Enhanced OCR function optimized for iPhone camera streams"""
    try:
        # Resize if too small
        if image_region.shape[0] < 40:
            scale = 40 / image_region.shape[0]
            width = int(image_region.shape[1] * scale)
            image_region = cv2.resize(image_region, (width, 40), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        
        # Enhanced preprocessing for iPhone camera quality
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.equalizeHist(gray)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # OCR with license plate specific config
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Clean and validate the text
        cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
        
        if 3 <= len(cleaned_text) <= 8:
            # Simple confidence based on text clarity
            confidence = min(0.9, len(cleaned_text) / 8.0 + 0.3)
            return cleaned_text, confidence
        
        return None, 0
        
    except Exception as e:
        return None, 0

def detect_license_plates_iphone(frame, license_plate_detector, camera_id):
    """License plate detection optimized for iPhone camera streams"""
    try:
        # iPhone camera optimization
        enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=15)
        
        # Denoise for better detection
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Detection with adjusted confidence for iPhone cameras
        detections = license_plate_detector(enhanced, conf=0.25)[0]
        frame_plates = []
        
        for detection in detections.boxes.data.tolist():
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                if score < 0.25:
                    continue
                    
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
                # Add padding for better OCR
                padding = 12
                y1 = max(0, y1 - padding)
                y2 = min(frame.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                plate_region = enhanced[y1:y2, x1:x2]
                
                if plate_region.size == 0:
                    continue
                
                # Enhanced OCR
                text, confidence = enhanced_ocr(plate_region)
                
                if text and len(text) >= 3:
                    text_height = plate_region.shape[0]
                    relative_text_size = text_height / max(1, (y2 - y1))
                    
                    frame_plates.append({
                        'plate_number': text,
                        'confidence': confidence * score,
                        'text_size': relative_text_size,
                        'bbox': (x1, y1, x2, y2),
                        'camera_id': camera_id
                    })
        
        return frame_plates
        
    except Exception as e:
        print(f"Detection error on iPhone {camera_id}: {str(e)}")
        return []

def test_iphone_connection(camera_source):
    """Test iPhone camera connection"""
    print(f"Testing connection to: {camera_source}")
    
    # Test HTTP connection first if it's a URL
    if isinstance(camera_source, str) and camera_source.startswith('http'):
        try:
            response = requests.get(camera_source, timeout=5, stream=True)
            if response.status_code == 200:
                print("✅ HTTP connection successful")
            else:
                print(f"❌ HTTP connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ HTTP test failed: {e}")
            return False
    
    # Test OpenCV connection
    cap = cv2.VideoCapture(camera_source)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print("✅ OpenCV connection successful")
            return True
        else:
            print("❌ OpenCV connection failed - no frame received")
            return False
    else:
        print("❌ OpenCV connection failed - cannot open stream")
        return False

def iphone_camera_thread(camera_id, camera_source, license_plate_detector, vehicle_tracker, plate_detector, display_queue):
    """iPhone camera thread with optimized streaming"""
    print(f"Starting iPhone {camera_id} thread with source: {camera_source}")
    
    # Test connection first
    if not test_iphone_connection(camera_source):
        logging.error(f"iPhone {camera_id}: Failed to connect to {camera_source}")
        return
    
    # Initialize capture with iPhone optimizations
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        logging.error(f"iPhone {camera_id}: Failed to open stream")
        return
    
    # iPhone streaming optimizations
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    logging.info(f"iPhone {camera_id}: Connected successfully")
    
    frame_count = 0
    last_plate_check = time.time()
    last_display_update = time.time()
    consecutive_failures = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:  # 30 consecutive failures
                logging.error(f"iPhone {camera_id}: Too many consecutive failures, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(camera_source)
                consecutive_failures = 0
                continue
            time.sleep(0.1)
            continue
            
        consecutive_failures = 0  # Reset on successful frame
        frame_count += 1
        current_time = time.time()
        
        # Process every 4th frame for iPhone performance
        if frame_count % 4 == 0:
            plates = detect_license_plates_iphone(frame, license_plate_detector, camera_id)
            
            for plate in plates:
                plate_number = plate['plate_number']
                confidence = plate['confidence']
                text_size = plate['text_size']
                bbox = plate['bbox']
                
                plate_detector.update_scores(plate_number, confidence, text_size)
                
                # Draw detection box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Detected: {plate_number}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Check for stable plates every 2.5 seconds
        if current_time - last_plate_check > 2.5:
            stable_plates = plate_detector.get_stable_plates()
            for plate_number, avg_score in stable_plates:
                vehicle_id = vehicle_tracker.add_vehicle(plate_number, camera_id)
                
                # Show confirmed detection with larger text
                cv2.putText(frame, f"CONFIRMED - ID: {vehicle_id}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Plate: {plate_number}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            last_plate_check = current_time
        
        # Add iPhone info
        cv2.putText(frame, f"iPhone {camera_id}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Connection status
        status_color = (0, 255, 0) if consecutive_failures == 0 else (0, 255, 255)
        cv2.putText(frame, "CONNECTED", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Update display
        if current_time - last_display_update > 0.08:  # ~12 FPS display
            if not display_queue.full():
                try:
                    display_queue.put((camera_id, frame.copy()), block=False)
                except:
                    pass
            last_display_update = current_time
    
    cap.release()

def main():
    print("📱 iPhone Vehicle Tracking System")
    print("=" * 50)
    
    try:
        # Get model path
        license_plate_model = input("Enter path to license plate model (or press Enter for 'yolov8n.pt'): ").strip()
        if not license_plate_model:
            license_plate_model = "yolov8n.pt"
        
        print(f"Using model: {license_plate_model}")
        
        # iPhone camera setup with examples
        print("\n📱 iPhone Camera Setup:")
        print("Recommended apps and URL formats:")
        print("  IP Webcam Pro: http://192.168.1.XXX:8080/video")
        print("  EpocCam: Use the URL shown in the app")
        print("  AtomicCam: http://192.168.1.XXX:8888/mjpeg")
        print("  DroidCam: http://192.168.1.XXX:4747/mjpegfeed?640x480")
        print("\nMake sure both iPhones are on the same WiFi network!")
        
        camera1_source = input("\nEnter iPhone 1 stream URL: ").strip()
        camera2_source = input("Enter iPhone 2 stream URL: ").strip()
        
        if not camera1_source or not camera2_source:
            print("❌ Both camera URLs are required!")
            return
        
        # Load model
        print("\n🔄 Loading YOLO model...")
        try:
            license_plate_detector = YOLO(license_plate_model)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you have the YOLO model file in the current directory")
            return
        
        # Initialize components
        vehicle_tracker = VehicleTracker()
        plate_detector1 = PlateDetection(1)
        plate_detector2 = PlateDetection(2)
        
        # Create display queue
        from queue import Queue
        display_queue = Queue(maxsize=6)
        
        print("\n🚀 Starting iPhone camera connections...")
        
        # Start camera threads
        thread1 = threading.Thread(target=iphone_camera_thread, args=(
            1, camera1_source, license_plate_detector, 
            vehicle_tracker, plate_detector1, display_queue
        ))
        thread2 = threading.Thread(target=iphone_camera_thread, args=(
            2, camera2_source, license_plate_detector, 
            vehicle_tracker, plate_detector2, display_queue
        ))
        
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        
        print("\n✅ System started! Waiting for iPhone connections...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save tracking data")
        print("  'r' - Print summary report")
        print("  'c' - Clear all tracking data")
        
        # Display loop
        camera_frames = {1: None, 2: None}
        
        while True:
            # Get latest frames
            try:
                while not display_queue.empty():
                    camera_id, frame = display_queue.get(block=False)
                    camera_frames[camera_id] = frame
            except:
                pass
            
            # Display frames
            for cam_id in [1, 2]:
                if camera_frames[cam_id] is not None:
                    frame = camera_frames[cam_id]
                    height, width = frame.shape[:2]
                    
                    # Resize for display if too large
                    if width > 900:
                        scale = 900 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imshow(f"iPhone {cam_id} Stream", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"iphone_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                vehicle_tracker.save_tracking_data(filename)
                print(f"💾 Data saved to {filename}")
            elif key == ord('r'):
                print_summary(vehicle_tracker)
            elif key == ord('c'):
                vehicle_tracker.vehicles.clear()
                vehicle_tracker.plate_to_id.clear()
                vehicle_tracker.next_id = 1
                print("🧹 Tracking data cleared")
        
        # Cleanup
        cv2.destroyAllWindows()
        print_summary(vehicle_tracker)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        cv2.destroyAllWindows()

def print_summary(vehicle_tracker):
    vehicles = vehicle_tracker.get_all_vehicles()
    print(f"\n📊 SESSION SUMMARY")
    print("=" * 50)
    print(f"Total vehicles tracked: {len(vehicles)}")
    
    cross_camera_count = 0
    for vehicle_id, data in vehicles.items():
        cameras_seen = list(data['camera_detections'].keys())
        status = "🎯 CROSS-CAMERA" if len(cameras_seen) > 1 else "📱 SINGLE-CAMERA"
        if len(cameras_seen) > 1:
            cross_camera_count += 1
        
        duration = (data['last_seen'] - data['first_seen']).total_seconds()
        print(f"{status} - Vehicle {vehicle_id} (Plate: {data['plate']}) - iPhones: {cameras_seen} - Duration: {duration:.1f}s")
    
    print(f"\n🎯 Cross-camera matches: {cross_camera_count}")
    print(f"📱 Single-camera only: {len(vehicles) - cross_camera_count}")

if __name__ == "__main__":
    main()
