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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('license_plate_detection.log'),
        logging.StreamHandler()
    ]
)

class PlateDetection:
    def __init__(self, camera_id, detection_duration=60):
        self.camera_id = camera_id
        self.camera_name = f"Camera {camera_id}"
        self.detection_duration = detection_duration
        self.start_time = None
        
        self.plate_scores = defaultdict(list)
        self.recent_detections = defaultdict(list)
        self.active_detections = {}
        self.confirmed_plates = defaultdict(int)
        self.all_detections_count = defaultdict(int)
        self.plate_history = Counter()
        self.all_plates = Counter()
        
        self.frame_window = 30
        self.confidence_threshold = 0.6
        self.min_scores = 6
        self.min_recent = 4
        self.recent_time_window = 8
        self.active_detection_timeout = 5
        
    def update_scores(self, plate_number, confidence, text_size):
        score = confidence * 0.6 + text_size * 0.4
        self.plate_scores[plate_number].append(score)
        self.plate_history[plate_number] += 1
        self.all_plates[plate_number] += 1
        
        if len(self.plate_scores[plate_number]) > self.frame_window:
            self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
        
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        self.recent_detections[plate_number].append(current_time)
        self.all_detections_count[plate_number] += 1
        self._cleanup_old_data(current_time)
        
    def add_detection(self, plate_number, confidence, text_size, bbox):
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        score = confidence * 0.6 + text_size * 0.4
        
        self.plate_scores[plate_number].append(score)
        self.recent_detections[plate_number].append(current_time)
        self.plate_history[plate_number] += 1
        self.all_plates[plate_number] += 1
        self.all_detections_count[plate_number] += 1
        
        self.active_detections[plate_number] = {
            'bbox': bbox,
            'timestamp': current_time,
            'confidence': confidence,
            'score': score
        }
        
        self._cleanup_old_data(current_time)
        
        if len(self.plate_scores[plate_number]) > self.frame_window:
            self.plate_scores[plate_number] = self.plate_scores[plate_number][-self.frame_window:]
    
    def _cleanup_old_data(self, current_time):
        cutoff_time = current_time - self.recent_time_window
        for plate in list(self.recent_detections.keys()):
            self.recent_detections[plate] = [
                t for t in self.recent_detections[plate] if t > cutoff_time
            ]
            if not self.recent_detections[plate]:
                del self.recent_detections[plate]
        
        plates_to_remove = []
        for plate, detection in self.active_detections.items():
            if current_time - detection['timestamp'] > self.active_detection_timeout:
                plates_to_remove.append(plate)
        
        for plate in plates_to_remove:
            del self.active_detections[plate]

    def get_best_plate(self):
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
        return self.all_plates.most_common(n)
    
    def get_stable_plates(self):
        stable_plates = []
        current_time = time.time()
        
        for plate, scores in self.plate_scores.items():
            recent_count = len(self.recent_detections.get(plate, []))
            
            if len(scores) >= self.min_scores and recent_count >= self.min_recent:
                avg_score = sum(scores[-self.min_scores:]) / self.min_scores
                
                if avg_score > self.confidence_threshold:
                    if plate in self.recent_detections:
                        last_detection = max(self.recent_detections[plate])
                        if current_time - last_detection < 2:
                            stable_plates.append((plate, avg_score))
                            self.recent_detections[plate] = []
                            self.confirmed_plates[plate] += 1
        
        return stable_plates
    
    def get_active_detections(self):
        return self.active_detections.copy()
    
    def get_detection_summary(self):
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
        if not self.all_detections_count:
            return None, 0
        
        best_plate = max(self.all_detections_count.items(), key=lambda x: x[1])
        return best_plate[0], best_plate[1]
    
    def is_detection_complete(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.detection_duration
    
    def get_remaining_time(self):
        if self.start_time is None:
            return self.detection_duration
        elapsed = time.time() - self.start_time
        return max(0, self.detection_duration - elapsed)

# Cached models and readers
_license_plate_detector = None
_easyocr_reader = None
_tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_do_invert=0'

def get_cached_models():
    global _license_plate_detector, _easyocr_reader
    
    if _license_plate_detector is None:
        license_plate_model = "license_plate_detector.pt"
        if not os.path.isfile(license_plate_model):
            logging.error(f"Model file not found at: {license_plate_model}")
            return None, None
        
        logging.info("Loading license plate detection model...")
        _license_plate_detector = YOLO(license_plate_model)
        logging.info("✅ License plate model loaded and cached")
    
    if _easyocr_reader is None:
        logging.info("Initializing EasyOCR...")
        try:
            _easyocr_reader = easyocr.Reader(['en'], gpu=True)
            logging.info("✅ EasyOCR initialized with GPU and cached")
        except Exception as e:
            logging.warning(f"EasyOCR GPU failed: {str(e)}, trying CPU...")
            try:
                _easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logging.info("✅ EasyOCR initialized with CPU and cached")
            except Exception as e2:
                logging.error(f"EasyOCR initialization failed: {str(e2)}")
                return _license_plate_detector, None
    
    return _license_plate_detector, _easyocr_reader

def extract_text_with_tesseract(image):
    global _tesseract_config
    try:
        text = pytesseract.image_to_string(image, config=_tesseract_config)
        return text.strip()
    except Exception as e:
        return ""

def extract_text_with_easyocr(reader, image):
    try:
        results = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=20)
        cleaned_texts = []
        for (bbox, text, confidence) in results:
            text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
            if 2 <= len(text) <= 10:
                cleaned_texts.append((text, confidence))
        return cleaned_texts
    except Exception as e:
        return []

# Pre-compute gamma correction lookup table
_gamma_lut = None
def get_gamma_lut():
    global _gamma_lut
    if _gamma_lut is None:
        gamma = 1.5
        _gamma_lut = np.empty((1, 256), np.uint8)
        for i in range(256):
            _gamma_lut[0, i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
    return _gamma_lut

def enhance_plate_region(plate_region):
    min_height = 40
    if plate_region.shape[0] < min_height:
        scale = min_height / plate_region.shape[0]
        width = int(plate_region.shape[1] * scale)
        plate_region = cv2.resize(plate_region, (width, min_height), interpolation=cv2.INTER_CUBIC)

    if len(plate_region.shape) == 3:
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_region

    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh_methods = []

    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh_methods.append(adaptive_thresh)

    _, otsu_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_methods.append(otsu_thresh)

    _, simple_thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    thresh_methods.append(simple_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_images = []
    for thresh in thresh_methods:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(closed)

    processed_images.append(gray)
    processed_images.append(filtered)

    return processed_images, plate_region

def detect_license_plates(frame, license_plate_detector, reader):
    try:
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            scale_x = width / new_width
            scale_y = height / new_height
        else:
            frame_resized = frame
            scale_x = scale_y = 1.0
        
        processed_frame = frame_resized.copy()
        
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        gamma_lut = get_gamma_lut()
        processed_frame = cv2.LUT(processed_frame, gamma_lut)
        
        alpha = 1.3
        beta = 10
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=alpha, beta=beta)

        conf_threshold = 0.35
        detections = license_plate_detector(processed_frame, conf=conf_threshold)[0]
        frame_plates = []

        for detection in detections.boxes.data.tolist():
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection
                if score < conf_threshold:
                    continue
                
                # Scale coordinates back to original frame size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                padding = 5
                y1 = max(0, y1 - padding)
                y2 = min(frame.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                plate_region = frame[y1:y2, x1:x2]  # Use original frame for OCR
                
                if plate_region.size == 0:
                    continue
                
                try:
                    processed_images, resized_plate = enhance_plate_region(plate_region)
                    
                    all_texts = []
                    
                    for img in processed_images:
                        tesseract_text = extract_text_with_tesseract(img)
                        if tesseract_text:
                            all_texts.append((tesseract_text, 0.8))
                    
                    easyocr_texts = extract_text_with_easyocr(reader, resized_plate)
                    all_texts.extend(easyocr_texts)
                    
                    for text, confidence in all_texts:
                        if not text.strip():
                            continue
                        
                        text_height = resized_plate.shape[0]
                        relative_text_size = text_height / (y2 - y1)
                        
                        cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                        
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
    file_path = "detection.txt"
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(plate_number)
            
        logging.info(f"✅ License plate '{plate_number}' written to {file_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Error writing to file {file_path}: {str(e)}")
        return False

def open_dashboard_page():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    dashboard_url = f"http://{ip}:8000/dashboard/"

    try:
        import webbrowser
        webbrowser.open(dashboard_url)
        logging.info(f"✅ Opening dashboard page: {dashboard_url}")
        return True
    except Exception as e:
        logging.error(f"❌ Error opening dashboard page: {str(e)}")
        return False

def detect_license_plates_for_duration(camera_id, duration=60):
    print(f"🔍 Enhanced License Plate Detection on Camera {camera_id}")
    print(f"⏱️  Detection Duration: {duration} seconds")
    print("🔧 Features: Stability scoring, temporal consistency, enhanced OCR")
    print("=" * 60)
    
    try:
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
        
        license_plate_detector, reader = get_cached_models()
        if license_plate_detector is None or reader is None:
            return None
        
        plate_detector = PlateDetection(camera_id, duration)
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 25)
        
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
            
            # Process every 3rd frame for better performance
            process_interval = 3
            if frame_count % process_interval == 0:
                plates = detect_license_plates(frame, license_plate_detector, reader)
                
                for plate in plates:
                    plate_number = plate['plate_number']
                    confidence = plate['confidence']
                    text_size = plate['text_size']
                    bbox = plate['bbox']
                    
                    plate_detector.add_detection(plate_number, confidence, text_size, bbox)
            
            active_detections = plate_detector.get_active_detections()
            for plate_number, detection in active_detections.items():
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                score = detection['score']
                
                if confidence > 0.8:
                    color = (0, 255, 0)
                elif confidence > 0.6:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{plate_number} (C:{confidence:.2f}, S:{score:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            check_interval = 2.0
            if current_time - last_plate_check > check_interval:
                stable_plates = plate_detector.get_stable_plates()
                for plate_number, avg_score in stable_plates:
                    confirmed_detections.append({
                        'plate': plate_number,
                        'score': avg_score,
                        'timestamp': current_time
                    })
                    
                    cv2.putText(frame, f"CONFIRMED: {plate_number}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    logging.info(f"🎯 CONFIRMED DETECTION: {plate_number} (Score: {avg_score:.3f})")
                
                last_plate_check = current_time
            
            remaining = plate_detector.get_remaining_time()
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            best_plate, best_count = plate_detector.get_most_detected_plate()
            if best_plate:
                cv2.putText(frame, f"Leading: {best_plate} ({best_count}x)", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            top_plates = plate_detector.get_top_plates(3)
            if top_plates:
                cv2.putText(frame, f"Top 3:", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                for i, (plate, count) in enumerate(top_plates[:3]):
                    cv2.putText(frame, f"{i+1}. {plate} ({count})", (10, 150 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(frame, f"Confirmed: {len(confirmed_detections)}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f"Enhanced License Plate Detection - Camera {camera_id}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logging.info("Early exit requested")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        most_detected_plate, detection_count = plate_detector.get_most_detected_plate()
        
        print(f"\n🏁 ENHANCED DETECTION COMPLETE")
        print("=" * 60)
        
        if most_detected_plate:
            print(f"🎯 MOST DETECTED PLATE: {most_detected_plate}")
            print(f"📊 TOTAL DETECTION COUNT: {detection_count}")
            print(f"✅ TOTAL CONFIRMATIONS: {len(confirmed_detections)}")
            
            if write_plate_to_file(most_detected_plate):
                print(f"💾 License plate saved to detection.txt")
                
                print(f"🌐 Opening dashboard page...")
                if open_dashboard_page():
                    print(f"✅ Dashboard page opened successfully")
                else:
                    print(f"❌ Failed to open dashboard page")
            else:
                print(f"❌ Failed to save license plate to file")
            
            top_plates = plate_detector.get_top_plates(5)
            print(f"\n📋 TOP 5 DETECTED PLATES:")
            for i, (plate, count) in enumerate(top_plates):
                print(f"  {i+1}. {plate} - Detected {count} times")
            
            summary = plate_detector.get_detection_summary()
            print(f"\n📋 DETAILED DETECTION SUMMARY:")
            for plate, stats in sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True):
                count = stats['count']
                confirmed_count = stats['confirmed_count']
                avg_conf = stats['avg_confidence']
                max_conf = stats['max_confidence']
                print(f"  {plate}: {count} total detections, {confirmed_count} confirmations")
                print(f"    Avg confidence: {avg_conf:.3f}, Max: {max_conf:.3f}")
            
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
    try:
        camera_id = 0
        detection_duration = 60
        
        logging.info("🔧 Checking Django server status...")
        
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
        cv2.destroyAllWindows()
        print(f"🔚 Program terminated")

if __name__ == "__main__":
    main()