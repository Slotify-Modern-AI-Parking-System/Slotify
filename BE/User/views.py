# from django.shortcuts import render, get_object_or_404
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# from django.utils.decorators import method_decorator
# from django.views import View
# import json
# import subprocess
# import threading
# import uuid
# import os
# import sys
# from .models import LicensePlateDetection

# class ParkingEntryView(View):
#     def get(self, request):
#         """Render the parking entry page"""
#         return render(request, 'entry.html')

# @csrf_exempt
# @require_http_methods(["POST"])
# def start_detection(request):
#     """Start license plate detection process"""
#     try:
#         data = json.loads(request.body)
#         camera_id = data.get('camera_id', 0)
#         duration = data.get('duration', 60)
        
#         # Generate unique session ID
#         session_id = str(uuid.uuid4())
        
#         # Start detection in background thread
#         def run_detection():
#             try:
#                 # Path to your license plate detection script
#                 script_path = "C:\\Users\\jigsp\\Desktop\\Slotify\\ALPR\\main2.py"
# # Update this path
                
#                 # Run the script as subprocess
#                 result = subprocess.run(
#                     [sys.executable, script_path, str(camera_id)],
#                     capture_output=True,
#                     text=True,
#                     timeout=duration + 30  # Add buffer time
#                 )
                
#                 # Parse the output to extract detected plate
#                 output_lines = result.stdout.strip().split('\n')
#                 detected_plate = None
                
#                 for line in output_lines:
#                     if "FINAL RESULT:" in line:
#                         detected_plate = line.split("FINAL RESULT:")[-1].strip()
#                         break
#                     elif "MOST DETECTED PLATE:" in line:
#                         detected_plate = line.split("MOST DETECTED PLATE:")[-1].strip()
#                         break
                
#                 if detected_plate and detected_plate != "None":
#                     # Save detection to database
#                     LicensePlateDetection.objects.create(
#                         plate_number=detected_plate,
#                         camera_id=camera_id,
#                         session_id=session_id,
#                         is_confirmed=True
#                     )
                
#             except subprocess.TimeoutExpired:
#                 print(f"Detection timeout for session {session_id}")
#             except Exception as e:
#                 print(f"Detection error for session {session_id}: {str(e)}")
        
#         # Start detection thread
#         detection_thread = threading.Thread(target=run_detection)
#         detection_thread.daemon = True
#         detection_thread.start()
        
#         return JsonResponse({
#             'success': True,
#             'session_id': session_id,
#             'message': 'Detection started'
#         })
        
#     except Exception as e:
#         return JsonResponse({
#             'success': False,
#             'error': str(e)
#         }, status=500)

# @require_http_methods(["GET"])
# def check_detection_status(request, session_id):
#     """Check if detection is complete and return results"""
#     try:
#         detection = LicensePlateDetection.objects.filter(
#             session_id=session_id
#         ).first()
        
#         if detection:
#             return JsonResponse({
#                 'success': True,
#                 'detected': True,
#                 'plate_number': detection.plate_number,
#                 'detection_time': detection.detection_time.isoformat(),
#                 'is_confirmed': detection.user_confirmed
#             })
#         else:
#             return JsonResponse({
#                 'success': True,
#                 'detected': False,
#                 'message': 'Detection in progress...'
#             })
            
#     except Exception as e:
#         return JsonResponse({
#             'success': False,
#             'error': str(e)
#         }, status=500)

# @csrf_exempt
# @require_http_methods(["POST"])
# def confirm_plate(request, session_id):
#     """User confirms the detected license plate"""
#     try:
#         detection = get_object_or_404(LicensePlateDetection, session_id=session_id)
        
#         data = json.loads(request.body)
#         is_correct = data.get('is_correct', False)
#         corrected_plate = data.get('corrected_plate', '')
        
#         if is_correct:
#             detection.user_confirmed = True
#             detection.save()
            
#             return JsonResponse({
#                 'success': True,
#                 'message': 'License plate confirmed',
#                 'plate_number': detection.plate_number
#             })
#         else:
#             # If user provides correction
#             if corrected_plate:
#                 detection.plate_number = corrected_plate
#                 detection.user_confirmed = True
#                 detection.save()
                
#                 return JsonResponse({
#                     'success': True,
#                     'message': 'License plate corrected and confirmed',
#                     'plate_number': detection.plate_number
#                 })
#             else:
#                 return JsonResponse({
#                     'success': False,
#                     'message': 'Please provide the correct license plate number'
#                 })
                
#     except Exception as e:
#         return JsonResponse({
#             'success': False,
#             'error': str(e)
#         }, status=500)


# views.py
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import threading
import time
from datetime import datetime
from collections import deque
import logging
import subprocess
import sys
import os

# Global variables to manage detection state
detection_status = {
    'is_running': False,
    'current_plate': None,
    'detection_time': None,
    'confidence': 0,
    'awaiting_confirmation': False,
    'camera_id': 0
}

# Store recent detections
recent_detections = deque(maxlen=10)
detection_process = None

logger = logging.getLogger(__name__)

def index(request):
    """Main dashboard view"""
    context = {
        'detection_status': detection_status,
        'recent_detections': list(recent_detections)
    }
    return render(request, 'entry.html', context)

@require_http_methods(["POST"])
@csrf_exempt
def start_detection(request):
    """Start the car detection process"""
    global detection_process, detection_status
    
    try:
        data = json.loads(request.body)
        camera_id = data.get('camera_id', 0)
        
        if detection_status['is_running']:
            return JsonResponse({
                'success': False, 
                'message': 'Detection is already running'
            })
        
        # Update status
        detection_status.update({
            'is_running': True,
            'camera_id': camera_id,
            'current_plate': None,
            'awaiting_confirmation': False
        })
        
        # Start car detection script in background
        script_path = "C:\\Users\\jigsp\\Desktop\\Slotify\\ALPR\\carDetector.py"

        
        detection_process = subprocess.Popen([
            sys.executable, script_path, str(camera_id)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        logger.info(f"Started car detection process for camera {camera_id}")
        
        return JsonResponse({
            'success': True,
            'message': f'Detection started on camera {camera_id}'
        })
        
    except Exception as e:
        logger.error(f"Error starting detection: {str(e)}")
        detection_status['is_running'] = False
        return JsonResponse({
            'success': False,
            'message': f'Error starting detection: {str(e)}'
        })

@require_http_methods(["POST"])
@csrf_exempt
def stop_detection(request):
    """Stop the car detection process"""
    global detection_process, detection_status
    
    try:
        if detection_process:
            detection_process.terminate()
            detection_process.wait(timeout=5)
            detection_process = None
        
        detection_status.update({
            'is_running': False,
            'current_plate': None,
            'awaiting_confirmation': False
        })
        
        logger.info("Detection process stopped")
        
        return JsonResponse({
            'success': True,
            'message': 'Detection stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping detection: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error stopping detection: {str(e)}'
        })

@require_http_methods(["POST"])
@csrf_exempt
def plate_detected(request):
    """Endpoint called by the license plate detection script"""
    global detection_status, recent_detections
    
    try:
        data = json.loads(request.body)
        plate_number = data.get('plate_number')
        confidence = data.get('confidence', 0)
        camera_id = data.get('camera_id', 0)
        
        if not plate_number:
            return JsonResponse({
                'success': False,
                'message': 'No plate number provided'
            })
        
        # Update detection status
        detection_status.update({
            'current_plate': plate_number,
            'detection_time': datetime.now(),
            'confidence': confidence,
            'awaiting_confirmation': True,
            'camera_id': camera_id
        })
        
        # Add to recent detections
        recent_detections.append({
            'plate': plate_number,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'camera_id': camera_id,
            'status': 'pending'
        })
        
        logger.info(f"License plate detected: {plate_number} (confidence: {confidence})")
        
        return JsonResponse({
            'success': True,
            'message': 'Plate detection recorded'
        })
        
    except Exception as e:
        logger.error(f"Error recording plate detection: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error recording detection: {str(e)}'
        })

@require_http_methods(["POST"])
@csrf_exempt
def confirm_plate(request):
    """Handle user confirmation of detected license plate"""
    global detection_status, recent_detections
    
    try:
        data = json.loads(request.body)
        confirmed = data.get('confirmed', False)
        plate_number = data.get('plate_number')
        
        if not plate_number or plate_number != detection_status.get('current_plate'):
            return JsonResponse({
                'success': False,
                'message': 'Invalid plate confirmation'
            })
        
        # Update recent detections
        if recent_detections:
            recent_detections[-1]['status'] = 'confirmed' if confirmed else 'rejected'
            recent_detections[-1]['confirmed_at'] = datetime.now()
        
        # Log the confirmation
        status_text = "confirmed" if confirmed else "rejected"
        logger.info(f"License plate {plate_number} {status_text} by user")
        
        # Reset detection status
        detection_status.update({
            'current_plate': None,
            'awaiting_confirmation': False
        })
        
        # Here you can add logic to handle the confirmed/rejected plate
        # For example, save to database, send notifications, etc.
        if confirmed:
            # Handle confirmed plate (e.g., grant access, log entry, etc.)
            pass
        else:
            # Handle rejected plate (e.g., request manual entry, alert admin, etc.)
            pass
        
        return JsonResponse({
            'success': True,
            'message': f'Plate {status_text} successfully',
            'plate_number': plate_number,
            'confirmed': confirmed
        })
        
    except Exception as e:
        logger.error(f"Error confirming plate: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error confirming plate: {str(e)}'
        })

def get_status(request):
    """Get current detection status"""
    return JsonResponse({
        'success': True,
        'status': detection_status,
        'recent_detections': [
            {
                'plate': det['plate'],
                'confidence': det['confidence'],
                'timestamp': det['timestamp'].isoformat(),
                'camera_id': det['camera_id'],
                'status': det['status']
            } for det in recent_detections
        ]
    })

def detection_stream(request):
    """Server-sent events stream for real-time updates"""
    def event_stream():
        last_plate = None
        last_status = None
        
        while True:
            current_plate = detection_status.get('current_plate')
            current_status = detection_status.get('is_running')
            
            # Send updates when status changes or new plate detected
            if current_plate != last_plate or current_status != last_status:
                data = {
                    'type': 'status_update',
                    'detection_status': detection_status.copy(),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Convert datetime objects to strings for JSON serialization
                if data['detection_status']['detection_time']:
                    data['detection_status']['detection_time'] = data['detection_status']['detection_time'].isoformat()
                
                yield f"data: {json.dumps(data)}\n\n"
                
                last_plate = current_plate
                last_status = current_status
            
            time.sleep(1)  # Check for updates every second
    
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['Connection'] = 'keep-alive'
    return response