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
from Admin.models import *
from slotifyBE.models import *
from django.views.decorators.http import require_POST
from math import sqrt
import random
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

def welcome(request):
    return render(request, "welcome.html")

def dashboard(request):
    return render(request, "dashboard.html")

def destination(request):
    return render(request, "destination.html")

import json
import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.dispatch import Signal, receiver
from slotifyBE.models import ParkingLot

# Custom signals for parking lot login/logout
parking_lot_logged_in = Signal()
parking_lot_logged_out = Signal()

# Dictionary to keep track of running processes
running_processes = {}

@receiver(parking_lot_logged_in)
def start_car_detector(sender, lot_id, **kwargs):
    script_path = "../../ALPR/carDetector.py"
    """Start the car detector script when parking lot logs in"""
    
    
    # Check if script is already running for this lot
    if lot_id in running_processes and running_processes[lot_id].poll() is None:
        print(f"Car detector already running for lot {lot_id}")
        return
    
    try:
        # First, check if the script file exists
        import os
        if not os.path.exists(script_path):
            print(f"ERROR: Script not found at {script_path}")
            return
        
        print(f"Script found at {script_path}")
        
        # Use the same Python executable that's running Django
        import sys
        python_executable = sys.executable
        print(f"Using Python executable: {python_executable}")
        
        # Start the script as a subprocess with more verbose output
        process = subprocess.Popen([
            python_executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(script_path))
        
        # Store the process
        running_processes[lot_id] = process
        print(f"Car detector started for parking lot {lot_id} with PID {process.pid}")
        
        # Check if process started successfully after a brief moment
        import time
        time.sleep(0.5)
        if process.poll() is not None:
            # Process has already terminated
            stdout, stderr = process.communicate()
            print(f"Process terminated immediately. Return code: {process.returncode}")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
        else:
            print(f"Process is running successfully with PID {process.pid}")
        
    except Exception as e:
        print(f"Error starting car detector for lot {lot_id}: {str(e)}")
        import traceback
        traceback.print_exc()

@receiver(parking_lot_logged_out)
def stop_car_detector(sender, lot_id, **kwargs):
    """Stop the car detector script when parking lot logs out"""
    if lot_id in running_processes:
        process = running_processes[lot_id]
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                print(f"Car detector stopped for parking lot {lot_id}")
            del running_processes[lot_id]
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate gracefully
            process.kill()
            print(f"Car detector force killed for parking lot {lot_id}")
        except Exception as e:
            print(f"Error stopping car detector for lot {lot_id}: {str(e)}")

# @csrf_exempt
# @require_http_methods(["POST"])
# def parking_lot_login(request):
#     try:
#         data = json.loads(request.body)
#         username = data.get("username")
#         password = data.get("password")

#         if not username or not password:
#             return JsonResponse({"success": False, "message": "Username and password required"}, status=400)

#         try:
#             lot = ParkingLot.objects.get(username=username, password=password)
#         except ParkingLot.DoesNotExist:
#             return JsonResponse({"success": False, "message": "Invalid username or password"}, status=401)

#         print(f"About to trigger signal for lot {lot.id}")  # Debug print
        
#         # Trigger the car detector script
#         parking_lot_logged_in.send(sender=ParkingLot, lot_id=lot.id)
        
#         print(f"Signal sent for lot {lot.id}")  # Debug print

#         # Prepare the data to return
#         lot_data = {
#             "id": lot.id,
#             "name": lot.name,
#             "location": lot.location,
#             "total_spaces": lot.total_spaces,
#             "available_spaces": lot.available_spaces,
#             "registered_by": lot.registered_by.id,
#             "confirmed": lot.confirmed,
#             "username": lot.username,
#         }

#         return JsonResponse({
#             "success": True,
#             "message": "Login successful",
#             "data": lot_data
#         })

#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
#     except Exception as e:
#         print(f"Login error: {str(e)}")  # Debug print
# Add this new view to check process status


@csrf_exempt
@require_http_methods(["POST"])
def parking_lot_login(request):
    try:
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return JsonResponse({"success": False, "message": "Username and password required"}, status=400)
        
        try:
            lot = ParkingLot.objects.get(username=username, password=password)
        except ParkingLot.DoesNotExist:
            return JsonResponse({"success": False, "message": "Invalid username or password"}, status=401)
        
        print(f"About to trigger signal for lot {lot.id}")  # Debug print
        
        # Trigger the car detector script
        parking_lot_logged_in.send(sender=ParkingLot, lot_id=lot.id)
        
        print(f"Signal sent for lot {lot.id}")  # Debug print
        
        # Prepare the data to return
        lot_data = {
            "id": lot.id,
            "name": lot.name,
            "location": lot.location,
            "total_spaces": lot.total_spaces,
            "available_spaces": lot.available_spaces,
            "registered_by": lot.registered_by.id,
            "confirmed": lot.confirmed,
            "username": lot.username,
        }
        
        return JsonResponse({
            "success": True,
            "message": "Login successful",
            "data": lot_data,
            "lotId": lot.id  # Add lotId to response for client-side storage
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
    except Exception as e:
        print(f"Login error: {str(e)}")  # Debug print
        return JsonResponse({"success": False, "message": "Internal server error"}, status=500)
@csrf_exempt
@require_http_methods(["GET"])
def check_processes(request):
    """Debug endpoint to check running processes"""
    status = {}
    for lot_id, process in running_processes.items():
        if process.poll() is None:
            status[lot_id] = {"status": "running", "pid": process.pid}
        else:
            # Get output from terminated process
            stdout, stderr = process.communicate()
            status[lot_id] = {
                "status": "terminated", 
                "return_code": process.returncode,
                "stdout": stdout.decode()[:500],  # First 500 chars
                "stderr": stderr.decode()[:500]   # First 500 chars
            }
    
# Alternative: Run script in thread instead of subprocess
import threading
import importlib.util

# Dictionary to keep track of running threads
running_threads = {}

def run_car_detector_in_thread(lot_id, script_path):
    """Run the car detector script in a separate thread"""
    try:
        # Import and run the script as a module
        spec = importlib.util.spec_from_file_location("carDetector", script_path)
        car_detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(car_detector)
        
        # If the script has a main function, call it
        if hasattr(car_detector, 'main'):
            car_detector.main()
        
    except Exception as e:
        print(f"Error running car detector in thread for lot {lot_id}: {str(e)}")
        import traceback
        traceback.print_exc()

# Alternative signal receiver using threading
@receiver(parking_lot_logged_in)
def start_car_detector_thread(sender, lot_id, **kwargs):

    script_path = "../../ALPR/carDetector.py"
    """Start the car detector script in a thread when parking lot logs in"""
    
    
    # Check if thread is already running for this lot
    if lot_id in running_threads and running_threads[lot_id].is_alive():
        print(f"Car detector thread already running for lot {lot_id}")
        return
    
    try:
        # Start the script in a separate thread
        thread = threading.Thread(
            target=run_car_detector_in_thread, 
            args=(lot_id, script_path),
            daemon=True  # Dies when main program dies
        )
        thread.start()
        
        # Store the thread
        running_threads[lot_id] = thread
        print(f"Car detector thread started for parking lot {lot_id}")
        
    except Exception as e:
        print(f"Error starting car detector thread for lot {lot_id}: {str(e)}")
        import traceback
        traceback.print_exc()

@csrf_exempt
@require_http_methods(["POST"])
def parking_lot_logout(request):
    try:
        data = json.loads(request.body)
        lot_id = data.get("lot_id")
        
        if not lot_id:
            return JsonResponse({"success": False, "message": "Lot ID required"}, status=400)
        
        # Trigger signal to stop car detector
        parking_lot_logged_out.send(sender=ParkingLot, lot_id=lot_id)
        
        return JsonResponse({
            "success": True,
            "message": "Logout successful"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


@require_http_methods(["GET"])
@csrf_exempt
def get_license_plate(request):
    """
    GET API to read license plate detection from file and clear it
    """
    file_path = "../../ALPR/detection.txt"
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return JsonResponse({
                'success': False,
                'message': 'Detection file not found',
                'license_plate': None
            }, status=404)
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # Check if file has content
        if not content:
            return JsonResponse({
                'success': True,
                'message': 'No license plate detected',
                'license_plate': None
            }, status=200)
        
        # Clear the file after reading
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('')
        
        logger.info(f"License plate detected and returned: {content}")
        
        return JsonResponse({
            'success': True,
            'message': 'License plate detected successfully',
            'license_plate': content
        }, status=200)
        
    except FileNotFoundError:
        logger.error(f"Detection file not found: {file_path}")
        return JsonResponse({
            'success': False,
            'message': 'Detection file not found',
            'license_plate': None
        }, status=404)
        
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_path}")
        return JsonResponse({
            'success': False,
            'message': 'Permission denied accessing detection file',
            'license_plate': None
        }, status=403)
        
    except Exception as e:
        logger.error(f"Error reading detection file: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error reading detection file: {str(e)}',
            'license_plate': None
        }, status=500)


# def get_directions(entry_x, entry_y, target_x, target_y, scale_factor=0.1):
#     dx = target_x - entry_x
#     dy = target_y - entry_y
    
#     # Convert image units to meters using scale factor
#     dx_m = dx * scale_factor
#     dy_m = dy * scale_factor
    
#     directions = []
    
#     # Based on the parking lot image coordinate system:
#     # - Positive Y means going straight up into the lot from entry
#     # - Negative X means going RIGHT from entry perspective (image coordinates are flipped)
#     # - Positive X means going LEFT from entry perspective
    
#     # Handle vertical movement (straight into lot)
#     if abs(dy_m) > 0.1:  # Only add direction if movement is significant
#         if dy_m > 0:
#             directions.append(f"Go straight {abs(round(dy_m, 1))} meters")
#         else:
#             # If target has lower Y than entry, it's closer to entry
#             directions.append(f"Go straight {abs(round(dy_m, 1))} meters toward entrance")
    
#     # Handle horizontal movement (left/right from entry perspective)
#     # Note: X-axis is flipped in image coordinates
#     if abs(dx_m) > 0.1:  # Only add direction if movement is significant
#         if dx_m < 0:  # Negative X means RIGHT from entry perspective
#             directions.append(f"Turn right and go {abs(round(dx_m, 1))} meters")
#         else:  # Positive X means LEFT from entry perspective
#             directions.append(f"Turn left and go {abs(round(dx_m, 1))} meters")
    
#     # If no significant movement, the slot is very close to entry
#     if not directions:
#         directions.append("Slot is directly at the entry point")
    
#     return " → ".join(directions)


# Zone direction mappings
ZONE_DIRECTIONS = {
    'A': "Turn right",
    'B': "Go straight then turn right", 
    'C': "Turn left",
    'D': "Go straight then turn left"
}

def load_zone_data(location):
    """
    Load zone data from zone_summary_{location}.txt file
    Returns a dictionary mapping slot labels to zones
    """
    import os
    from django.conf import settings
    
    try:
        # HARDCODED FOR TESTING - using the actual filename with proper path
        filename = "zone_summary_221_fairway_road_s.txt"
        
        # Try multiple possible locations
        possible_paths = [
            filename,  # Current working directory
            os.path.join(os.path.dirname(__file__), filename),  # Same directory as views.py
            os.path.join(settings.BASE_DIR, filename),  # Project root
            os.path.join(settings.BASE_DIR, 'data', filename),  # data folder in project root
        ]
        
        actual_filename = None
        for path in possible_paths:
            print(f"Checking path: {path}")
            if os.path.exists(path):
                actual_filename = path
                print(f"Found file at: {actual_filename}")
                break
        
        if not actual_filename:
            print(f"File not found in any of these locations: {possible_paths}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Views.py directory: {os.path.dirname(__file__)}")
            print(f"BASE_DIR: {settings.BASE_DIR}")
            return {}
        
        print(f"Loading: {actual_filename}")
        slot_to_zone = {}
        
        with open(actual_filename, 'r') as file:
            content = file.read()
            
        # Parse each zone section
        zones = content.split('Zone ')
        for zone_section in zones[1:]:  # Skip first empty split
            lines = zone_section.strip().split('\n')
            zone_letter = lines[0].split(':')[0].strip()  # Get A, B, C, or D
            
            # Find the slots line
            for line in lines:
                if line.startswith('Slots:'):
                    slots_text = line.replace('Slots:', '').strip()
                    slots = [slot.strip() for slot in slots_text.split(',')]
                    
                    # Map each slot to its zone
                    for slot in slots:
                        slot_to_zone[slot] = zone_letter
                    break
        
        return slot_to_zone
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return {}
    except Exception as e:
        print(f"Error loading zone data: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_directions(zone):
    """
    Returns directions for the specified zone
    """
    return ZONE_DIRECTIONS.get(zone, "Zone not found")

@csrf_exempt
@require_POST
def assign_nearest_parking_slot(request):
    try:
        data = json.loads(request.body)
        slot_type = data.get('type')
        lot_id = data.get('lotId')
        
        if slot_type not in ['Regular', 'Reserved', 'Accessible']:
            return JsonResponse({'success': False, 'message': 'Invalid slot type'}, status=400)
        
        # Get lot location to determine zone file
        try:
            lot = ParkingLot.objects.get(id=lot_id)
            # HARDCODED FOR TESTING - ignore the actual location
            # location = lot.location.strip().lower().replace(" ", "_")
            location = "hardcoded_for_testing"  # This parameter is now ignored in load_zone_data
        except ParkingLot.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Parking lot not found'}, status=404)
        
        # Load zone data from file (now hardcoded)
        slot_to_zone = load_zone_data(location)
        if not slot_to_zone:
            return JsonResponse({'success': False, 'message': 'Zone data not available'}, status=404)
        
        # Build filter for slot type
        slot_filter = {
            'lotId_id': lot_id,
            'slot_assigned': False
        }
        
        if slot_type == 'Regular':
            slot_filter['is_regular'] = True
        elif slot_type == 'Reserved':
            slot_filter['is_reservation'] = True
        elif slot_type == 'Accessible':
            slot_filter['is_accessible'] = True
        
        # Get all matching available slots
        available_slots = ParkingLotCoordinate.objects.filter(**slot_filter)
        
        if not available_slots.exists():
            return JsonResponse({'success': False, 'message': 'No available slots of this type'}, status=404)
        
        # Randomly assign an available slot
        assigned_slot = random.choice(available_slots)
        
        # Mark slot as assigned
        assigned_slot.slot_assigned = True
        assigned_slot.save()
        
        # Get zone from loaded zone data
        zone = slot_to_zone.get(assigned_slot.label, 'Unknown')
        directions = get_directions(zone)
        
        return JsonResponse({
            'success': True,
            'message': f'{slot_type} slot assigned successfully',
            'slot': {
                'label': assigned_slot.label,
                'zone': zone,
                'directions': directions
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)}, status=500)
# @csrf_exempt
# @require_POST
# def assign_nearest_parking_slot(request):
#     try:
#         data = json.loads(request.body)
#         slot_type = data.get('type')
#         lot_id = data.get('lotId')
        
#         if slot_type not in ['Regular', 'Reserved', 'Accessible']:
#             return JsonResponse({'success': False, 'message': 'Invalid slot type'}, status=400)
        
#         # Get the entry point for the lot
#         entry_point = ParkingLotCoordinate.objects.filter(
#             lotId_id=lot_id,
#             is_Entry=True
#         ).first()
        
#         if not entry_point:
#             return JsonResponse({'success': False, 'message': 'Entry point not defined for this lot'}, status=404)
        
#         entry_x, entry_y = entry_point.entry_x, entry_point.entry_y
        
#         # Build filter for slot type
#         slot_filter = {
#             'lotId_id': lot_id,
#             'slot_assigned': False
#         }
        
#         if slot_type == 'Regular':
#             slot_filter['is_regular'] = True
#         elif slot_type == 'Reserved':
#             slot_filter['is_reservation'] = True
#         elif slot_type == 'Accessible':
#             slot_filter['is_accessible'] = True
        
#         # Get all matching available slots
#         available_slots = ParkingLotCoordinate.objects.filter(**slot_filter)
        
#         if not available_slots.exists():
#             return JsonResponse({'success': False, 'message': 'No available slots of this type'}, status=404)
        
#         # Find the nearest slot using Euclidean distance
#         def distance(slot):
#             return sqrt((slot.x_coordinate - entry_x)**2 + (slot.y_coordinate - entry_y)**2)
        
#         nearest_slot = min(available_slots, key=distance)
        
#         # Assign the slot
#         nearest_slot.slot_assigned = True
#         nearest_slot.save()
        
#         directions = get_directions(entry_x, entry_y, nearest_slot.x_coordinate, nearest_slot.y_coordinate)
        
#         return JsonResponse({
#             'success': True,
#             'message': f'{slot_type} slot assigned successfully (nearest)',
#             'slot': {
#                 'label': nearest_slot.label,
#                 'x_coordinate': nearest_slot.x_coordinate,
#                 'y_coordinate': nearest_slot.y_coordinate,
#                 'entry_x': nearest_slot.entry_x,
#                 'entry_y': nearest_slot.entry_y,
#                 'distance_from_entry': round(distance(nearest_slot), 2),
#                 'directions': directions
#             }
#         })
        
#     except json.JSONDecodeError:
#         return JsonResponse({'success': False, 'message': 'Invalid JSON'}, status=400)
#     except Exception as e:
#         return JsonResponse({'success': False, 'message': str(e)}, status=500)

