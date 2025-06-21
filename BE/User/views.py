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
    """Start the car detector script when parking lot logs in"""
    script_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/carDetector.py"
    
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
            "data": lot_data
        })

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
    except Exception as e:
        print(f"Login error: {str(e)}")  # Debug print
# Add this new view to check process status
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
    """Start the car detector script in a thread when parking lot logs in"""
    script_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/carDetector.py"
    
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

