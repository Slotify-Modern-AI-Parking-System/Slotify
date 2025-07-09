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

def thankyou(request):
    return render(request, "thankyou.html")

# import json
# import subprocess
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# from django.dispatch import Signal, receiver
# from slotifyBE.models import ParkingLot

# # Custom signals for parking lot login/logout
# parking_lot_logged_in = Signal()
# parking_lot_logged_out = Signal()

# # Dictionary to keep track of running processes/threads
# running_processes = {}
# running_threads = {}

# # Choose one approach - I recommend subprocess approach for better isolation
# USE_SUBPROCESS = True  # Set to False to use threading approach

# @receiver(parking_lot_logged_in)
# def start_car_detector(sender, lot_id, **kwargs):
#     """Start the car detector script when parking lot logs in"""
    
#     # Define script path - make sure this is correct
#     script_path = os.path.abspath("../../ALPR/carDetector.py")
    
#     print(f"Starting car detector for lot {lot_id}")
#     print(f"Script path: {script_path}")
    
#     if USE_SUBPROCESS:
#         start_car_detector_subprocess(lot_id, script_path)
#     else:
#         start_car_detector_thread(lot_id, script_path)

# def start_car_detector_subprocess(lot_id, script_path):
#     """Start car detector using subprocess"""
#     global running_processes
    
#     # Check if script is already running for this lot
#     if lot_id in running_processes:
#         process = running_processes[lot_id]
#         if process.poll() is None:  # Still running
#             print(f"Car detector already running for lot {lot_id}")
#             return
#         else:
#             # Process died, remove it
#             del running_processes[lot_id]
    
#     try:
#         # Verify script exists
#         if not os.path.exists(script_path):
#             print(f"ERROR: Script not found at {script_path}")
#             # Try alternative paths
#             alternative_paths = [
#                 os.path.join(os.path.dirname(__file__), "../../ALPR/carDetector.py"),
#                 os.path.join(os.getcwd(), "ALPR/carDetector.py"),
#                 "carDetector.py"
#             ]
            
#             for alt_path in alternative_paths:
#                 abs_alt_path = os.path.abspath(alt_path)
#                 print(f"Trying alternative path: {abs_alt_path}")
#                 if os.path.exists(abs_alt_path):
#                     script_path = abs_alt_path
#                     print(f"Found script at: {script_path}")
#                     break
#             else:
#                 print("ERROR: Could not find carDetector.py script")
#                 return
        
#         # Get Python executable
#         python_executable = sys.executable
#         print(f"Using Python executable: {python_executable}")
        
#         # Get script directory for setting working directory
#         script_dir = os.path.dirname(script_path)
        
#         # Start the script as a subprocess
#         process = subprocess.Popen([
#             python_executable, script_path
#         ], 
#         stdout=subprocess.PIPE, 
#         stderr=subprocess.PIPE, 
#         cwd=script_dir,
#         bufsize=1,
#         universal_newlines=True
#         )
        
#         # Store the process
#         running_processes[lot_id] = process
#         print(f"Car detector started for parking lot {lot_id} with PID {process.pid}")
        
#         # Check if process started successfully
#         time.sleep(1)  # Give it a moment to start
        
#         if process.poll() is not None:
#             # Process has already terminated
#             stdout, stderr = process.communicate()
#             print(f"Process terminated immediately. Return code: {process.returncode}")
#             print(f"STDOUT: {stdout}")
#             print(f"STDERR: {stderr}")
#             # Remove failed process
#             if lot_id in running_processes:
#                 del running_processes[lot_id]
#         else:
#             print(f"Process is running successfully with PID {process.pid}")
        
#     except Exception as e:
#         print(f"Error starting car detector subprocess for lot {lot_id}: {str(e)}")
#         traceback.print_exc()

# def start_car_detector_thread(lot_id, script_path):
#     """Start car detector using threading"""
#     global running_threads
    
#     # Check if thread is already running for this lot
#     if lot_id in running_threads and running_threads[lot_id].is_alive():
#         print(f"Car detector thread already running for lot {lot_id}")
#         return
    
#     try:
#         # Start the script in a separate thread
#         thread = threading.Thread(
#             target=run_car_detector_in_thread, 
#             args=(lot_id, script_path),
#             daemon=True,
#             name=f"CarDetector-{lot_id}"
#         )
#         thread.start()
        
#         # Store the thread
#         running_threads[lot_id] = thread
#         print(f"Car detector thread started for parking lot {lot_id}")
        
#     except Exception as e:
#         print(f"Error starting car detector thread for lot {lot_id}: {str(e)}")
#         traceback.print_exc()

# def run_car_detector_in_thread(lot_id, script_path):
#     """Run the car detector script in a separate thread"""
#     try:
#         print(f"Running car detector in thread for lot {lot_id}")
        
#         # Change to script directory
#         original_cwd = os.getcwd()
#         script_dir = os.path.dirname(script_path)
#         if script_dir:
#             os.chdir(script_dir)
        
#         # Import and run the script as a module
#         spec = importlib.util.spec_from_file_location("carDetector", script_path)
#         if spec is None:
#             raise ImportError(f"Could not load spec from {script_path}")
            
#         car_detector = importlib.util.module_from_spec(spec)
        
#         # Add script directory to sys.path temporarily
#         if script_dir not in sys.path:
#             sys.path.insert(0, script_dir)
        
#         try:
#             spec.loader.exec_module(car_detector)
            
#             # If the script has a main function, call it
#             if hasattr(car_detector, 'main'):
#                 print(f"Calling main() function for lot {lot_id}")
#                 car_detector.main()
#             else:
#                 print(f"No main() function found in carDetector.py for lot {lot_id}")
                
#         finally:
#             # Restore original working directory and sys.path
#             os.chdir(original_cwd)
#             if script_dir in sys.path:
#                 sys.path.remove(script_dir)
        
#     except Exception as e:
#         print(f"Error running car detector in thread for lot {lot_id}: {str(e)}")
#         traceback.print_exc()

# @receiver(parking_lot_logged_out)
# def stop_car_detector(sender, lot_id, **kwargs):
#     """Stop the car detector script when parking lot logs out"""
#     print(f"Stopping car detector for lot {lot_id}")
    
#     if USE_SUBPROCESS:
#         stop_car_detector_subprocess(lot_id)
#     else:
#         stop_car_detector_thread(lot_id)

# def stop_car_detector_subprocess(lot_id):
#     """Stop subprocess car detector"""
#     global running_processes
    
#     if lot_id in running_processes:
#         process = running_processes[lot_id]
#         try:
#             if process.poll() is None:  # Process is still running
#                 print(f"Terminating car detector process for lot {lot_id}")
#                 process.terminate()
                
#                 # Wait for graceful shutdown
#                 try:
#                     process.wait(timeout=5)
#                     print(f"Car detector stopped gracefully for parking lot {lot_id}")
#                 except subprocess.TimeoutExpired:
#                     # Force kill if it doesn't terminate gracefully
#                     print(f"Force killing car detector for lot {lot_id}")
#                     process.kill()
#                     process.wait()
#                     print(f"Car detector force killed for parking lot {lot_id}")
            
#             del running_processes[lot_id]
            
#         except Exception as e:
#             print(f"Error stopping car detector for lot {lot_id}: {str(e)}")
#             traceback.print_exc()

# def stop_car_detector_thread(lot_id):
#     """Stop thread car detector (note: Python threads can't be forcefully stopped)"""
#     global running_threads
    
#     if lot_id in running_threads:
#         thread = running_threads[lot_id]
#         if thread.is_alive():
#             print(f"Car detector thread for lot {lot_id} is still running")
#             print("Note: Python threads cannot be forcefully stopped")
#             # You might need to implement a stop flag in your carDetector.py
        
#         del running_threads[lot_id]

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
        
#         print(f"Parking lot {lot.id} logged in successfully")
#         print(f"About to trigger signal for lot {lot.id}")
        
#         # Trigger the car detector script
#         parking_lot_logged_in.send(sender=ParkingLot, lot_id=lot.id)
        
#         print(f"Signal sent for lot {lot.id}")
        
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
#             "data": lot_data,
#             "lotId": lot.id
#         })
        
#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
#     except Exception as e:
#         print(f"Login error: {str(e)}")
#         traceback.print_exc()
#         return JsonResponse({"success": False, "message": "Internal server error"}, status=500)

# @csrf_exempt
# @require_http_methods(["POST"])
# def parking_lot_logout(request):
#     try:
#         data = json.loads(request.body)
#         lot_id = data.get("lot_id")
        
#         if not lot_id:
#             return JsonResponse({"success": False, "message": "Lot ID required"}, status=400)
        
#         print(f"Parking lot {lot_id} logging out")
        
#         # Trigger signal to stop car detector
#         parking_lot_logged_out.send(sender=ParkingLot, lot_id=lot_id)
        
#         return JsonResponse({
#             "success": True,
#             "message": "Logout successful"
#         })
        
#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
#     except Exception as e:
#         print(f"Logout error: {str(e)}")
#         traceback.print_exc()
#         return JsonResponse({"success": False, "message": str(e)}, status=500)

# @csrf_exempt
# @require_http_methods(["GET"])
# def check_processes(request):
#     """Debug endpoint to check running processes"""
#     status = {
#         "approach": "subprocess" if USE_SUBPROCESS else "threading",
#         "processes": {},
#         "threads": {}
#     }
    
#     # Check subprocess status
#     for lot_id, process in running_processes.items():
#         if process.poll() is None:
#             status["processes"][lot_id] = {"status": "running", "pid": process.pid}
#         else:
#             try:
#                 stdout, stderr = process.communicate(timeout=1)
#                 status["processes"][lot_id] = {
#                     "status": "terminated", 
#                     "return_code": process.returncode,
#                     "stdout": stdout[:500] if stdout else "",
#                     "stderr": stderr[:500] if stderr else ""
#                 }
#             except subprocess.TimeoutExpired:
#                 status["processes"][lot_id] = {
#                     "status": "terminated", 
#                     "return_code": process.returncode,
#                     "stdout": "timeout",
#                     "stderr": "timeout"
#                 }
    
#     # Check thread status
#     for lot_id, thread in running_threads.items():
#         status["threads"][lot_id] = {
#             "status": "alive" if thread.is_alive() else "dead",
#             "name": thread.name
#         }
    
#     return JsonResponse(status)

# @require_http_methods(["GET"])
# @csrf_exempt
# def get_license_plate(request):
#     """
#     GET API to read license plate detection from file and clear it
#     """
#     file_path = "../../ALPR/detection.txt"
    
#     try:
#         # Check if file exists
#         if not os.path.exists(file_path):
#             return JsonResponse({
#                 'success': False,
#                 'message': 'Detection file not found',
#                 'license_plate': None
#             }, status=404)
        
#         # Read the file content
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read().strip()
        
#         # Check if file has content
#         if not content:
#             return JsonResponse({
#                 'success': True,
#                 'message': 'No license plate detected',
#                 'license_plate': None
#             }, status=200)
        
#         # Clear the file after reading
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write('')
        
#         logger.info(f"License plate detected and returned: {content}")
        
#         return JsonResponse({
#             'success': True,
#             'message': 'License plate detected successfully',
#             'license_plate': content
#         }, status=200)
        
#     except FileNotFoundError:
#         logger.error(f"Detection file not found: {file_path}")
#         return JsonResponse({
#             'success': False,
#             'message': 'Detection file not found',
#             'license_plate': None
#         }, status=404)
        
#     except PermissionError:
#         logger.error(f"Permission denied accessing file: {file_path}")
#         return JsonResponse({
#             'success': False,
#             'message': 'Permission denied accessing detection file',
#             'license_plate': None
#         }, status=403)
        
#     except Exception as e:
#         logger.error(f"Error reading detection file: {str(e)}")
#         return JsonResponse({
#             'success': False,
#             'message': f'Error reading detection file: {str(e)}',
#             'license_plate': None
#         }, status=500)


import json
import subprocess
import os
import sys
import time
import threading
import importlib.util
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.dispatch import Signal, receiver
from slotifyBE.models import ParkingLot, ParkingLotCoordinate
import logging



# View function to handle the cleanup (can be called via URL for cron jobs)

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import stripe
import json
from .models import ParkingReservation, Payment


# def reservation_form(request):
#     return render(request, 'reservation.html')
# @csrf_exempt

# def create_reservation(request):
#     if request.method == 'POST':
#         name = request.POST.get('name')
#         email = request.POST.get('email')
#         license_plate = request.POST.get('license_plate')
#         hours = int(request.POST.get('hours'))
#         total_amount = hours * 10
        
#         reservation = ParkingReservation.objects.create(
#             name=name,
#             email=email,
#             license_plate=license_plate,
#             hours=hours,
#             total_amount=total_amount
#         )
        
#         try:
#             intent = stripe.PaymentIntent.create(
#                 amount=int(total_amount * 100),
#                 currency='usd',
#                 metadata={'reservation_id': reservation.id}
#             )
            
#             Payment.objects.create(
#                 reservation=reservation,
#                 stripe_payment_intent_id=intent.id,
#                 amount=total_amount
#             )
            
#             return JsonResponse({
#                 'client_secret': intent.client_secret,
#                 'reservation_id': reservation.id
#             })
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=400)
    
#     return JsonResponse({'error': 'Invalid request'}, status=400)

# @csrf_exempt
# def payment_success(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         payment_intent_id = data.get('payment_intent_id')
        
#         try:
#             payment = Payment.objects.get(stripe_payment_intent_id=payment_intent_id)
#             payment.status = 'completed'
#             payment.save()
#             return JsonResponse({'status': 'success'})
#         except Payment.DoesNotExist:
#             return JsonResponse({'error': 'Payment not found'}, status=404)
    
#     return JsonResponse({'error': 'Invalid request'}, status=400)

def reservation_form(request):
    return render(request, 'reservation.html')
@csrf_exempt
def create_reservation(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        license_plate = request.POST.get('license_plate')
        location = request.POST.get('location')
        hours = int(request.POST.get('hours'))
        total_amount = hours * 10
        
        # Check if parking lot exists at the location
        try:
            parking_lot = ParkingLot.objects.get(location=location)
        except ParkingLot.DoesNotExist:
            return JsonResponse({'error': 'No parking lot found at this location'}, status=400)
        
        # Find available slot (but don't assign yet)
        try:
            available_slot = ParkingLotCoordinate.objects.filter(
                lotId=parking_lot.id,
                slot_assigned=False
            ).first()
            
            if not available_slot:
                return JsonResponse({'error': 'No available slots at this location'}, status=400)
                
        except Exception as e:
            return JsonResponse({'error': 'Error finding available slot'}, status=400)
        
        # Create reservation without assigning slot yet
        reservation = ParkingReservation.objects.create(
            name=name,
            email=email,
            license_plate=license_plate,
            location=location,
            parking_lot=parking_lot,
            assigned_slot=available_slot,
            hours=hours,
            total_amount=total_amount
        )
        
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(total_amount * 100),
                currency='usd',
                metadata={
                    'reservation_id': reservation.id,
                    'slot_id': available_slot.id
                }
            )
            
            Payment.objects.create(
                reservation=reservation,
                stripe_payment_intent_id=intent.id,
                amount=total_amount
            )
            
            return JsonResponse({
                'client_secret': intent.client_secret,
                'reservation_id': reservation.id,
                'slot_info': f'Slot {available_slot.label or available_slot.id} in Zone {available_slot.zone or "N/A"} will be assigned after payment'
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def payment_success(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        payment_intent_id = data.get('payment_intent_id')
        
        try:
            payment = Payment.objects.get(stripe_payment_intent_id=payment_intent_id)
            payment.status = 'completed'
            payment.save()
            
            # Get slot ID from Stripe metadata
            stripe_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            slot_id = stripe_intent.metadata.get('slot_id')
            
            if slot_id:
                # Now assign the slot after successful payment
                slot = ParkingLotCoordinate.objects.get(id=slot_id)
                slot.slot_assigned = True
                slot.save()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Payment successful! Slot {slot.label or slot.id} assigned.'
            })
        except Payment.DoesNotExist:
            return JsonResponse({'error': 'Payment not found'}, status=404)
        except ParkingLotCoordinate.DoesNotExist:
            return JsonResponse({'error': 'Slot not found'}, status=404)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)




# Set up logger
import logging
import os
import sys
import subprocess
import threading
import time
import json
import traceback
import importlib.util
from django.dispatch import Signal, receiver
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

# Custom signals for parking lot login/logout
parking_lot_logged_in = Signal()
parking_lot_logged_out = Signal()

# Dictionary to keep track of running processes/threads
running_processes = {}
running_threads = {}

# Choose one approach - I recommend subprocess approach for better isolation
USE_SUBPROCESS = True  # Set to False to use threading approach

@receiver(parking_lot_logged_in)
def start_car_detectors(sender, lot_id, **kwargs):
    """Start both car detector scripts when parking lot logs in"""
    
    # Define script paths with absolute paths
    entry_script_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/carDetector.py"
    exit_script_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/carExit.py"
    
    print(f"Starting car detectors for lot {lot_id}")
    print(f"Entry script path: {entry_script_path}")
    print(f"Exit script path: {exit_script_path}")
    
    # Check if both scripts exist before starting
    if not os.path.exists(entry_script_path):
        print(f"ERROR: Entry script not found at {entry_script_path}")
        return
    
    if not os.path.exists(exit_script_path):
        print(f"ERROR: Exit script not found at {exit_script_path}")
        return
    
    if USE_SUBPROCESS:
        # Start entry detector
        success_entry = start_car_detector_subprocess(lot_id, entry_script_path, "entry")
        
        # Wait a moment between starting processes
        time.sleep(2)
        
        # Start exit detector
        success_exit = start_car_detector_subprocess(lot_id, exit_script_path, "exit")
        
        print(f"Entry detector started: {success_entry}")
        print(f"Exit detector started: {success_exit}")
    else:
        start_car_detector_thread(lot_id, entry_script_path, "entry")
        time.sleep(1)  # Small delay between thread starts
        start_car_detector_thread(lot_id, exit_script_path, "exit")

def start_car_detector_subprocess(lot_id, script_path, script_type):
    """Start car detector using subprocess"""
    global running_processes
    
    # Create unique key for each script type
    process_key = f"{lot_id}_{script_type}"
    
    print(f"Attempting to start {script_type} detector for lot {lot_id}")
    print(f"Process key: {process_key}")
    
    # Check if script is already running for this lot and type
    if process_key in running_processes:
        process = running_processes[process_key]
        if process.poll() is None:  # Still running
            print(f"Car detector ({script_type}) already running for lot {lot_id} with PID {process.pid}")
            return True
        else:
            # Process died, remove it
            print(f"Removing dead process for {script_type}")
            del running_processes[process_key]
    
    try:
        # Verify script exists
        if not os.path.exists(script_path):
            print(f"ERROR: Script not found at {script_path}")
            return False
        
        # Get Python executable
        python_executable = sys.executable
        print(f"Using Python executable: {python_executable}")
        
        # Get script directory for setting working directory
        script_dir = os.path.dirname(script_path)
        print(f"Script directory: {script_dir}")
        
        # Start the script as a subprocess with more detailed logging
        print(f"Starting subprocess: {python_executable} {script_path}")
        
        process = subprocess.Popen([
            python_executable, script_path
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        cwd=script_dir,
        bufsize=1,
        universal_newlines=True
        )
        
        # Store the process
        running_processes[process_key] = process
        print(f"Car detector ({script_type}) started for parking lot {lot_id} with PID {process.pid}")
        
        # Check if process started successfully with longer wait
        time.sleep(3)  # Give it more time to start
        
        if process.poll() is not None:
            # Process has already terminated
            stdout, stderr = process.communicate()
            print(f"Process ({script_type}) terminated immediately. Return code: {process.returncode}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            # Remove failed process
            if process_key in running_processes:
                del running_processes[process_key]
            return False
        else:
            print(f"Process ({script_type}) is running successfully with PID {process.pid}")
            return True
        
    except Exception as e:
        print(f"Error starting car detector ({script_type}) subprocess for lot {lot_id}: {str(e)}")
        traceback.print_exc()
        return False

def start_car_detector_thread(lot_id, script_path, script_type):
    """Start car detector using threading"""
    global running_threads
    
    # Create unique key for each script type
    thread_key = f"{lot_id}_{script_type}"
    
    print(f"Attempting to start {script_type} detector thread for lot {lot_id}")
    
    # Check if thread is already running for this lot and type
    if thread_key in running_threads and running_threads[thread_key].is_alive():
        print(f"Car detector ({script_type}) thread already running for lot {lot_id}")
        return
    
    try:
        # Start the script in a separate thread
        thread = threading.Thread(
            target=run_car_detector_in_thread, 
            args=(lot_id, script_path, script_type),
            daemon=True,
            name=f"CarDetector-{script_type}-{lot_id}"
        )
        thread.start()
        
        # Store the thread
        running_threads[thread_key] = thread
        print(f"Car detector ({script_type}) thread started for parking lot {lot_id}")
        
    except Exception as e:
        print(f"Error starting car detector ({script_type}) thread for lot {lot_id}: {str(e)}")
        traceback.print_exc()

def run_car_detector_in_thread(lot_id, script_path, script_type):
    """Run the car detector script in a separate thread"""
    try:
        print(f"Running car detector ({script_type}) in thread for lot {lot_id}")
        
        # Change to script directory
        original_cwd = os.getcwd()
        script_dir = os.path.dirname(script_path)
        if script_dir:
            os.chdir(script_dir)
        
        # Import and run the script as a module
        module_name = f"carDetector_{script_type}_{lot_id}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {script_path}")
            
        car_detector = importlib.util.module_from_spec(spec)
        
        # Add script directory to sys.path temporarily
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        try:
            spec.loader.exec_module(car_detector)
            
            # If the script has a main function, call it
            if hasattr(car_detector, 'main'):
                print(f"Calling main() function for lot {lot_id} ({script_type})")
                car_detector.main()
            else:
                print(f"No main() function found in {os.path.basename(script_path)} for lot {lot_id}")
                
        finally:
            # Restore original working directory and sys.path
            os.chdir(original_cwd)
            if script_dir in sys.path:
                sys.path.remove(script_dir)
        
    except Exception as e:
        print(f"Error running car detector ({script_type}) in thread for lot {lot_id}: {str(e)}")
        traceback.print_exc()

@receiver(parking_lot_logged_out)
def stop_car_detectors(sender, lot_id, **kwargs):
    """Stop both car detector scripts when parking lot logs out"""
    print(f"Stopping car detectors for lot {lot_id}")
    
    if USE_SUBPROCESS:
        stop_car_detector_subprocess(lot_id, "entry")
        stop_car_detector_subprocess(lot_id, "exit")
    else:
        stop_car_detector_thread(lot_id, "entry")
        stop_car_detector_thread(lot_id, "exit")

def stop_car_detector_subprocess(lot_id, script_type):
    """Stop subprocess car detector"""
    global running_processes
    
    process_key = f"{lot_id}_{script_type}"
    
    if process_key in running_processes:
        process = running_processes[process_key]
        try:
            if process.poll() is None:  # Process is still running
                print(f"Terminating car detector ({script_type}) process for lot {lot_id}")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    print(f"Car detector ({script_type}) stopped gracefully for parking lot {lot_id}")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    print(f"Force killing car detector ({script_type}) for lot {lot_id}")
                    process.kill()
                    process.wait()
                    print(f"Car detector ({script_type}) force killed for parking lot {lot_id}")
            
            del running_processes[process_key]
            
        except Exception as e:
            print(f"Error stopping car detector ({script_type}) for lot {lot_id}: {str(e)}")
            traceback.print_exc()

def stop_car_detector_thread(lot_id, script_type):
    """Stop thread car detector (note: Python threads can't be forcefully stopped)"""
    global running_threads
    
    thread_key = f"{lot_id}_{script_type}"
    
    if thread_key in running_threads:
        thread = running_threads[thread_key]
        if thread.is_alive():
            print(f"Car detector ({script_type}) thread for lot {lot_id} is still running")
            print("Note: Python threads cannot be forcefully stopped")
            # You might need to implement a stop flag in your scripts
        
        del running_threads[thread_key]

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
        
        print(f"Parking lot {lot.id} logged in successfully")
        print(f"About to trigger signal for lot {lot.id}")
        
        # Trigger the car detector scripts
        parking_lot_logged_in.send(sender=ParkingLot, lot_id=lot.id)
        
        print(f"Signal sent for lot {lot.id}")
        
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
            "lotId": lot.id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
    except Exception as e:
        print(f"Login error: {str(e)}")
        traceback.print_exc()
        return JsonResponse({"success": False, "message": "Internal server error"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def parking_lot_logout(request):
    try:
        data = json.loads(request.body)
        lot_id = data.get("lot_id")
        
        if not lot_id:
            return JsonResponse({"success": False, "message": "Lot ID required"}, status=400)
        
        print(f"Parking lot {lot_id} logging out")
        
        # Trigger signal to stop car detectors
        parking_lot_logged_out.send(sender=ParkingLot, lot_id=lot_id)
        
        return JsonResponse({
            "success": True,
            "message": "Logout successful"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
    except Exception as e:
        print(f"Logout error: {str(e)}")
        traceback.print_exc()
        return JsonResponse({"success": False, "message": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def check_processes(request):
    """Debug endpoint to check running processes"""
    status = {
        "approach": "subprocess" if USE_SUBPROCESS else "threading",
        "processes": {},
        "threads": {},
        "total_processes": len(running_processes),
        "total_threads": len(running_threads)
    }
    
    # Check subprocess status
    for process_key, process in running_processes.items():
        try:
            if process.poll() is None:
                status["processes"][process_key] = {"status": "running", "pid": process.pid}
            else:
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    status["processes"][process_key] = {
                        "status": "terminated", 
                        "return_code": process.returncode,
                        "stdout": stdout[:500] if stdout else "",
                        "stderr": stderr[:500] if stderr else ""
                    }
                except subprocess.TimeoutExpired:
                    status["processes"][process_key] = {
                        "status": "terminated", 
                        "return_code": process.returncode,
                        "stdout": "timeout",
                        "stderr": "timeout"
                    }
        except Exception as e:
            status["processes"][process_key] = {"status": "error", "error": str(e)}
    
    # Check thread status
    for thread_key, thread in running_threads.items():
        status["threads"][thread_key] = {
            "status": "alive" if thread.is_alive() else "dead",
            "name": thread.name
        }
    
    return JsonResponse(status)

@require_http_methods(["GET"])
@csrf_exempt
def get_license_plate(request):
    """
    GET API to read license plate detection from entry file and clear it
    """
    file_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/detection.txt"
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return JsonResponse({
                'success': False,
                'message': 'Entry detection file not found',
                'license_plate': None
            }, status=404)
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # Check if file has content
        if not content:
            return JsonResponse({
                'success': True,
                'message': 'No license plate detected at entry',
                'license_plate': None
            }, status=200)
        
        # Clear the file after reading
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('')
        
        logger.info(f"Entry license plate detected and returned: {content}")
        
        return JsonResponse({
            'success': True,
            'message': 'Entry license plate detected successfully',
            'license_plate': content
        }, status=200)
        
    except FileNotFoundError:
        logger.error(f"Entry detection file not found: {file_path}")
        return JsonResponse({
            'success': False,
            'message': 'Entry detection file not found',
            'license_plate': None
        }, status=404)
        
    except PermissionError:
        logger.error(f"Permission denied accessing entry file: {file_path}")
        return JsonResponse({
            'success': False,
            'message': 'Permission denied accessing entry detection file',
            'license_plate': None
        }, status=403)
        
    except Exception as e:
        logger.error(f"Error reading entry detection file: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error reading entry detection file: {str(e)}',
            'license_plate': None
        }, status=500)

@require_http_methods(["GET"])
@csrf_exempt
def get_exit_license_plate(request):
    """
    GET API to read license plate detection from exit file and clear it
    """
    file_path = "/Users/jainamdoshi/Desktop/Projects/Slotify/ALPR/exitDetection.txt"
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return JsonResponse({
                'success': False,
                'message': 'Exit detection file not found',
                'license_plate': None
            }, status=404)
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # Check if file has content
        if not content:
            return JsonResponse({
                'success': True,
                'message': 'No license plate detected at exit',
                'license_plate': None
            }, status=200)
        
        # Clear the file after reading
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('')
        
        logger.info(f"Exit license plate detected and returned: {content}")
        
        return JsonResponse({
            'success': True,
            'message': 'Exit license plate detected successfully',
            'license_plate': content
        }, status=200)
        
    except FileNotFoundError:
        logger.error(f"Exit detection file not found: {file_path}")
        return JsonResponse({
            'success': False,
            'message': 'Exit detection file not found',
            'license_plate': None
        }, status=404)
        
    except PermissionError:
        logger.error(f"Permission denied accessing exit file: {file_path}")
        return JsonResponse({
            'success': False,
            'message': 'Permission denied accessing exit detection file',
            'license_plate': None
        }, status=403)
        
    except Exception as e:
        logger.error(f"Error reading exit detection file: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error reading exit detection file: {str(e)}',
            'license_plate': None
        }, status=500)






# # Set up logger
# logger = logging.getLogger(__name__)

# # Custom signals for parking lot login/logout
# parking_lot_logged_in = Signal()
# parking_lot_logged_out = Signal()

# # Dictionary to keep track of running processes/threads
# running_processes = {}
# running_threads = {}

# # Choose one approach - I recommend subprocess approach for better isolation
# USE_SUBPROCESS = True  # Set to False to use threading approach

# @receiver(parking_lot_logged_in)
# def start_car_detectors(sender, lot_id, **kwargs):
#     """Start both car detector scripts when parking lot logs in"""
    
#     # Define script paths - make sure these are correct
#     entry_script_path = os.path.abspath("../../ALPR/carDetector.py")
#     exit_script_path = os.path.abspath("../../ALPR/carExit.py")
    
#     print(f"Starting car detectors for lot {lot_id}")
#     print(f"Entry script path: {entry_script_path}")
#     print(f"Exit script path: {exit_script_path}")
    
#     if USE_SUBPROCESS:
#         start_car_detector_subprocess(lot_id, entry_script_path, "entry")
#         start_car_detector_subprocess(lot_id, exit_script_path, "exit")
#     else:
#         start_car_detector_thread(lot_id, entry_script_path, "entry")
#         start_car_detector_thread(lot_id, exit_script_path, "exit")

# def start_car_detector_subprocess(lot_id, script_path, script_type):
#     """Start car detector using subprocess"""
#     global running_processes
    
#     # Create unique key for each script type
#     process_key = f"{lot_id}_{script_type}"
    
#     # Check if script is already running for this lot and type
#     if process_key in running_processes:
#         process = running_processes[process_key]
#         if process.poll() is None:  # Still running
#             print(f"Car detector ({script_type}) already running for lot {lot_id}")
#             return
#         else:
#             # Process died, remove it
#             del running_processes[process_key]
    
#     try:
#         # Verify script exists
#         if not os.path.exists(script_path):
#             print(f"ERROR: Script not found at {script_path}")
#             # Try alternative paths
#             script_name = os.path.basename(script_path)
#             alternative_paths = [
#                 os.path.join(os.path.dirname(__file__), f"../../ALPR/{script_name}"),
#                 os.path.join(os.getcwd(), f"ALPR/{script_name}"),
#                 script_name
#             ]
            
#             for alt_path in alternative_paths:
#                 abs_alt_path = os.path.abspath(alt_path)
#                 print(f"Trying alternative path: {abs_alt_path}")
#                 if os.path.exists(abs_alt_path):
#                     script_path = abs_alt_path
#                     print(f"Found script at: {script_path}")
#                     break
#             else:
#                 print(f"ERROR: Could not find {script_name} script")
#                 return
        
#         # Get Python executable
#         python_executable = sys.executable
#         print(f"Using Python executable: {python_executable}")
        
#         # Get script directory for setting working directory
#         script_dir = os.path.dirname(script_path)
        
#         # Start the script as a subprocess
#         process = subprocess.Popen([
#             python_executable, script_path
#         ], 
#         stdout=subprocess.PIPE, 
#         stderr=subprocess.PIPE, 
#         cwd=script_dir,
#         bufsize=1,
#         universal_newlines=True
#         )
        
#         # Store the process
#         running_processes[process_key] = process
#         print(f"Car detector ({script_type}) started for parking lot {lot_id} with PID {process.pid}")
        
#         # Check if process started successfully
#         time.sleep(1)  # Give it a moment to start
        
#         if process.poll() is not None:
#             # Process has already terminated
#             stdout, stderr = process.communicate()
#             print(f"Process ({script_type}) terminated immediately. Return code: {process.returncode}")
#             print(f"STDOUT: {stdout}")
#             print(f"STDERR: {stderr}")
#             # Remove failed process
#             if process_key in running_processes:
#                 del running_processes[process_key]
#         else:
#             print(f"Process ({script_type}) is running successfully with PID {process.pid}")
        
#     except Exception as e:
#         print(f"Error starting car detector ({script_type}) subprocess for lot {lot_id}: {str(e)}")
#         traceback.print_exc()

# def start_car_detector_thread(lot_id, script_path, script_type):
#     """Start car detector using threading"""
#     global running_threads
    
#     # Create unique key for each script type
#     thread_key = f"{lot_id}_{script_type}"
    
#     # Check if thread is already running for this lot and type
#     if thread_key in running_threads and running_threads[thread_key].is_alive():
#         print(f"Car detector ({script_type}) thread already running for lot {lot_id}")
#         return
    
#     try:
#         # Start the script in a separate thread
#         thread = threading.Thread(
#             target=run_car_detector_in_thread, 
#             args=(lot_id, script_path, script_type),
#             daemon=True,
#             name=f"CarDetector-{script_type}-{lot_id}"
#         )
#         thread.start()
        
#         # Store the thread
#         running_threads[thread_key] = thread
#         print(f"Car detector ({script_type}) thread started for parking lot {lot_id}")
        
#     except Exception as e:
#         print(f"Error starting car detector ({script_type}) thread for lot {lot_id}: {str(e)}")
#         traceback.print_exc()

# def run_car_detector_in_thread(lot_id, script_path, script_type):
#     """Run the car detector script in a separate thread"""
#     try:
#         print(f"Running car detector ({script_type}) in thread for lot {lot_id}")
        
#         # Change to script directory
#         original_cwd = os.getcwd()
#         script_dir = os.path.dirname(script_path)
#         if script_dir:
#             os.chdir(script_dir)
        
#         # Import and run the script as a module
#         module_name = f"carDetector_{script_type}_{lot_id}"
#         spec = importlib.util.spec_from_file_location(module_name, script_path)
#         if spec is None:
#             raise ImportError(f"Could not load spec from {script_path}")
            
#         car_detector = importlib.util.module_from_spec(spec)
        
#         # Add script directory to sys.path temporarily
#         if script_dir not in sys.path:
#             sys.path.insert(0, script_dir)
        
#         try:
#             spec.loader.exec_module(car_detector)
            
#             # If the script has a main function, call it
#             if hasattr(car_detector, 'main'):
#                 print(f"Calling main() function for lot {lot_id} ({script_type})")
#                 car_detector.main()
#             else:
#                 print(f"No main() function found in {os.path.basename(script_path)} for lot {lot_id}")
                
#         finally:
#             # Restore original working directory and sys.path
#             os.chdir(original_cwd)
#             if script_dir in sys.path:
#                 sys.path.remove(script_dir)
        
#     except Exception as e:
#         print(f"Error running car detector ({script_type}) in thread for lot {lot_id}: {str(e)}")
#         traceback.print_exc()

# @receiver(parking_lot_logged_out)
# def stop_car_detectors(sender, lot_id, **kwargs):
#     """Stop both car detector scripts when parking lot logs out"""
#     print(f"Stopping car detectors for lot {lot_id}")
    
#     if USE_SUBPROCESS:
#         stop_car_detector_subprocess(lot_id, "entry")
#         stop_car_detector_subprocess(lot_id, "exit")
#     else:
#         stop_car_detector_thread(lot_id, "entry")
#         stop_car_detector_thread(lot_id, "exit")

# def stop_car_detector_subprocess(lot_id, script_type):
#     """Stop subprocess car detector"""
#     global running_processes
    
#     process_key = f"{lot_id}_{script_type}"
    
#     if process_key in running_processes:
#         process = running_processes[process_key]
#         try:
#             if process.poll() is None:  # Process is still running
#                 print(f"Terminating car detector ({script_type}) process for lot {lot_id}")
#                 process.terminate()
                
#                 # Wait for graceful shutdown
#                 try:
#                     process.wait(timeout=5)
#                     print(f"Car detector ({script_type}) stopped gracefully for parking lot {lot_id}")
#                 except subprocess.TimeoutExpired:
#                     # Force kill if it doesn't terminate gracefully
#                     print(f"Force killing car detector ({script_type}) for lot {lot_id}")
#                     process.kill()
#                     process.wait()
#                     print(f"Car detector ({script_type}) force killed for parking lot {lot_id}")
            
#             del running_processes[process_key]
            
#         except Exception as e:
#             print(f"Error stopping car detector ({script_type}) for lot {lot_id}: {str(e)}")
#             traceback.print_exc()

# def stop_car_detector_thread(lot_id, script_type):
#     """Stop thread car detector (note: Python threads can't be forcefully stopped)"""
#     global running_threads
    
#     thread_key = f"{lot_id}_{script_type}"
    
#     if thread_key in running_threads:
#         thread = running_threads[thread_key]
#         if thread.is_alive():
#             print(f"Car detector ({script_type}) thread for lot {lot_id} is still running")
#             print("Note: Python threads cannot be forcefully stopped")
#             # You might need to implement a stop flag in your scripts
        
#         del running_threads[thread_key]

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
        
#         print(f"Parking lot {lot.id} logged in successfully")
#         print(f"About to trigger signal for lot {lot.id}")
        
#         # Trigger the car detector scripts
#         parking_lot_logged_in.send(sender=ParkingLot, lot_id=lot.id)
        
#         print(f"Signal sent for lot {lot.id}")
        
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
#             "data": lot_data,
#             "lotId": lot.id
#         })
        
#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
#     except Exception as e:
#         print(f"Login error: {str(e)}")
#         traceback.print_exc()
#         return JsonResponse({"success": False, "message": "Internal server error"}, status=500)

# @csrf_exempt
# @require_http_methods(["POST"])
# def parking_lot_logout(request):
#     try:
#         data = json.loads(request.body)
#         lot_id = data.get("lot_id")
        
#         if not lot_id:
#             return JsonResponse({"success": False, "message": "Lot ID required"}, status=400)
        
#         print(f"Parking lot {lot_id} logging out")
        
#         # Trigger signal to stop car detectors
#         parking_lot_logged_out.send(sender=ParkingLot, lot_id=lot_id)
        
#         return JsonResponse({
#             "success": True,
#             "message": "Logout successful"
#         })
        
#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
#     except Exception as e:
#         print(f"Logout error: {str(e)}")
#         traceback.print_exc()
#         return JsonResponse({"success": False, "message": str(e)}, status=500)

# @csrf_exempt
# @require_http_methods(["GET"])
# def check_processes(request):
#     """Debug endpoint to check running processes"""
#     status = {
#         "approach": "subprocess" if USE_SUBPROCESS else "threading",
#         "processes": {},
#         "threads": {}
#     }
    
#     # Check subprocess status
#     for process_key, process in running_processes.items():
#         if process.poll() is None:
#             status["processes"][process_key] = {"status": "running", "pid": process.pid}
#         else:
#             try:
#                 stdout, stderr = process.communicate(timeout=1)
#                 status["processes"][process_key] = {
#                     "status": "terminated", 
#                     "return_code": process.returncode,
#                     "stdout": stdout[:500] if stdout else "",
#                     "stderr": stderr[:500] if stderr else ""
#                 }
#             except subprocess.TimeoutExpired:
#                 status["processes"][process_key] = {
#                     "status": "terminated", 
#                     "return_code": process.returncode,
#                     "stdout": "timeout",
#                     "stderr": "timeout"
#                 }
    
#     # Check thread status
#     for thread_key, thread in running_threads.items():
#         status["threads"][thread_key] = {
#             "status": "alive" if thread.is_alive() else "dead",
#             "name": thread.name
#         }
    
#     return JsonResponse(status)

# @require_http_methods(["GET"])
# @csrf_exempt
# def get_license_plate(request):
#     """
#     GET API to read license plate detection from entry file and clear it
#     """
#     file_path = "../../ALPR/detection.txt"
    
#     try:
#         # Check if file exists
#         if not os.path.exists(file_path):
#             return JsonResponse({
#                 'success': False,
#                 'message': 'Entry detection file not found',
#                 'license_plate': None
#             }, status=404)
        
#         # Read the file content
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read().strip()
        
#         # Check if file has content
#         if not content:
#             return JsonResponse({
#                 'success': True,
#                 'message': 'No license plate detected at entry',
#                 'license_plate': None
#             }, status=200)
        
#         # Clear the file after reading
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write('')
        
#         logger.info(f"Entry license plate detected and returned: {content}")
        
#         return JsonResponse({
#             'success': True,
#             'message': 'Entry license plate detected successfully',
#             'license_plate': content
#         }, status=200)
        
#     except FileNotFoundError:
#         logger.error(f"Entry detection file not found: {file_path}")
#         return JsonResponse({
#             'success': False,
#             'message': 'Entry detection file not found',
#             'license_plate': None
#         }, status=404)
        
#     except PermissionError:
#         logger.error(f"Permission denied accessing entry file: {file_path}")
#         return JsonResponse({
#             'success': False,
#             'message': 'Permission denied accessing entry detection file',
#             'license_plate': None
#         }, status=403)
        
#     except Exception as e:
#         logger.error(f"Error reading entry detection file: {str(e)}")
#         return JsonResponse({
#             'success': False,
#             'message': f'Error reading entry detection file: {str(e)}',
#             'license_plate': None
#         }, status=500)

# @require_http_methods(["GET"])
# @csrf_exempt
# def get_exit_license_plate(request):
#     """
#     GET API to read license plate detection from exit file and clear it
#     """
#     file_path = "../../ALPR/exitDetection.txt"
    
#     try:
#         # Check if file exists
#         if not os.path.exists(file_path):
#             return JsonResponse({
#                 'success': False,
#                 'message': 'Exit detection file not found',
#                 'license_plate': None
#             }, status=404)
        
#         # Read the file content
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read().strip()
        
#         # Check if file has content
#         if not content:
#             return JsonResponse({
#                 'success': True,
#                 'message': 'No license plate detected at exit',
#                 'license_plate': None
#             }, status=200)
        
#         # Clear the file after reading
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write('')
        
#         logger.info(f"Exit license plate detected and returned: {content}")
        
#         return JsonResponse({
#             'success': True,
#             'message': 'Exit license plate detected successfully',
#             'license_plate': content
#         }, status=200)
        
#     except FileNotFoundError:
#         logger.error(f"Exit detection file not found: {file_path}")
#         return JsonResponse({
#             'success': False,
#             'message': 'Exit detection file not found',
#             'license_plate': None
#         }, status=404)
        
#     except PermissionError:
#         logger.error(f"Permission denied accessing exit file: {file_path}")
#         return JsonResponse({
#             'success': False,
#             'message': 'Permission denied accessing exit detection file',
#             'license_plate': None
#         }, status=403)
        
#     except Exception as e:
#         logger.error(f"Error reading exit detection file: {str(e)}")
#         return JsonResponse({
#             'success': False,
#             'message': f'Error reading exit detection file: {str(e)}',
#             'license_plate': None
#         }, status=500)




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

