from django.shortcuts import render
from django.contrib.auth.models import User 
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import storage
import os
import json
from django.contrib.auth.hashers import make_password
from google.oauth2 import service_account
from datetime import timedelta
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
import logging
from django.contrib.auth import login
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.http import HttpResponse
import os
import subprocess
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import logging
from slotifyBE.models import *


# Create your views here.

def adminLogin(request):
    return render(request, "adminLogin.html")

def adminDashboard(request):
    return render(request, 'adminDashboard.html')

@csrf_exempt
def login_admin(request):
    if request.method == 'POST':
        try:
            # Load JSON data from request body
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')

            if not email or not password:
                return JsonResponse({'error': 'Email and password are required'}, status=400)

            user = authenticate(request, username=email, password=password)

            if user is not None:
                login(request, user)
                return JsonResponse({'message': 'Login successful'}, status=200)
            else:
                return JsonResponse({'error': 'Invalid email or password'}, status=401)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method. Only POST is allowed.'}, status=405)

@csrf_exempt
def unconfirmed_parkinglots(request):
    if request.method == "GET":
        parking_lots = ParkingLot.objects.filter(confirmed=False)
        result = []

        for lot in parking_lots:
            try:
                owner_profile = lot.registered_by.owner_profile
                owner_data = {
                    'ownerId': owner_profile.id,
                    'firstName': owner_profile.firstName,
                    'lastName': owner_profile.lastName,
                    'emailId': owner_profile.emailId,
                    'contactNumber': owner_profile.contactNumber,
                    'idProof': owner_profile.idProof,
                    'verified': owner_profile.verified
                }
            except OwnerProfile.DoesNotExist:
                owner_data = None

            result.append({
                'parkingLotId': lot.id,
                'name': lot.name,
                'location': lot.location,
                'totalSpaces': lot.total_spaces,
                'availableSpaces': lot.available_spaces,
                'confirmed': lot.confirmed,
                'owner': owner_data
            })

        return JsonResponse({'unconfirmedParkingLots': result}, status=200)
    else:
        return JsonResponse({'error': 'Only GET method allowed'}, status=405)

@csrf_exempt
def dashboard_counts(request):
    if request.method == "GET":
        try:
            # Get counts for different parking lot statuses
            pending_count = ParkingLot.objects.filter(confirmed=False).count()
            approved_count = ParkingLot.objects.filter(confirmed=True).count()
            total_count = ParkingLot.objects.count()
            
            return JsonResponse({
                'pendingParkingLots': pending_count,
                'approvedParkingLots': approved_count,
                'totalParkingLots': total_count
            }, status=200)
            
        except Exception as e:
            return JsonResponse({
                'error': 'Failed to fetch dashboard counts',
                'details': str(e)
            }, status=500)
    else:
        return JsonResponse({'error': 'Only GET method allowed'}, status=405)

import sys
@csrf_exempt
@require_http_methods(["POST"])
def run_python_script(request):
    """
    API endpoint to run Python scripts and update ParkingLot confirmation
    Expected JSON payload: {
        "script_path": "relative/path/to/script.py OR absolute/path/to/script.py",
        "address": "optional address parameter"
    }
    """
    try:
        # Parse JSON data from request body
        data = json.loads(request.body)
        script_path = data.get('script_path')
        address = data.get('address', '')  # Optional address field
        
        if not script_path:
            return JsonResponse({
                'success': False,
                'error': 'script_path is required'
            }, status=400)
        
        # Handle both absolute and relative paths
        if os.path.isabs(script_path):
            # Absolute path provided
            full_script_path = os.path.normpath(script_path)
        else:
            # Relative path - construct full path
            full_script_path = os.path.normpath(os.path.join(r"C:\Users\jigsp\OneDrive\Desktop\Slotify\BE\parking_lot\Address",script_path))
        
        # Security check: ensure script is within project directory (only for relative paths)
        if not os.path.isabs(script_path):
            if not os.path.abspath(full_script_path).startswith(os.path.abspath(settings.BASE_DIR)):
                return JsonResponse({
                    'success': False,
                    'error': 'Script path not allowed'
                }, status=403)
        
        # Check if script exists
        if not os.path.exists(full_script_path):
            return JsonResponse({
                'success': False,
                'error': f'Script not found: {full_script_path}'
            }, status=404)
        
        # Change to script directory to handle relative imports
        script_dir = os.path.dirname(full_script_path)
        original_cwd = os.getcwd()
        
        try:
            # Validate script file permissions and readability
            try:
                with open(full_script_path, 'r') as f:
                    first_line = f.readline().strip()
                    print(f"Script first line: {first_line}")
            except PermissionError as pe:
                print(f"Permission error reading script: {pe}")
                return JsonResponse({
                    'success': False,
                    'error': f'Permission denied reading script: {full_script_path}',
                    'details': str(pe)
                }, status=403)
            except Exception as fe:
                print(f"Error reading script file: {fe}")
                return JsonResponse({
                    'success': False,
                    'error': f'Cannot read script file: {full_script_path}',
                    'details': str(fe)
                }, status=400)
            
            # Change working directory with error handling
            if script_dir:
                try:
                    os.chdir(script_dir)
                    print(f"Changed working directory to: {script_dir}")
                except OSError as oe:
                    print(f"Failed to change directory to {script_dir}: {oe}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Cannot access script directory: {script_dir}',
                        'details': str(oe)
                    }, status=400)
            
            # Use sys.executable to get the correct Python interpreter
            python_executable = sys.executable
            script_filename = os.path.basename(full_script_path)
            
            # Build command with detailed logging
            cmd = [python_executable, script_filename]
            if address:
                cmd.extend(['--address', address])
            
            print(f"Python executable: {python_executable}")
            print(f"Script filename: {script_filename}")
            print(f"Full script path: {full_script_path}")
            print(f"Script directory: {script_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Executing command: {' '.join(cmd)}")
            print(f"Command arguments: {cmd}")
            
            # Check if Python executable exists and is accessible
            if not os.path.exists(python_executable):
                print(f"Python executable not found: {python_executable}")
                return JsonResponse({
                    'success': False,
                    'error': f'Python executable not found: {python_executable}'
                }, status=500)
            
            # Verify script file exists in current directory after chdir
            if not os.path.exists(script_filename):
                print(f"Script file not found in current directory: {script_filename}")
                print(f"Files in current directory: {os.listdir('.')}")
                return JsonResponse({
                    'success': False,
                    'error': f'Script file not accessible: {script_filename}',
                    'current_dir': os.getcwd(),
                    'files_in_dir': os.listdir('.')
                }, status=404)
            
            try:
                # Run the Python script with comprehensive error handling
                print("Starting subprocess execution...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    shell=False,  # Explicitly set shell=False for security
                    cwd=None,  # Use current working directory (already changed above)
                    env=os.environ.copy()  # Pass environment variables
                )
                print(f"Subprocess completed with return code: {result.returncode}")
                
            except FileNotFoundError as fnf:
                print(f"FileNotFoundError during subprocess execution: {fnf}")
                return JsonResponse({
                    'success': False,
                    'error': 'Python executable or script not found during execution',
                    'details': str(fnf),
                    'command': ' '.join(cmd)
                }, status=404)
                
            except PermissionError as pe:
                print(f"PermissionError during subprocess execution: {pe}")
                return JsonResponse({
                    'success': False,
                    'error': 'Permission denied executing script',
                    'details': str(pe),
                    'command': ' '.join(cmd)
                }, status=403)
                
            except OSError as oe:
                print(f"OSError during subprocess execution: {oe}")
                return JsonResponse({
                    'success': False,
                    'error': 'Operating system error during script execution',
                    'details': str(oe),
                    'command': ' '.join(cmd)
                }, status=500)
            
            # Log subprocess output for debugging
            print(f"Script STDOUT: {result.stdout}")
            print(f"Script STDERR: {result.stderr}")
            print(f"Return code: {result.returncode}")
            
            # Analyze return code and provide specific error messages
            if result.returncode != 0:
                print(f"Script execution failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Error details from stderr: {result.stderr}")
                if result.returncode == 1:
                    error_msg = "Script execution failed - General error"
                elif result.returncode == 2:
                    error_msg = "Script execution failed - Invalid arguments or usage"
                elif result.returncode == 126:
                    error_msg = "Script execution failed - Permission denied or not executable"
                elif result.returncode == 127:
                    error_msg = "Script execution failed - Command not found"
                elif result.returncode == -9:
                    error_msg = "Script execution failed - Process was killed (possibly out of memory)"
                else:
                    error_msg = f"Script execution failed with return code: {result.returncode}"
                
                print(f"Interpreted error: {error_msg}")
            
            # Prepare base response
            response_data = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'address_used': address,
                'command_executed': ' '.join(cmd),
                'working_directory': os.getcwd(),
                'python_executable': python_executable,
                'script_path_used': full_script_path,
                'script_directory': script_dir
            }
            
            # Add interpretation of common error codes
            if result.returncode != 0:
                if result.returncode == 1:
                    response_data['error_interpretation'] = "General script error - check script logic and stderr"
                elif result.returncode == 2:
                    response_data['error_interpretation'] = "Invalid arguments passed to script"
                elif result.returncode == 126:
                    response_data['error_interpretation'] = "Permission denied or script not executable"
                elif result.returncode == 127:
                    response_data['error_interpretation'] = "Command not found - check Python path"
                elif result.returncode == -9:
                    response_data['error_interpretation'] = "Process killed - possibly out of memory"
                else:
                    response_data['error_interpretation'] = f"Unknown error code: {result.returncode}"
            
            # If script executed successfully and address is provided, update ParkingLot
            if result.returncode == 0 and address:
                try:
                    # Find parking lots matching the address (case-insensitive partial match)
                    parking_lots = ParkingLot.objects.filter(
                        location__icontains=address
                    )
                    
                    if parking_lots.exists():
                        # Update all matching parking lots to confirmed
                        updated_count = parking_lots.update(confirmed=True)
                        
                        response_data.update({
                            'parking_lot_updated': True,
                            'updated_count': updated_count,
                            'updated_locations': list(parking_lots.values_list('location', flat=True))
                        })
                    else:
                        response_data.update({
                            'parking_lot_updated': False,
                            'message': f'No parking lots found matching address: {address}'
                        })
                        
                except Exception as db_error:
                    response_data.update({
                        'parking_lot_updated': False,
                        'db_error': f'Error updating parking lot: {str(db_error)}'
                    })
            elif result.returncode == 0 and not address:
                response_data.update({
                    'parking_lot_updated': False,
                    'message': 'No address provided for parking lot confirmation'
                })
            else:
                response_data.update({
                    'parking_lot_updated': False,
                    'message': 'Script execution failed, parking lot not updated'
                })
            
            return JsonResponse(response_data)
            
        except subprocess.TimeoutExpired as te:
            print(f"Subprocess timeout expired: {te}")
            print(f"Command that timed out: {' '.join(cmd)}")
            return JsonResponse({
                'success': False,
                'error': 'Script execution timed out (300 seconds)',
                'parking_lot_updated': False,
                'command': ' '.join(cmd),
                'timeout_details': str(te)
            }, status=408)
            
        except Exception as subprocess_error:
            print(f"Unexpected error during subprocess execution: {subprocess_error}")
            print(f"Error type: {type(subprocess_error).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return JsonResponse({
                'success': False,
                'error': f'Script execution error: {str(subprocess_error)}',
                'error_type': type(subprocess_error).__name__,
                'parking_lot_updated': False,
                'command': ' '.join(cmd) if 'cmd' in locals() else 'Command not constructed'
            }, status=500)
            
        finally:
            # Restore original working directory with error handling
            try:
                os.chdir(original_cwd)
                print(f"Restored working directory to: {original_cwd}")
            except OSError as oe:
                print(f"Warning: Failed to restore original working directory: {oe}")
                # Don't fail the request for this, just log it
            
    except json.JSONDecodeError as jde:
        print(f"JSON decode error: {jde}")
        print(f"Request body: {request.body}")
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body',
            'details': str(jde)
        }, status=400)
        
    except Exception as e:
        print(f"Unexpected error in run_python_script: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_type': type(e).__name__
        }, status=500)
import os
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Set your target folder path
TARGET_FOLDER = r"C:\Users\jigsp\OneDrive\Desktop\Slotify\BE\parking_lot\Address"


@csrf_exempt
def upload_image(request):
    if request.method != 'POST':
        return HttpResponseBadRequest("Only POST method is allowed.")

    image = request.FILES.get('image')
    address = request.POST.get('address')

    if not image or not address:
        return HttpResponseBadRequest("Missing 'image' or 'address' parameter.")

    # Ensure the target folder exists
    os.makedirs(TARGET_FOLDER, exist_ok=True)

    # Get file extension (e.g., .jpg, .png)
    _, ext = os.path.splitext(image.name)
    
    # Keep the original address format without replacing spaces with underscores
    filename = f"{address}{ext}"

    file_path = os.path.join(TARGET_FOLDER, filename)

    # Save image
    with open(file_path, 'wb+') as destination:
        for chunk in image.chunks():
            destination.write(chunk)

    return JsonResponse({'message': 'Image uploaded successfully.', 'saved_as': filename})
