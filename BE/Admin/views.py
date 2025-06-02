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


# @csrf_exempt
# @require_http_methods(["POST"])
# def run_python_script(request):
#     """
#     API endpoint to run Python scripts
#     Expected JSON payload: {
#         "script_path": "relative/path/to/script.py",
#         "address": "optional address parameter"
#     }
#     """
#     try:
#         # Parse JSON data from request body
#         data = json.loads(request.body)
#         script_path = data.get('script_path')
#         address = data.get('address', '')  # Optional address field
        
#         if not script_path:
#             return JsonResponse({
#                 'success': False,
#                 'error': 'script_path is required'
#             }, status=400)
        
#         # Construct full path to script
#         full_script_path = os.path.join(settings.BASE_DIR, script_path)
        
#         # Security check: ensure script is within project directory
#         if not os.path.abspath(full_script_path).startswith(os.path.abspath(settings.BASE_DIR)):
#             return JsonResponse({
#                 'success': False,
#                 'error': 'Script path not allowed'
#             }, status=403)
        
#         # Check if script exists
#         if not os.path.exists(full_script_path):
#             return JsonResponse({
#                 'success': False,
#                 'error': f'Script not found: {script_path}'
#             }, status=404)
        
#         # Change to script directory to handle relative imports
#         script_dir = os.path.dirname(full_script_path)
#         original_cwd = os.getcwd()
        
#         try:
#             os.chdir(script_dir)
            
#             # Prepare command with optional address argument
#             cmd = ['python', os.path.basename(full_script_path)]
#             if address:
#                 cmd.extend(['--address', address])
            
#             # Run the Python script
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 timeout=300  # 5 minute timeout
#             )
            
#             return JsonResponse({
#                 'success': result.returncode == 0,
#                 'stdout': result.stdout,
#                 'stderr': result.stderr,
#                 'return_code': result.returncode,
#                 'address_used': address  # Include address in response for reference
#             })
            
#         except subprocess.TimeoutExpired:
#             return JsonResponse({
#                 'success': False,
#                 'error': 'Script execution timed out'
#             }, status=408)
            
#         finally:
#             # Restore original working directory
#             os.chdir(original_cwd)
            
#     except json.JSONDecodeError:
#         return JsonResponse({
#             'success': False,
#             'error': 'Invalid JSON in request body'
#         }, status=400)
        
#     except Exception as e:
#         logger.error(f"Error running script: {str(e)}")
#         return JsonResponse({
#             'success': False,
#             'error': f'Internal server error: {str(e)}'
#         }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_python_script(request):
    """
    API endpoint to run Python scripts and update ParkingLot confirmation
    Expected JSON payload: {
        "script_path": "relative/path/to/script.py",
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
        
        # Construct full path to script
        full_script_path = os.path.join(settings.BASE_DIR, script_path)
        
        # Security check: ensure script is within project directory
        if not os.path.abspath(full_script_path).startswith(os.path.abspath(settings.BASE_DIR)):
            return JsonResponse({
                'success': False,
                'error': 'Script path not allowed'
            }, status=403)
        
        # Check if script exists
        if not os.path.exists(full_script_path):
            return JsonResponse({
                'success': False,
                'error': f'Script not found: {script_path}'
            }, status=404)
        
        # Change to script directory to handle relative imports
        script_dir = os.path.dirname(full_script_path)
        original_cwd = os.getcwd()
        
        try:
            os.chdir(script_dir)
            
            # Prepare command with optional address argument
            cmd = ['python', os.path.basename(full_script_path)]
            if address:
                cmd.extend(['--address', address])
            
            # Run the Python script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Prepare base response
            response_data = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'address_used': address
            }
            
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
            
        except subprocess.TimeoutExpired:
            return JsonResponse({
                'success': False,
                'error': 'Script execution timed out',
                'parking_lot_updated': False
            }, status=408)
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

import os
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Set your target folder path
TARGET_FOLDER = "/Users/jainamdoshi/Desktop/Projects/Slotify/BE/parking_lot/Address"

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
