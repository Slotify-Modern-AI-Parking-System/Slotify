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
@require_http_methods(["POST"])
def run_python_script(request):
    """
    API endpoint to run Python scripts
    Expected JSON payload: {"script_path": "relative/path/to/script.py"}
    """
    try:
        # Parse JSON data from request body
        data = json.loads(request.body)
        script_path = data.get('script_path')
        
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
            
            # Run the Python script
            result = subprocess.run(
                ['python', os.path.basename(full_script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return JsonResponse({
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            })
            
        except subprocess.TimeoutExpired:
            return JsonResponse({
                'success': False,
                'error': 'Script execution timed out'
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
        logger.error(f"Error running script: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)