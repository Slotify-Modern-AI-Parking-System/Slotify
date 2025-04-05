from django.contrib.auth.models import User 
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import storage
from .models import ParkingLot, OwnerProfile
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

logger = logging.getLogger(__name__)

GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "C:/Users/Ryan_/Downloads/decent-surf-448118-e5-44d0948444db.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)

def userRegister(request):
    return render(request, "userRegister.html")

def parking_register_page(request):
    return render(request, 'parkingregister.html')

def landing(request):
    return render(request, "landing.html")

def options_page(request):
    return render(request, 'parkingOptions.html')

from django.views.decorators.csrf import csrf_exempt  # If you're using fetch, you'll likely need this

def userSignIn(request):
    return render(request, "userSignIn.html")

def logout_view(request):
    logout(request)  # Clears the session
    return redirect('landing')  # Send user to landing page

@csrf_exempt
def login_owner(request):
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
            logger.error(f"Error during login: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method. Only POST is allowed.'}, status=405)

@csrf_exempt
def register_parking_lot(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            location = data.get("location")

            if not location:
                return JsonResponse({"error": "Location is required"}, status=400)

            owner_email = request.user.email if request.user.is_authenticated else "Unknown"

            # Send confirmation email to devs
            send_mail(
                subject="New Parking Lot Address Submitted for Confirmation",
                message=f"A new parking lot address was submitted:\n\nAddress: {location}\nSubmitted by: {owner_email}",
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=["Ryan_Wallace2019@outlook.com"],  # put your email address here you can add multiple emails a@gmail.com, b@gmail.com
                fail_silently=False,
            )

            return JsonResponse({"message": "Address submitted successfully."}, status=200)

        except Exception as e:
            logger.error(f"Error during parking lot submission: {str(e)}")
            return JsonResponse({"error": "Server error"}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

def get_parking_lots(request):
    parking_lots = ParkingLot.objects.all().values("id", "name", "location", "total_spaces", "available_spaces")
    return JsonResponse(list(parking_lots), safe=False)

@csrf_exempt
def register_owner(request):
    if request.method == 'POST':
        try:
            # Get data from the request
            first_name = request.POST.get('firstName')
            last_name = request.POST.get('lastName')
            email_id = request.POST.get('emailId')
            password = request.POST.get('password')
            contact_number = request.POST.get('contactNumber')
            id_proof_file = request.FILES.get('idProof')  # The file uploaded

            if not all([first_name, last_name, email_id, password, contact_number]):
                return JsonResponse({'error': 'All fields except idProof are required.'}, status=400)

            if User.objects.filter(username=email_id).exists():
                return JsonResponse({'error': 'Email ID already exists.'}, status=400)
            if OwnerProfile.objects.filter(contactNumber=contact_number).exists():
                return JsonResponse({'error': 'Contact Number already exists.'}, status=400)

            user = User.objects.create_user(
                username=email_id,
                password=password
            )

            hashed_password = make_password(password)

            owner = OwnerProfile.objects.create(
                user=user, 
                firstName=first_name,
                lastName=last_name,
                emailId=email_id,
                password=hashed_password, 
                contactNumber=contact_number,
                verified=False
            )

            if id_proof_file:
                storage_client = storage.Client()
                bucket_name = "slotifydocuments"  # Your Google Cloud Storage bucket
                bucket = storage_client.bucket(bucket_name)

                new_file_name = f"{owner.id}_{first_name}_{last_name}".replace(" ", "_")
                blob = bucket.blob(f"id_proofs/{new_file_name}")


                blob.upload_from_file(id_proof_file, content_type=id_proof_file.content_type)

                signed_url = blob.generate_signed_url(
                    expiration=timedelta(hours=1),
                    method='GET'
                )

                owner.idProof = signed_url
                owner.save()

            login(request, user)

            return JsonResponse({'message': 'Owner registered successfully!'}, status=201)

        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method. Only POST is allowed.'}, status=405)

def get_owner_dashboard(request):
    """Fetches owner dashboard details including total registered parking lots and verification status."""

    # Check if the user is authenticated (default Django User model check)
    if not request.user.is_authenticated:
        return redirect('userRegister')  # Redirect to registration page if not authenticated

    try:
        owner = OwnerProfile.objects.get(user=request.user)

        parking_lots = ParkingLot.objects.filter(registered_by=request.user)

        total_lots = parking_lots.count()
        total_available_spaces = sum(lot.available_spaces for lot in parking_lots)

        # Prepare the data to pass to the template
        dashboard_data = {
            "firstName": owner.firstName,
            "lastName": owner.lastName,
            "emailId": owner.emailId,
            "totalParkingLots": total_lots,
            "availableSpaces": total_available_spaces,
            "idProof": owner.idProof 
        }

        return render(request, "ownerDashboard.html", {"dashboard_data": dashboard_data})

    except OwnerProfile.DoesNotExist:
        return JsonResponse({'error': 'Owner profile not found'}, status=404)
    except Exception as e:
        logger.error(f"Error fetching owner dashboard: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
