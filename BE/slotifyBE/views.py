from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import storage
import os
import json
from .models import *
from django.contrib.auth.hashers import make_password
from google.oauth2 import service_account
from datetime import timedelta
from google.cloud import storage
from django.contrib.auth.hashers import make_password
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def userRegister(request):
    return render(request, "userRegister.html")

GOOGLE_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), '/Users/jainamdoshi/Desktop/Slotify/Slotify/decent-surf-448118-e5-3a45c35c5902.json')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)

@csrf_exempt
def register_owner(request):
    if request.method == 'POST':
        try:
            first_name = request.POST.get('firstName')
            last_name = request.POST.get('lastName')
            email_id = request.POST.get('emailId')
            password = request.POST.get('password')
            contact_number = request.POST.get('contactNumber')
            id_proof_file = request.FILES.get('idProof')

            if not all([first_name, last_name, email_id, password, contact_number]):
                return JsonResponse({'error': 'All fields except idProof are required.'}, status=400)

            if OwnerProfile.objects.filter(emailId=email_id).exists():
                return JsonResponse({'error': 'Email ID already exists.'}, status=400)
            if OwnerProfile.objects.filter(contactNumber=contact_number).exists():
                return JsonResponse({'error': 'Contact Number already exists.'}, status=400)

            hashed_password = make_password(password)

            owner = OwnerProfile.objects.create(
                firstName=first_name,
                lastName=last_name,
                emailId=email_id,
                password=hashed_password,
                contactNumber=contact_number,
                verified=False
            )

            if id_proof_file:
                bucket_name = "slotifydocuments"

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)

                new_file_name = f"{owner.id}_{first_name}_{last_name}".replace(" ", "_")
                blob = bucket.blob(f"id_proofs/{new_file_name}")

                blob.upload_from_file(id_proof_file.file, content_type=id_proof_file.content_type)
                
                url_expiration = timedelta(hours=1)  
                id_proof_url = blob.generate_signed_url(expiration=url_expiration, version="v4")
                owner.idProof = id_proof_url
                owner.save()
                
            return JsonResponse({'message': 'Registration successful!'}, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method. Only POST is allowed.'}, status=405)
