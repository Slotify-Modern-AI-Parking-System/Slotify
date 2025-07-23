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
import random
from django.contrib.auth import login
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail, EmailMultiAlternatives, EmailMessage
from django.conf import settings
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.http import HttpResponse
from .models import ContactQuery
from django.template.loader import render_to_string
from django.core.cache import cache
from django.db import models
from User.models import ParkingReservation, LicensePlateDetection
from django.utils import timezone
from django.db.models.functions import ExtractHour, TruncDate
from django.db.models import Count, Sum, Avg
from django.utils.timezone import now, timedelta
from collections import defaultdict
from django.utils.decorators import method_decorator



logger = logging.getLogger(__name__)

# <<<<<<< HEAD
# <<<<<<< HEAD
# # Define your credentials path relative to the project
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GOOGLE_CREDENTIALS_PATH = os.path.join(BASE_DIR, 'credentials', 'slotify_key.json')
# =======
# GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/Users/jainamdoshi/Desktop/Projects/Slotify/BE/decent-surf-448118-e5-3a45c35c5902.json")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
# >>>>>>> bcaa875e (Added Admin App and FE and Connected Script Trigger for Parking Lot Division.)
# =======

# Define your credentials path relative to the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOGLE_CREDENTIALS_PATH = os.path.join(BASE_DIR, 'credentials', 'slotify_key.json')

credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)
        

@csrf_exempt
@login_required
def update_profile(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            first_name = data.get("firstName")
            last_name = data.get("lastName")
            email = data.get("emailId")
            phone = data.get("contactNumber")

            owner = OwnerProfile.objects.get(user=request.user)
            owner.firstName = first_name
            owner.lastName = last_name
            owner.emailId = email
            owner.contactNumber = phone
            owner.save()

            # Also update user model's email (optional)
            request.user.email = email
            request.user.save()

            return JsonResponse({"success": True})
        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)}, status=400)

    return JsonResponse({"success": False, "message": "Invalid request"}, status=405)

@login_required
def revenue_chart_api(request):
    user = request.user
    range = request.GET.get("range", "30d")
    days = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(range, 30)

    now_time = timezone.now()
    start_date = now_time - timedelta(days=days)

    reservations = ParkingReservation.objects.filter(
        parking_lot__registered_by=user,
        created_at__gte=start_date
    )

    revenue_trend = (
        reservations.annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(total=Sum('total_amount'))
        .order_by('day')
    )

    chart_labels = [entry['day'].strftime('%Y-%m-%d') for entry in revenue_trend]
    chart_data = [float(entry['total']) for entry in revenue_trend]

    return JsonResponse({
        "labels": chart_labels,
        "data": chart_data,
    })

@csrf_exempt
@login_required
def update_logged_in_password(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        current_password = data.get("current_password")
        new_password = data.get("new_password")

        user = request.user
        if not user.check_password(current_password):
            return JsonResponse({"success": False, "message": "Incorrect current password."}, status=400)

        user.set_password(new_password)
        user.save()
        return JsonResponse({"success": True})
    return JsonResponse({"success": False, "message": "Invalid method"}, status=405)


@login_required
def revenue_view(request):
    user = request.user
    reservations = ParkingReservation.objects.filter(parking_lot__registered_by=user)

    now_time = timezone.now()
    one_month_ago = now_time - timedelta(days=30)
    recent_reservations = reservations.filter(created_at__gte=one_month_ago)

    total_earnings = reservations.aggregate(total=Sum('total_amount'))['total'] or 0
    total_bookings = reservations.count()
    average_rate = reservations.aggregate(avg=Avg('total_amount'))['avg'] or 0
    growth_rate = (
        (recent_reservations.aggregate(total=Sum('total_amount'))['total'] or 0) / total_earnings * 100
        if total_earnings else 0
    )

    # Total distinct customers (ever)
    all_customers = reservations.values('customer').distinct().count()

    # Customers who booked before the last month
    past_customers = reservations.filter(created_at__lt=one_month_ago).values_list('customer', flat=True).distinct()

    # Those who came back this month
    retained_customers = recent_reservations.filter(customer__in=past_customers).values('customer').distinct().count()

    # Final retention % calculation
    if all_customers > 0:
        customer_retention = round((retained_customers / all_customers) * 100, 1)
    else:
        customer_retention = 0

    # Additional metrics
    peak_hours_revenue = reservations.filter(created_at__hour__range=(17, 21)).aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    weekend_revenue = reservations.filter(created_at__week_day__in=[1, 7]).aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    average_session = reservations.aggregate(avg=Avg('hours'))['avg'] or 0

    # Top lots
    top_lots = (
        reservations.values('parking_lot__name')
        .annotate(bookings=Count('id'), revenue=Sum('total_amount'))
        .order_by('-revenue')[:3]
    )

    # Breakdown by session type
    hourly_count = daily_count = monthly_count = 0
    for r in reservations:
        if r.hours <= 2:
            hourly_count += 1
        elif r.hours >= 24:
            monthly_count += 1
        else:
            daily_count += 1

    total = hourly_count + daily_count + monthly_count
    if total == 0:
        breakdown = {
            "Hourly Parking": 0,
            "Daily Rates": 0,
        }
    else:
        breakdown = {
            "Hourly Parking": round(hourly_count / total * 100, 1),
            "Daily Rates": round(daily_count / total * 100, 1),
        }

    # Chart.js dynamic data: revenue grouped by date
    start_date = now_time - timedelta(days=29)
    revenue_trend = (
        reservations.filter(created_at__gte=start_date)
        .annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(total=Sum('total_amount'))
        .order_by('day')
    )

    chart_labels = [entry['day'].strftime('%Y-%m-%d') for entry in revenue_trend]
    chart_data = [float(entry['total']) for entry in revenue_trend]

    context = {
        "chart_labels": chart_labels,
        "chart_data": chart_data,
        # Include all your other metrics as you already do:
        "total_earnings": total_earnings,
        "total_bookings": total_bookings,
        "average_rate": average_rate,
        "growth_rate": growth_rate,
        "peak_hours_revenue": peak_hours_revenue,
        "weekend_revenue": weekend_revenue,
        "average_session": average_session,
        "customer_retention": customer_retention,
        "top_lots": top_lots,
        "breakdown": breakdown,
    }

    return render(request, "partials/revenue.html", context)

@csrf_exempt
def resend_otp(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')

            owner = OwnerProfile.objects.get(emailId=email)
            otp = generate_otp()
            cache.set(f"otp_{email}", otp, timeout=300)

            send_verification_email(email, f"{owner.firstName} {owner.lastName}", otp)
            return JsonResponse({'message': 'OTP sent'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=405)

@login_required
def occupancy_rate_data(request):
    lot_id = request.GET.get("lot_id")
    now = timezone.now()
    start_time = now - timedelta(days=7)

    lots = ParkingLot.objects.filter(registered_by=request.user)
    if lot_id:
        lots = lots.filter(id=lot_id)

    total_spaces = lots.aggregate(total=Sum('total_spaces'))['total'] or 1

    detections = LicensePlateDetection.objects.filter(
        entry_time__lte=now,
        exit_time__gte=start_time,
        location__in=lots
    )

    hourly_occupancy = defaultdict(int)

    # Build occupancy rate over past 7 days, hour by hour
    for hour_offset in range(0, 7 * 24):
        hour = start_time + timedelta(hours=hour_offset)
        hour_key = hour.strftime("%Y-%m-%d %H:00")
        count = 0
        for detection in detections:
            if detection.entry_time and detection.exit_time:
                if detection.entry_time <= hour and detection.exit_time >= hour:
                    count += 1
        hourly_occupancy[hour_key] = round((count / total_spaces) * 100, 1)

    return JsonResponse(hourly_occupancy)

@login_required
def revenue_data(request):
    user = request.user
    reservations = ParkingReservation.objects.filter(parking_lot__registered_by=user)

    revenue_by_day = {}
    for r in reservations:
        date_str = r.created_at.strftime("%Y-%m-%d")
        revenue_by_day[date_str] = revenue_by_day.get(date_str, 0) + float(r.total_amount)

    # Optional: sort by date
    sorted_data = dict(sorted(revenue_by_day.items()))
    return JsonResponse(sorted_data)


def get_dashboard_data_for_user(user):
    lots = ParkingLot.objects.filter(registered_by=user)
    total_lots = lots.count()
    confirmed_lots = lots.filter(confirmed=True).count()
    available_spaces = lots.aggregate(total=Sum('available_spaces'))['total'] or 0
    total_spaces = lots.aggregate(total=Sum('total_spaces'))['total'] or 1

    # Calculate overall occupancy rate
    occupancy_rate = int(((total_spaces - available_spaces) / total_spaces) * 100)

    # Get recent bookings
    recent_activities = ParkingReservation.objects.filter(parking_lot__registered_by=user).order_by('-created_at')[:2]

    try:
        owner = OwnerProfile.objects.get(user=user)
    except OwnerProfile.DoesNotExist:
        owner = None

    return {
        'totalParkingLots': total_lots,
        'totalConfirmedLots': confirmed_lots,
        'availableSpaces': available_spaces,
        'firstName': owner.firstName if owner else '',
        'lastName': owner.lastName if owner else '',
        'emailId': owner.emailId if owner else '',
        'contactNumber': owner.contactNumber if owner else '',
        'idProof': owner.idProof if owner and owner.idProof else '',
        'recentActivities': [
            {'event': 'New booking received', 'time': r.created_at.strftime("%b %d, %Y %H:%M")} for r in recent_activities
        ],
        'occupancyRate': f"{occupancy_rate}%",
        'verified': owner.verified if owner else False,
        'verificationTime': owner.verification_time.strftime("%b %d, %Y %H:%M") if owner and owner.verification_time else '',
        'owned_lots': list(lots.values("id", "name")),
    }
@login_required
def overview_partial(request):
    data = get_dashboard_data_for_user(request.user)

    # ADD THIS: Inject owned lots
    parking_lots = ParkingLot.objects.filter(registered_by=request.user)
    data["owned_lots"] = list(parking_lots.values("id", "name"))

    return render(request, 'partials/overview.html', {'dashboard_data': data})

def revenue_partial(request):
    return revenue_view(request)


@login_required
def profile_partial(request):
    data = get_dashboard_data_for_user(request.user)
    return render(request, 'partials/profile.html', {'dashboard_data': data})

def terms_of_service(request):
    return render(request, 'termsOfService.html')

@csrf_exempt
def resubmitPassword(request):
    if request.method == 'GET':
        return render(request, 'resubmitPassword.html')

    elif request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8')) if request.content_type == 'application/json' else request.POST
            email = data.get("email")

            owner = OwnerProfile.objects.get(emailId=email)
            user = owner.user  
            otp = generate_otp()

            # Cache OTP and new password
            cache.set(email + '_otp_reset', otp, timeout=300)

            send_verification_email(email, f"{user.first_name} {user.last_name}", otp)

            return JsonResponse({"message": "OTP sent.", "redirect_url": f"/verify-email/?email={email}&context=password_reset"})

        except User.DoesNotExist:
            return JsonResponse({"error": "Email not registered."}, status=400)

@csrf_exempt
def verify_reset_otp(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        otp = data.get('otp')

        cached_otp = cache.get(email + '_otp_reset')
        new_password = cache.get(email + '_new_pass')

        if not cached_otp or otp != cached_otp:
            return JsonResponse({'error': 'Invalid or expired OTP.'}, status=400)

        try:
            owner = OwnerProfile.objects.get(emailId=email)
            user = owner.user
            user.set_password(new_password)
            user.save()

            # Clear OTP and temp password
            cache.delete(email + '_otp_reset')
            cache.delete(email + '_new_pass')

            return JsonResponse({'redirect_url': '/userSignIn'})
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found'}, status=404)
        
@csrf_exempt
def render_password_final(request):
    if request.method == "GET":
        email = request.GET.get("email")
        return render(request, "resetPasswordFinal.html", {"email": email})

@csrf_exempt
def update_password(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body) if request.content_type == 'application/json' else request.POST
            email = data.get("email")
            new_password = data.get("password")

            owner = OwnerProfile.objects.get(emailId=email)
            user = owner.user
            user.set_password(new_password)
            user.save()

            # Clean up cache
            cache.delete(email + "_otp_reset")
            cache.delete(email + "_new_pass")

            return JsonResponse({"message": "Password updated."})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def send_verification_email(email, full_name, otp_code):
    subject = "Your Slotify Verification Code"
    print(f"[DEBUG] Preparing to send OTP to {email}, code: {otp_code}")

    try:
        html_content = render_to_string('otp_email.html', {
            'full_name': full_name,
            'otp_code': otp_code
        })

        email_msg = EmailMessage(
            subject=subject,
            body=html_content,
            from_email='slotify78@gmail.com',
            to=[email],
        )
        email_msg.content_subtype = "html"  # Important: sets content to HTML
        email_msg.send()

        print("[DEBUG] Email sent successfully to", email)

    except Exception as e:
        print("[ERROR] Failed to send OTP email:", str(e))

def generate_otp():
    return str(random.randint(100000, 999999))

def otp_page(request):
        email = request.GET.get('email', '')
        return render(request, 'otp.html', {'email': email})

@csrf_exempt
def verify_otp(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            otp = data.get('otp')

            if not email or not otp:
                return JsonResponse({'message': 'Missing email or OTP.'}, status=400)

            cached_otp = cache.get(f"otp_{email}")
            if cached_otp is None:
                return JsonResponse({'message': 'OTP expired. Please request a new one.'}, status=400)

            if otp == str(cached_otp):
                cache.delete(f"otp_{email}")  # Clear OTP

                # ✅ Mark user as verified in DB
                try:
                    owner = OwnerProfile.objects.get(emailId=email)
                    owner.verified = True
                    owner.verification_time = timezone.now()  # ✅ Add this line
                    owner.save()
                except OwnerProfile.DoesNotExist:
                    return JsonResponse({'message': 'Owner profile not found.'}, status=404)

                return JsonResponse({'redirect_url': '/options/'})
            else:
                return JsonResponse({'message': 'Incorrect OTP. Please try again.'}, status=400)
        except Exception as e:
            return JsonResponse({'message': f'Error: {str(e)}'}, status=500)

    return JsonResponse({'message': 'Invalid request method.'}, status=405)

def userRegister(request):
    return render(request, "userRegister.html")

def parking_register_page(request):
    return render(request, 'parkingregister.html')

def landing(request):
    return render(request, "landing.html")

def options_page(request):
    return render(request, 'parkingOptions.html')

def nearby_parking(request):
    return render(request, "parkinglist.html")

from django.views.decorators.csrf import csrf_exempt  # If you're using fetch, you'll likely need this

def userSignIn(request):
    return render(request, "userSignIn.html")

def logout_view(request):
    logout(request)  # Clears the session
    return redirect('landing')  # Send user to landing page

def about(request):
    return render(request, "about.html")

def features(request):
    return render(request, "features.html")

def contact(request):
    return render(request, "contact.html")


@csrf_exempt
def submit_contact_query(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            full_name = data.get("name")
            email = data.get("email")
            contact_number = data.get("phone")
            query = data.get("query")

            # Save to database
            ContactQuery.objects.create(
                name=full_name,
                email=email,
                phone=contact_number,
                message=query
            )

            # Render HTML email with context
            html_content = render_to_string("contact_confirmation.html", {
                "full_name": full_name,
                "email": email,
                "contact_number": contact_number,
                "query": query,
            })

            # Send to both user and admin
            subject = "Slotify - We've Received Your Message"
            email_msg = EmailMultiAlternatives(
                subject=subject,
                body=html_content,
                from_email="slotify78@gmail.com",
                to=[email],
                bcc=["slotify78@gmail.com"]
            )
            email_msg.attach_alternative(html_content, "text/html")
            email_msg.send()

            return JsonResponse({"message": "Query submitted and email sent successfully."}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

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
                return JsonResponse({'message': 'Login successful', 'user_id': user.id}, status=200)
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
            name = data.get("name")
            location = data.get("location")

            if not location:
                return JsonResponse({"error": "Location is required"}, status=400)

            # Check if location already exists in the database
            if ParkingLot.objects.filter(location__iexact=location).exists():
                return JsonResponse(
                    {"error": "Location already registered. Please choose a different location."}, 
                    status=400
                )

            if request.user.is_authenticated:
                try:
                    owner_profile = OwnerProfile.objects.get(user=request.user)
                    owner_email = owner_profile.emailId or request.user.username
                except OwnerProfile.DoesNotExist:
                    owner_email = request.user.username
            else:
                owner_email = "Unknown"

            lot = ParkingLot.objects.create(
                location=location,
                name=name,
                total_spaces=0,
                available_spaces=0,
                registered_by=request.user,
                confirmed=False
            )

            # Generate dynamic confirmation URL
            host = request.get_host()
            scheme = 'https' if request.is_secure() else 'http'
            confirmation_url = f"{scheme}://{host}/confirmParking?id={lot.id}"

            # Get the owner's email from OwnerProfile
            try:
                owner_profile = OwnerProfile.objects.get(user=lot.registered_by)
                recipient_email = owner_profile.emailId
            except OwnerProfile.DoesNotExist:
                # Fallback to user's email if OwnerProfile doesn't exist
                recipient_email = lot.registered_by.email or owner_email
                logger.warning(f"OwnerProfile not found for user {lot.registered_by.username}, using fallback email: {recipient_email}")

            send_mail(
                subject="Parking Lot Registration Confirmation Required",
                message=(
                    f"Dear {owner_profile.firstName if 'owner_profile' in locals() else lot.registered_by.username},\n\n"
                    f"Your parking lot registration has been submitted and requires confirmation:\n\n"
                    f"Parking Lot Name: {name or 'Not specified'}\n"
                    f"Address: {location}\n"
                    f"Registered by: {owner_email}\n\n"
                    f"Please click the link below to confirm your parking lot registration:\n"
                    f"{confirmation_url}\n\n"
                    f"If you did not submit this registration, please ignore this email.\n\n"
                    f"Best regards,\n"
                    f"Slotify Team"
                ),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[recipient_email],
                fail_silently=False,
            )

            return JsonResponse({"message": "Address submitted successfully."}, status=200)

        except Exception as e:
            logger.error(f"Error during parking lot submission: {str(e)}")
            return JsonResponse({"error": "Server error"}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

# @csrf_exempt
# def register_parking_lot(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             name = data.get("name")
#             location = data.get("location")

#             if not location:
#                 return JsonResponse({"error": "Location is required"}, status=400)

#             # Check if location already exists in the database
#             if ParkingLot.objects.filter(location__iexact=location).exists():
#                 return JsonResponse(
#                     {"error": "Location already registered. Please choose a different location."}, 
#                     status=400
#                 )

#             if request.user.is_authenticated:
#                 try:
#                     owner_profile = OwnerProfile.objects.get(user=request.user)
#                     owner_email = owner_profile.emailId or request.user.username
#                 except OwnerProfile.DoesNotExist:
#                     owner_email = request.user.username
#             else:
#                 owner_email = "Unknown"

#             lot = ParkingLot.objects.create(
#                 location=location,
#                 name=name,
#                 total_spaces=0,
#                 available_spaces=0,
#                 registered_by=request.user,
#                 confirmed=False
#             )

#             # Generate dynamic confirmation URL
#             host = request.get_host()
#             scheme = 'https' if request.is_secure() else 'http'
#             confirmation_url = f"{scheme}://{host}/confirmParking?id={lot.id}"

#             send_mail(
#                 subject="New Parking Lot Address Submitted for Confirmation",
#                 message=(
#                     f"A new parking lot address was submitted:\n\n"
#                     f"Address: {location}\n"
#                     f"Submitted by: {owner_email}\n\n"
#                     f"Click here to confirm:\n{confirmation_url}"
#                 ),
#                 from_email=settings.DEFAULT_FROM_EMAIL,
#                 recipient_list=["Ryan_Wallace2019@outlook.com"],  # update list to send to multiple recipients
#                 fail_silently=False,
#             )

#             return JsonResponse({"message": "Address submitted successfully."}, status=200)

#         except Exception as e:
#             logger.error(f"Error during parking lot submission: {str(e)}")
#             return JsonResponse({"error": "Server error"}, status=500)

#     return JsonResponse({"error": "Invalid request"}, status=400)

def confirm_parking(request):
    lot_id = request.GET.get('id')

    try:
        lot = ParkingLot.objects.get(id=lot_id)
        lot.confirmed = True
        lot.save()
        return HttpResponse(f"✅ Parking lot at {lot.location} has been confirmed.")
    except ParkingLot.DoesNotExist:
        return HttpResponse("❌ Parking lot not found.", status=404)

@csrf_exempt
def get_parking_lots(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_id = data.get('user_id')

            if not user_id:
                return JsonResponse({'error': 'user_id is required'}, status=400)

            # Get parking lots registered by this user
            parking_lots = ParkingLot.objects.filter(registered_by_id=user_id).values(
                "id", "name", "location", "total_spaces", "available_spaces", "confirmed","username", "password"
            )

            # Get owner's name
            try:
                owner = OwnerProfile.objects.get(user_id=user_id)
                full_name = f"{owner.firstName} {owner.lastName}"
            except OwnerProfile.DoesNotExist:
                full_name = "Unknown Owner"

            response_data = {
                "name": full_name,
                "parking_lots": list(parking_lots)
            }

            return JsonResponse(response_data, safe=False)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method. Only POST is allowed.'}, status=405)


@csrf_exempt
def register_owner(request):
    if request.method == 'POST':
        try:
            # Get data from the request
            first_name = request.POST.get('firstName')
            last_name = request.POST.get('lastName')
            email_id = request.POST.get('emailId')
            id_proof_file = request.FILES.get('idProof')
            password = request.POST.get('password')
            contact_number = request.POST.get('contactNumber')

            if not all([first_name, last_name, email_id, password, contact_number]):
                return JsonResponse({'error': 'All fields are required.'}, status=400)

            if User.objects.filter(username=email_id).exists():
                return JsonResponse({'error': 'Email ID already exists.'}, status=400)
            if OwnerProfile.objects.filter(contactNumber=contact_number).exists():
                return JsonResponse({'error': 'Contact Number already exists.'}, status=400)

            # Create user
            user = User.objects.create_user(
                username=email_id,
                email=email_id,
                password=password
            )

            hashed_password = make_password(password)

            # Create owner profile
            owner = OwnerProfile.objects.create(
                user=user,
                firstName=first_name,
                lastName=last_name,
                emailId=email_id,
                password=hashed_password,
                contactNumber=contact_number,
                verified=False
            )

# <<<<<<< HEAD
            if id_proof_file:
                storage_client = storage.Client(credentials=credentials)
                bucket_name = "slotifydocument3"  # Your Google Cloud Storage bucket
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

# =======
# >>>>>>> bcaa875e (Added Admin App and FE and Connected Script Trigger for Parking Lot Division.)
            login(request, user)

# Generate OTP and send email
            otp = generate_otp()
            cache.set(f"otp_{email_id}", otp, timeout=300)

            print("[DEBUG] OTP to send:", otp)

            send_verification_email(user.email, user.first_name, otp)

            # Redirect to OTP page
            return JsonResponse({'redirect_url': f'/verify-email/?email={email_id}'}, status=200)


        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method. Only POST is allowed.'}, status=405)

def get_owner_dashboard(request):
    """Fetches owner dashboard details including total registered parking lots and verification status."""
    parking_lots = ParkingLot.objects.filter(registered_by=request.user)
    total_lots = parking_lots.count()
    total_confirmed = parking_lots.filter(confirmed=True).count()
    total_available_spaces = sum(lot.available_spaces for lot in parking_lots)
    # Check if the user is authenticated (default Django User model check)
    if not request.user.is_authenticated:
        return redirect('userRegister')  # Redirect to registration page if not authenticated

    try:
        owner = OwnerProfile.objects.get(user=request.user)

        if not owner.verified:
            return render(request, 'verification.html', {
                'email': owner.emailId  # Send this so JS can access it
            })

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
            "idProof": owner.idProof,
            "totalConfirmedLots": total_confirmed,
            "owned_lots": list(parking_lots.values("id", "name"))
        }

        return render(request, "ownerDashboard.html", {"dashboard_data": dashboard_data})

    except OwnerProfile.DoesNotExist:
        return JsonResponse({'error': 'Owner profile not found'}, status=404)
    except Exception as e:
        logger.error(f"Error fetching owner dashboard: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def user_summary(request):
    if request.method == "GET":
        # ✅ Active owners (role = "Owner", active = True)
        active_owners = OwnerProfile.objects.filter(role="Owner", active=True)

        # ✅ Active users (role = "User", active = True)
        active_users = OwnerProfile.objects.filter(role="User", active=True)

        # ✅ Inactive users or owners (active = False)
        inactive_profiles = OwnerProfile.objects.filter(active=False)

        return JsonResponse({
            "total_active_users": active_users.count() + active_owners.count(),
            "total_active_owners": active_owners.count(),
            "total_active_users_only": active_users.count(),
            "total_inactive_owners_and_users": inactive_profiles.count(),
        })
    else:
        return JsonResponse({"error": "Only GET method allowed."}, status=405)


@csrf_exempt
def get_landingPageStats(request):
    if request.method == 'GET':
        owners_count = OwnerProfile.objects.count()
        parking_lot_count = ParkingLot.objects.count()

        return JsonResponse({
            'owners_count': owners_count,
            'parking_lot_count': parking_lot_count
        })
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
