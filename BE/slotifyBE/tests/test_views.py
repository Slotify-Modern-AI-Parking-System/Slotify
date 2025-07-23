# from django.test import TestCase, Client
# from django.core.files.uploadedfile import SimpleUploadedFile
# from django.contrib.auth.models import User
# from slotifyBE.models import OwnerProfile
# from django.urls import reverse

# class OptionsPageTest(TestCase):
#     def setUp(self):
#         self.client = Client()

#     def test_options_page_loads(self):
#         response = self.client.get('/options/')
#         self.assertEqual(response.status_code, 200)
#         self.assertContains(response, "Welcome to Slotify")


# File Location: slotifyBE/tests/test_views.py
# Create this directory structure: slotifyBE/tests/__init__.py and slotifyBE/tests/test_views.py

import json
import os
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.cache import cache
from django.core import mail
from django.utils import timezone
from datetime import timedelta
from slotifyBE.models import OwnerProfile, ParkingLot, ContactQuery
from User.models import ParkingReservation, LicensePlateDetection


class ViewsTestCase(TestCase):
    def setUp(self):
        """Set up test data"""
        self.client = Client()
        
        # Create test user
        self.user = User.objects.create_user(
            username='testowner@test.com',
            email='testowner@test.com',
            password='testpass123'
        )
        
        # Create owner profile
        self.owner_profile = OwnerProfile.objects.create(
            user=self.user,
            firstName='Test',
            lastName='Owner',
            emailId='testowner@test.com',
            password='hashedpass',
            contactNumber='1234567890',
            verified=True,
            verification_time=timezone.now()
        )
        
        # Create parking lot
        self.parking_lot = ParkingLot.objects.create(
            name='Test Parking Lot',
            location='123 Test Street',
            total_spaces=100,
            available_spaces=50,
            registered_by=self.user,
            confirmed=True
        )

    def tearDown(self):
        """Clean up after tests"""
        cache.clear()

    # Authentication Views Tests
    def test_register_owner_success(self):
        """Test successful owner registration"""
        with patch('slotifyBE.views.send_verification_email') as mock_email:
            response = self.client.post('/registerOwner/', {
                'firstName': 'New',
                'lastName': 'Owner',
                'emailId': 'newowner@test.com',
                'password': 'newpass123',
                'contactNumber': '0987654321'
            })
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.content)
            self.assertIn('redirect_url', response_data)
            self.assertTrue(User.objects.filter(username='newowner@test.com').exists())
            self.assertTrue(OwnerProfile.objects.filter(emailId='newowner@test.com').exists())
            mock_email.assert_called_once()

    def test_register_owner_duplicate_email(self):
        """Test owner registration with duplicate email"""
        response = self.client.post('/registerOwner/', {
            'firstName': 'Duplicate',
            'lastName': 'User',
            'emailId': 'testowner@test.com',  # Already exists
            'password': 'pass123',
            'contactNumber': '1111111111'
        })
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'Email ID already exists.')

    def test_login_owner_success(self):
        """Test successful owner login"""
        response = self.client.post('/loginOwner/', 
            json.dumps({
                'email': 'testowner@test.com',
                'password': 'testpass123'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['message'], 'Login successful')
        self.assertEqual(response_data['user_id'], self.user.id)

    def test_login_owner_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = self.client.post('/loginOwner/',
            json.dumps({
                'email': 'testowner@test.com',
                'password': 'wrongpassword'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 401)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'Invalid email or password')

    def test_logout_view(self):
        """Test logout functionality"""
        self.client.login(username='testowner@test.com', password='testpass123')
        response = self.client.get('/logout/')
        
        self.assertEqual(response.status_code, 302)  # Redirect to landing
        self.assertEqual(response.url, '/')

    # OTP and Verification Tests
    def test_verify_otp_success(self):
        """Test successful OTP verification"""
        # Set OTP in cache
        cache.set('otp_testowner@test.com', '123456', timeout=300)
        
        # Mark owner as unverified for test
        self.owner_profile.verified = False
        self.owner_profile.save()
        
        response = self.client.post('/verify-otp/',
            json.dumps({
                'email': 'testowner@test.com',
                'otp': '123456'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['redirect_url'], '/options/')
        
        # Check that owner is now verified
        self.owner_profile.refresh_from_db()
        self.assertTrue(self.owner_profile.verified)

    def test_verify_otp_invalid(self):
        """Test OTP verification with invalid OTP"""
        cache.set('otp_testowner@test.com', '123456', timeout=300)
        
        response = self.client.post('/verify-otp/',
            json.dumps({
                'email': 'testowner@test.com',
                'otp': '654321'  # Wrong OTP
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['message'], 'Incorrect OTP. Please try again.')

    def test_resend_otp(self):
        """Test OTP resend functionality"""
        with patch('slotifyBE.views.send_verification_email') as mock_email:
            response = self.client.post('/resend-otp/',
                json.dumps({'email': 'testowner@test.com'}),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.content)
            self.assertEqual(response_data['message'], 'OTP sent')
            mock_email.assert_called_once()

    # Dashboard and Profile Tests
    def test_get_owner_dashboard_authenticated(self):
        """Test owner dashboard access when authenticated and verified"""
        self.client.login(username='testowner@test.com', password='testpass123')
        response = self.client.get('/ownerDashboard/')
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test')
        self.assertContains(response, 'Owner')

    def test_get_owner_dashboard_unverified(self):
        """Test owner dashboard when user is unverified"""
        self.owner_profile.verified = False
        self.owner_profile.save()
        
        self.client.login(username='testowner@test.com', password='testpass123')
        response = self.client.get('/ownerDashboard/')
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'verification.html')

    def test_update_profile_success(self):
        """Test successful profile update"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        response = self.client.post('/update-profile/',
            json.dumps({
                'firstName': 'Updated',
                'lastName': 'Name',
                'emailId': 'updated@test.com',
                'contactNumber': '9999999999'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])
        
        # Check database update
        self.owner_profile.refresh_from_db()
        self.assertEqual(self.owner_profile.firstName, 'Updated')

    def test_update_logged_in_password_success(self):
        """Test password update with correct current password"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        response = self.client.post('/update-logged-in-password/',
            json.dumps({
                'current_password': 'testpass123',
                'new_password': 'newtestpass123'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])

    def test_update_logged_in_password_wrong_current(self):
        """Test password update with incorrect current password"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        response = self.client.post('/update-logged-in-password/',
            json.dumps({
                'current_password': 'wrongpassword',
                'new_password': 'newtestpass123'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['message'], 'Incorrect current password.')

    # Parking Lot Tests
    def test_register_parking_lot_success(self):
        """Test successful parking lot registration"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        with patch('slotifyBE.views.send_mail') as mock_mail:
            response = self.client.post('/submitParking/',
                json.dumps({
                    'name': 'New Parking Lot',
                    'location': '456 New Street'
                }),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.content)
            self.assertEqual(response_data['message'], 'Address submitted successfully.')
            
            # Check database
            self.assertTrue(ParkingLot.objects.filter(location='456 New Street').exists())
            mock_mail.assert_called_once()

    def test_register_parking_lot_duplicate_location(self):
        """Test parking lot registration with duplicate location"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        response = self.client.post('/submitParking/',
            json.dumps({
                'name': 'Duplicate Location',
                'location': '123 Test Street'  # Already exists
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'Location already registered. Please choose a different location.')

    def test_confirm_parking_success(self):
        """Test parking lot confirmation"""
        unconfirmed_lot = ParkingLot.objects.create(
            name='Unconfirmed Lot',
            location='789 Unconfirmed St',
            registered_by=self.user,
            confirmed=False
        )
        
        response = self.client.get(f'/confirmParking/?id={unconfirmed_lot.id}')
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '✅ Parking lot at 789 Unconfirmed St has been confirmed.')
        
        # Check database update
        unconfirmed_lot.refresh_from_db()
        self.assertTrue(unconfirmed_lot.confirmed)

    def test_get_parking_lots_success(self):
        """Test retrieving parking lots for a user"""
        response = self.client.post('/getParkingLots/',
            json.dumps({'user_id': self.user.id}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['name'], 'Test Owner')
        self.assertEqual(len(response_data['parking_lots']), 1)
        self.assertEqual(response_data['parking_lots'][0]['location'], '123 Test Street')

    # Revenue and Analytics Tests
    def test_revenue_view_authenticated(self):
        """Test revenue view access"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        # Create mock reservation for testing
        with patch('User.models.ParkingReservation.objects') as mock_reservations:
            mock_reservations.filter.return_value = mock_reservations
            mock_reservations.aggregate.return_value = {'total': 1000, 'avg': 50}
            mock_reservations.count.return_value = 20
            mock_reservations.values.return_value = mock_reservations
            mock_reservations.annotate.return_value = mock_reservations
            mock_reservations.order_by.return_value = []
            
            response = self.client.get('/revenue/')
            self.assertEqual(response.status_code, 200)

    def test_revenue_chart_api(self):
        """Test revenue chart API"""
        self.client.login(username='testowner@test.com', password='testpass123')
        
        with patch('User.models.ParkingReservation.objects') as mock_reservations:
            mock_reservations.filter.return_value = mock_reservations
            mock_reservations.annotate.return_value = mock_reservations
            mock_reservations.values.return_value = mock_reservations
            mock_reservations.order_by.return_value = [
                {'day': timezone.now().date(), 'total': 100.0}
            ]
            
            response = self.client.get('/api/revenue-chart/?range=7d')
            self.assertEqual(response.status_code, 200)
            
            response_data = json.loads(response.content)
            self.assertIn('labels', response_data)
            self.assertIn('data', response_data)

    # Contact and General Views Tests
    def test_submit_contact_query_success(self):
        """Test successful contact query submission"""
        with patch('slotifyBE.views.EmailMultiAlternatives') as mock_email:
            mock_email_instance = MagicMock()
            mock_email.return_value = mock_email_instance
            
            response = self.client.post('/submitContactQuery/',
                json.dumps({
                    'name': 'Test User',
                    'email': 'test@example.com',
                    'phone': '1234567890',
                    'query': 'Test query message'
                }),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.content)
            self.assertEqual(response_data['message'], 'Query submitted and email sent successfully.')
            
            # Check database
            self.assertTrue(ContactQuery.objects.filter(email='test@example.com').exists())

    def test_user_summary_stats(self):
        """Test user summary statistics"""
        response = self.client.get('/user-summary/')
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertIn('total_active_users', response_data)
        self.assertIn('total_active_owners', response_data)

    def test_get_landing_page_stats(self):
        """Test landing page statistics"""
        response = self.client.get('/getLandingPageStats/')
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertIn('owners_count', response_data)
        self.assertIn('parking_lot_count', response_data)
        self.assertEqual(response_data['owners_count'], 1)
        self.assertEqual(response_data['parking_lot_count'], 1)

    # Template View Tests
    def test_landing_page(self):
        """Test landing page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'landing.html')

    def test_user_register_page(self):
        """Test user registration page loads"""
        response = self.client.get('/userRegister/')
        self.assertEqual(response.status_code, 200)

    def test_user_signin_page(self):
        """Test user sign-in page loads"""
        response = self.client.get('/userSignIn/')
        self.assertEqual(response.status_code, 200)

    def test_about_page(self):
        """Test about page loads"""
        response = self.client.get('/about/')
        self.assertEqual(response.status_code, 200)

    def test_features_page(self):
        """Test features page loads"""
        response = self.client.get('/features/')
        self.assertEqual(response.status_code, 200)

    def test_contact_page(self):
        """Test contact page loads"""
        response = self.client.get('/contact/')
        self.assertEqual(response.status_code, 200)

    def test_terms_of_service_page(self):
        """Test terms of service page loads"""
        response = self.client.get('/terms-of-service/')
        self.assertEqual(response.status_code, 200)

    # Partial View Tests (HTMX/AJAX endpoints)
    def test_overview_partial_authenticated(self):
        """Test overview partial view"""
        self.client.login(username='testowner@test.com', password='testpass123')
        response = self.client.get('/partials/overview/')
        self.assertEqual(response.status_code, 200)

    def test_profile_partial_authenticated(self):
        """Test profile partial view"""
        self.client.login(username='testowner@test.com', password='testpass123')
        response = self.client.get('/partials/profile/')
        self.assertEqual(response.status_code, 200)

    # Password Reset Flow Tests
    def test_forgot_password_get(self):
        """Test forgot password page load"""
        response = self.client.get('/forgot-password/')
        self.assertEqual(response.status_code, 200)

    def test_forgot_password_post_success(self):
        """Test forgot password OTP sending"""
        with patch('slotifyBE.views.send_verification_email') as mock_email:
            response = self.client.post('/forgot-password/',
                json.dumps({'email': 'testowner@test.com'}),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.content)
            self.assertIn('redirect_url', response_data)
            mock_email.assert_called_once()

    def test_update_password_success(self):
        """Test password update in reset flow"""
        response = self.client.post('/update-password/',
            json.dumps({
                'email': 'testowner@test.com',
                'password': 'newpassword123'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['message'], 'Password updated.')

    # Error Handling Tests
    def test_view_requires_login_redirect(self):
        """Test that protected views require login"""
        response = self.client.get('/ownerDashboard/')
        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in POST requests"""
        response = self.client.post('/loginOwner/',
            'invalid json content',
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'Invalid JSON')

    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        response = self.client.post('/registerOwner/', {
            'firstName': 'Test',
            # Missing other required fields
        })
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'All fields are required.')

    # Method Not Allowed Tests
    def test_get_on_post_only_endpoint(self):
        """Test GET request on POST-only endpoints"""
        response = self.client.get('/loginOwner/')
        self.assertEqual(response.status_code, 405)
        
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'Invalid HTTP method. Only POST is allowed.')


# Additional Test Class for Integration Tests
class IntegrationTestCase(TestCase):
    """Integration tests for complete user workflows"""
    
    def setUp(self):
        self.client = Client()
    
    def test_complete_registration_flow(self):
        """Test complete user registration and verification flow"""
        # Step 1: Register owner
        with patch('slotifyBE.views.send_verification_email') as mock_email:
            response = self.client.post('/registerOwner/', {
                'firstName': 'Integration',
                'lastName': 'Test',
                'emailId': 'integration@test.com',
                'password': 'password123',
                'contactNumber': '5555555555'
            })
            
            self.assertEqual(response.status_code, 200)
            mock_email.assert_called_once()
        
        # Step 2: Verify OTP
        cache.set('otp_integration@test.com', '123456', timeout=300)
        
        response = self.client.post('/verify-otp/',
            json.dumps({
                'email': 'integration@test.com',
                'otp': '123456'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Step 3: Access dashboard
        user = User.objects.get(username='integration@test.com')
        self.client.login(username='integration@test.com', password='password123')
        
        response = self.client.get('/ownerDashboard/')
        self.assertEqual(response.status_code, 200)
    
    def test_login_and_parking_lot_registration_flow(self):
        """Test complete parking lot registration workflow"""
        # Create and login user
        user = User.objects.create_user(
            username='workflow@test.com',
            password='testpass123'
        )
        OwnerProfile.objects.create(
            user=user,
            firstName='Workflow',
            lastName='Test',
            emailId='workflow@test.com',
            password='hashed',
            contactNumber='7777777777',
            verified=True
        )
        
        # Login
        response = self.client.post('/loginOwner/',
            json.dumps({
                'email': 'workflow@test.com',
                'password': 'testpass123'
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        # Register parking lot
        with patch('slotifyBE.views.send_mail') as mock_mail:
            response = self.client.post('/submitParking/',
                json.dumps({
                    'name': 'Workflow Parking',
                    'location': '999 Workflow Street'
                }),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            mock_mail.assert_called_once()
        
        # Verify parking lot created
        parking_lot = ParkingLot.objects.get(location='999 Workflow Street')
        self.assertEqual(parking_lot.name, 'Workflow Parking')
        self.assertEqual(parking_lot.registered_by, user)
        self.assertFalse(parking_lot.confirmed)  # Should start as unconfirmed