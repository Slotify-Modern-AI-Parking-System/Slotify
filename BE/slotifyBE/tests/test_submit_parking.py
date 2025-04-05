from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from slotifyBE.models import OwnerProfile
from unittest.mock import patch

class SubmitParkingTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='test@example.com', password='testpass')
        self.profile = OwnerProfile.objects.create(
            user=self.user,
            firstName='Test',
            lastName='User',
            emailId='test@example.com',
            password='hashed',
            contactNumber='1234567890',
            verified=True
        )
        self.client.login(username='test@example.com', password='testpass')

    @patch('slotifyBE.views.send_mail')
    def test_submit_parking_address_sends_email(self, mock_send_mail):
        response = self.client.post('/submitParking/', content_type='application/json', data='{"location": "123 Fake St"}')
        self.assertEqual(response.status_code, 200)
        mock_send_mail.assert_called_once()
