from django.test import TestCase
from django.test import TestCase, Client
from django.contrib.auth.models import User

class ParkingPageLayoutTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='layouttest@example.com', password='testpass')
        self.client.login(username='layouttest@example.com', password='testpass')

    def test_parking_register_page_loads(self):
        response = self.client.get('/registerParking/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Register Parking")
        self.assertContains(response, "Parking Lot Address")

    def test_layout_contains_location_input(self):
        response = self.client.get('/registerParking/')
        self.assertIn(b'id="location"', response.content)