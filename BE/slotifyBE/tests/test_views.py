from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from slotifyBE.models import OwnerProfile
from django.urls import reverse

class OptionsPageTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_options_page_loads(self):
        response = self.client.get('/options/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Welcome to Slotify")