from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from slotifyBE.models import OwnerProfile
from django.urls import reverse

class OwnerRegistrationTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.test_email = "testuser@example.com"
        self.test_contact = "1234567890"

    def test_register_owner_success(self):
        image = SimpleUploadedFile("test.jpg", b"file_content", content_type="image/jpeg")

        response = self.client.post('/registerOwner/', {
            'firstName': 'Test',
            'lastName': 'User',
            'emailId': self.test_email,
            'password': 'password123',
            'contactNumber': self.test_contact,
            'idProof': image,
        })

        self.assertEqual(response.status_code, 201)
        self.assertTrue(User.objects.filter(username=self.test_email).exists())
        self.assertTrue(OwnerProfile.objects.filter(contactNumber=self.test_contact).exists())

    def test_register_owner_duplicate_email(self):
        # First registration
        User.objects.create_user(username=self.test_email, password='password123')
        
        image = SimpleUploadedFile("test.jpg", b"file_content", content_type="image/jpeg")
        response = self.client.post('/registerOwner/', {
            'firstName': 'Test',
            'lastName': 'User',
            'emailId': self.test_email,
            'password': 'password123',
            'contactNumber': '9999999999',
            'idProof': image,
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Email ID already exists', response.content)
