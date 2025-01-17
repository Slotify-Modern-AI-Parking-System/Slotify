from django.shortcuts import render

# Create your views here.
def userRegister(request):
    return render(request, "userRegister.html")