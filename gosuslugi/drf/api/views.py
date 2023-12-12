from django.shortcuts import render
from models import User
from serializer import UserSerializer
from rest_framework.viewsets import ModelViewSet

class get_users(ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer