import io
from django.shortcuts import redirect, render
from django.http import HttpResponse
import json,os,base64,urllib

from numpy import array

from .models import Greeting
from django.views.decorators.csrf import csrf_protect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import User, Data_ThuatToan, Upload_File
from hello.static.index import Select_Model 
from matplotlib import pyplot as plt


import os
# Create your views here.se


def index(request):
    if('username' in request.session):
        return render(request, 'home.html')
    else:
        return redirect('login')


def test(request):

    return render(request, "testpage.html")


def svm_imoc(request):

    return render(request, "svm_imoc.html")


def login(request):
    if request.method == 'POST':
        username = request.POST["Username"]
        password = request.POST["Password"]
        admin = {
            'username': username,
            'password': password
        }
        request.session['username'] = username
        request.session['password'] = password
        return redirect('home')
    else:
        return render(request, 'login.html', {'users': User.objects.all()})


def logout(request):
    request.session.flush()
    return redirect('/')


def auth_login(request):
    # if(request.method == 'POST'):
    #     print(request.POST)
    # else:
    return render(request, "login.html")


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'uploadFile/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'pages/simple_upload.html')


def process_test(request):
    if request.method == 'POST':
        # print()
        myfile = request.FILES['file-csv-open']
        fs = FileSystemStorage(settings.MEDIA_ROOT+'/media/')
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        # print(uploaded_file_url)
        array_nu, array_gamma, array_acc, x_ve = Select_Model(
            settings.MEDIA_ROOT+uploaded_file_url, settings.MEDIA_ROOT, int(request.POST['number-cut']), int(request.POST['number-age']), request.POST['select'])
        # plt.figure(figsize=(7,5))
        plt.plot(x_ve, array_acc, marker='o', linestyle='dashed',
                 label=request.POST['select'])
        # plt.plot(array_acc, x_ve,label='and')
        plt.ylabel('Accuracy')
        plt.title(request.POST['select'])
        plt.legend()
        fig=plt.gcf()
        buf=io.BytesIO()
        fig.savefig(buf,format='png')
        buf.seek(0)
        string=base64.b64encode(buf.read())
        plt.close()
        uri=urllib.parse.quote(string)
        array=[]
        for i in range(len(x_ve)):
            array.append({"accuracy":array_acc[i],"numbercut":x_ve[i]})
        return render(request, 'testpage.html', {'img': uri,"array":array})
        # return render(request, 'home.html', {
        #     'uploaded_file_url': uploaded_file_url
        # })
    else:
        return redirect('home')
