from django.db import models


# Create your models here.
class Greeting(models.Model):
    when = models.DateTimeField("date created", auto_now_add=True)


class Upload_File(models.Model):
    Path = models.CharField(max_length=200)
    NameFile = models.CharField(max_length=100)
    Date_Up = models.DateTimeField(auto_now_add=True)
# class ThuatToan_Data(models.Model):
#     TenThuatToan=models.CharField(max_length=50, blank=False)
#     ID_ThuatToan=models.CharField(max_length=50, blank=False)
#     Nu=models.FloatField(blank=False)
#     Gamma=models.FloatField(blank=False)
#     Acc=models.FloatField(blank=False)
#     Path_file=models.ForeignKey(Upload_File,on_delete=models.CASCADE)
class Data_ThuatToan(models.Model):
    TenThuatToan = models.CharField(max_length=50, blank=False)
    ID_ThuatToan = models.CharField(max_length=50, blank=False)
    Nu = models.FloatField(blank=False)
    Gamma = models.FloatField(blank=False)
    Acc = models.FloatField(blank=False)
    Path_file = models.ForeignKey(Upload_File, on_delete=models.CASCADE)
class User(models.Model):
    username=models.CharField(max_length=50,blank=False)
    password=models.CharField(max_length=50,blank=False)
    name=models.CharField(max_length=50,blank=True)
    birthday = models.DateField(null=True, blank=True, auto_now_add=True)
    images=models.ImageField(blank=True)

