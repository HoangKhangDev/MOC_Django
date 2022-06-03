from django.urls import path, include
import hello.views
from django.conf.urls.static import static
from django.conf import settings



urlpatterns = [
    path("", hello.views.index, name='home'),
    path("test/", hello.views.test, name='test'),
    path("svm_imoc/", hello.views.svm_imoc, name='svm_imoc'),
    path("login/", hello.views.login, name='login'),
    path("logout/", hello.views.logout, name='logout'),
    path("process_test/", hello.views.process_test, name='process_test'),
]
