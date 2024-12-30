from django.urls import path
from . import views
from django.conf import settings # new
from  django.conf.urls.static import static #new

urlpatterns = [
    path('', views.home, name="home"),
    path('login/', views.login, name="login"),
    path('signin/', views.signin, name="signin"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('patientdash/', views.patientdash, name="patientdash"),
    path('processing/', views.processing, name="processing"),
    path('uploads/', views.uploads, name="uploads"),
    path('patientpro', views.patientpro, name="patientpro"),
    path('patientupload', views.patientupload, name="patientupload"),
    path('settings/', views.settings, name="settings"),
    path('logout/', views.logout, name="logout"),
    path('profile/', views.profile, name="profile"),
    path('predicted/', views.predicted, name="predicted"),
    path('visualize/', views.visualize, name="visualize"),
    path('changepassword/', views.changepassword, name="changepassword"),
    path('predictionResults/', views.predictionResults, name="predictionResults"),
    path('ploadedFiles/', views.ploadedFiles, name="ploadedFiles"),
    path('team/', views.team, name="team"),
    path('viewPredictions/', views.viewPredictions, name="viewPredictions"),
    path('visualization/', views.visualization, name="visualization"),
    path('swiResult/', views.swiResult, name="swiResult"),
    path('pdResult/', views.pdResult, name="pdResult"),
    path('mraResult/', views.mraResult, name="mraResult"),
    path('t2Result/', views.t2Result, name="t2Result"),
    path('swiViewPredictions/', views.swiViewPredictions,
         name="swiViewPredictions"),
    path('pdViewPredictions/', views.pdViewPredictions, name="pdViewPredictions"),

    path('mraViewPredictions/', views.mraViewPredictions,
         name="mraViewPredictions"),

    path('t2ViewPredictions/', views.t2ViewPredictions,
         name="t2ViewPredictions"),
    path('start_processing/', views.start_processing, name="start_processing"),
    path('start_processingpd/', views.start_processingpd, name="start_processingpd"),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_URL)
