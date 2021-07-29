from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.test_page, name='test'),
    path('english-news-classifier/', views.english_news_classifier, name='english_news'),
    path('bangla-news-classifier/', views.bangla_news_classifier, name='bangla_news'),
    path('news-classifier/', views.classify_news, name='classifier'),
    path('about-us/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('index/', views.index, name='index'),
    path('newsapi/', views.get_news, name='newsapi'),
]