from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .library.model_library import classify_data
from .library.news_api import get_news_for_dashboard


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def home(request):
    context = {}
    context['result'] = []
    return render(request, 'fake_news_app/home.html', context)


def test_page(request):
    context = {}
    context['result'] = []
    return render(request, 'fake_news_app/test_page.html', context)


def english_news_classifier(request):
    context = {}
    context['result'] = []
    return render(request, 'fake_news_app/english_news_classifier.html', context)


def bangla_news_classifier(request):
    context = {}
    context['result'] = []
    return render(request, 'fake_news_app/bangla_news_classifier.html', context)


def classify_news(request):
    # model = request.GET.get('model')
    dataset = request.GET.get('dataset')
    news = request.GET.get('news')

    print(news)
    print("\n\n\n")

    try:
        result = classify_data(dataset, news)
    except Exception as e:
        print(e)
        result = []

    # data = {
    #     'result': 'REAL' if result else 'FAKE',
    #     'accuracy': dict_metric_accuracy[dataset+'_'+model],
    #     'confidence': dict_metric_confidence[dataset + '_' + model]
    #     }

    return JsonResponse(result, safe=False)


def get_news(request):
    data = get_news_for_dashboard()

    return JsonResponse(data, safe=False)


def about(request):
    context = {}
    context['result'] = []
    return render(request, 'fake_news_app/about.html', context)


def contact(request):
    context = {}
    context['result'] = []
    return render(request, 'fake_news_app/contact.html', context)