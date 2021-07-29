from newsapi import NewsApiClient

def get_news_for_dashboard():
    newsapi = NewsApiClient(api_key='9536ddc288774f5f9d473fceff05d46a')
    all_articles = newsapi.get_everything(q='bangladesh',
                                          language='en')

    data = all_articles["articles"]


    ret = {}
    for i in range(0,8):
        temp = {}
        temp["source"] = data[i]["source"]["name"]
        temp["title"] = data[i]["title"]
        temp["description"] = data[i]["description"]
        temp["url"] = data[i]["url"]
        temp["image"] = data[i]["urlToImage"]
        ret[str(i)]=temp

    return ret