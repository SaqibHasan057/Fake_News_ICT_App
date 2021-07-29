def textpreprocessing(sentence):
    ##Convert to lower case
    x = sentence.lower()

    ##Remove numbers
    import re
    x = re.sub(r'\d+','', x)

    ##Remove_hashtags_@
    x = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split())

    ##Remove punctuations
    import string
    translator = str.maketrans(string.punctuation+'—…', ' '*(len(string.punctuation)+2))
    x = x.translate(translator)

    ##Remove whitespaces
    x = " ".join(x.split())

    ##Remove stopwords
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # remove stopwords function
    def remove_stopwords(text):
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        filtered_text = " ".join(filtered_text)
        return filtered_text

    x = remove_stopwords(x)

    ##Lemmatization
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lemmatizer = WordNetLemmatizer()

    # lemmatize string
    def lemmatize_word(text):
        word_tokens = word_tokenize(text)
        # provide context i.e. part-of-speech
        lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
        lemmas = " ".join(lemmas)
        return lemmas

    x = lemmatize_word(x)

    print("Text preprocessing done!!")
    return x


def testBanglaNumber(word):
    l = ['১','২','৩','৪','৫','৬','৭','৮','৯','০']
    for i in l:
        if i in word:
            return True

    return False


def banglaCleanDataset(sentence):
    import string

    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation+"।")

    # remove punctuation from each token
    bengLine = sentence.translate(table)

    # tokenize on white space
    bengLine = bengLine.split()

    ##print("third ", bengLine)

    # remove tokens with numbers in them
    bengLine = [word for word in bengLine if not testBanglaNumber(word)]

    # remove non-Bangla words
    bengLine = [word for word in bengLine if not word.isalpha()]

    print("Bengali line: ", bengLine)
    # store as string
    x = ' '.join(bengLine)
    return x