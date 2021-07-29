from .textpreprocessing import textpreprocessing,banglaCleanDataset
from keras.preprocessing.sequence import pad_sequences
from keras_self_attention import SeqSelfAttention
from keras.models import load_model
from sklearn.externals import joblib
import numpy as np
from fake_news_app.library.extra_functions import HierarchicalAttentionNetwork,recall_m,precision_m,f1_m,loadTokenizer,tf_idf,tokenizedSequences
from keras import backend as K
import pickle
import os
import time

dict_metric_accuracy = {
'McIntire_MLP': '90.77',
'McIntire_CNN': '92.74',
'McIntire_LSTM': '90.77',
'McIntire_NURG': '94.87',
'McIntire_dEXPLAIN': '92.98',
'McIntire_Our': '93.29',

'Liar_MLP': '61.14',
'Liar_CNN': '59.12',
'Liar_LSTM': '63.16',
'Liar_NURG': '57.85',
'Liar_dEXPLAIN': '61.04',
'Liar_Our': '64.01',

'Twitter_MLP': '98.90',
'Twitter_CNN': '99.44',
'Twitter_LSTM': '99.42',
'Twitter_NURG': '99.51',
'Twitter_dEXPLAIN': '99.27',
'Twitter_Our': '99.33',

'Combined_MLP': '93.20',
'Combined_CNN': '95.12',
'Combined_LSTM': '94.48',
'Combined_NURG': '95.12',
'Combined_dEXPLAIN': '95.00',
'Combined_Our': '95.24',

'BanFakeNews_MLP': '90.58',
'BanFakeNews_CNN': '94.33',
'BanFakeNews_LSTM': '95.77',
'BanFakeNews_NURG': '96.06',
'BanFakeNews_dEXPLAIN': '94.33',
'BanFakeNews_Our': '96.63',
}
dict_metric_confidence = {
'McIntire_MLP': '90.05-91.48',
'McIntire_CNN': '92.10-93.38',
'McIntire_LSTM': '90.05-91.48',
'McIntire_NURG': '94.33-95.41',
'McIntire_dEXPLAIN': '92.35-93.60',
'McIntire_Our': '92.68-93.91',

'Liar_MLP': '60.19-62.09',
'Liar_CNN': '58.17-60.08',
'Liar_LSTM': '62.22-64.10',
'Liar_NURG': '56.89-58.81',
'Liar_dEXPLAIN': '60.09-61.99',
'Liar_Our': '64.39-65.19',

'Twitter_MLP': '98.84-98.96',
'Twitter_CNN': '99.40-99.49',
'Twitter_LSTM': '99.31-99.47',
'Twitter_NURG': '99.47-99.55',
'Twitter_dEXPLAIN': '99.21-99.32',
'Twitter_Our': '99.27-99.38',

'Combined_MLP': '93.10-94.34',
'Combined_CNN': '94.91-95.75',
'Combined_LSTM': '94.35-94.61',
'Combined_NURG': '94.99-95.24',
'Combined_dEXPLAIN': '94.87-95.12',
'Combined_Our': '94.90-95.36',

'BanFakeNews_MLP': '89.78-91.37',
'BanFakeNews_CNN': '93.70-94.96',
'BanFakeNews_LSTM': '95.22-96.32',
'BanFakeNews_NURG': '95.53-96.59',
'BanFakeNews_dEXPLAIN': '93.70-94.96',
'BanFakeNews_Our': '96.14-97.12',
}

loading_dict = {"HierarchicalAttentionNetwork":HierarchicalAttentionNetwork,"f1_m":f1_m,"precision_m":precision_m,"recall_m":recall_m}


def classify_data(dataset,news):
    if(dataset=="BanFakeNews"):
        input_sentence = banglaCleanDataset(news)
    else:
        input_sentence = textpreprocessing(news)

    print(input_sentence)
    print("input end")



    modelSet = ["MLP","CNN","LSTM","NURG","dEXPLAIN","Our"]

    cur_path = os.path.dirname(__file__)
    savedModel = 0

    Result = []


    for model in modelSet:
        print(model)
        modelFilename = "models/" + model + dataset
        modelPath = cur_path + "/" + modelFilename

        savedModel = load_model(modelPath, custom_objects=loading_dict,compile=False)
        savedModel.summary()

        tokenizerFileName = "vectorizer/" + dataset
        tokenizerPath = cur_path + "/" + tokenizerFileName

        tfidf,seqTokenizer = loadTokenizer(tokenizerPath)

        startTime = time.time()


        if(model=="MLP"):
            inputTfidf = tf_idf([input_sentence], tfidf)
            score = savedModel.predict(inputTfidf)[0]
        elif(model=="dEXPLAIN"):
            inputSeq = tokenizedSequences([input_sentence], seqTokenizer)

            time_steps = 1
            n_inputs = 5000
            inputSeq = np.array(inputSeq).reshape((-1, time_steps, n_inputs))

            score = savedModel.predict(inputSeq)[0]
        elif(model=="Our"):
            inputTfidf = tf_idf([input_sentence], tfidf)
            inputSeq = tokenizedSequences([input_sentence], seqTokenizer)
            score = savedModel.predict([inputTfidf,inputSeq])[0]
        else:
            inputSeq = tokenizedSequences([input_sentence], seqTokenizer)
            score = savedModel.predict(inputSeq)[0]


        endTime = time.time()
        #print(score)

        classification = None
        if(score>=0.5):
            classification = "FAKE"
        else:
            classification = "REAL"

        # Result[model]=[classification,endTime-startTime]
        print(model,"done!!")
        res = {
            'model': model,
            'result': classification,
            'time': endTime-startTime,
            'accuracy': dict_metric_accuracy[dataset+'_'+model],
            'confidence': dict_metric_confidence[dataset+'_'+model]
        }
        Result.append(res)

        K.clear_session()

    print(Result)

    return Result



# r = classify_data("mcintire", "Trump is a stupid piece of shit. Not good president!!")
# print(r)