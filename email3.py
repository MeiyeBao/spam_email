# importing system libraries

from os import walk
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from random import shuffle
import pandas as pd
import sklearn as sk


def get_base_emails(pathwalk1):

    allHamData1, allSpamData1 = [], []
    for root, dr, file in pathwalk1:
        if 'ham' in str(file):
            for obj in file:
                with open(root + '/' + obj, encoding='latin1') as ip:
                    allHamData1.append(" ".join(ip.readlines()))

        elif 'spam' in str(file):
            for obj in file:
                with open(root + '/' + obj, encoding='latin1') as ip:
                    allSpamData1.append(" ".join(ip.readlines()))
    return allHamData1, allSpamData1


def sort_into_dataframe(allHamData2, allSpamData2):
    # storing it in a dataframe

    hamPlusSpamData = allHamData2 + allSpamData2
    labels = ["ham"] * len(allHamData2) + ["spam"] * len(allSpamData2)

    raw_df = pd.DataFrame({"email": hamPlusSpamData,
                           "label": labels})
    return raw_df


def visualize_data_hist(data1):
    plt.hist(data1, bins=3)
    plt.savefig('raw_email_data')


def preview_datasets():
    pathwalk = walk("enron1/")
    allHamData, allSpamData = get_base_emails(pathwalk)
    # remove all redundant data
    allHamData = list(set(allHamData))
    allSpamData = list(set(allSpamData))

    rawdata = sort_into_dataframe(allHamData, allSpamData)
    print('raw data', rawdata.head())

    # get an overview of the data
    visualize_data_hist(rawdata.label)
    print('Dataset Preview:', rawdata.columns)

    return rawdata


def process_clean(data):
    # tokenization
    tokens = nltk.word_tokenize(data)
    tokens = [w.lower() for w in tokens if w.isalpha()]

    # finding uncommon words
    cnt = Counter(tokens)
    uncommons = cnt.most_common()[:-int(len(cnt) * 0.1):-1]

    # listing stopwords from NLTK
    stops = set(nltk.corpus.stopwords.words('english'))

    # removing stop words and uncommon words
    tokens = [w for w in tokens if (w not in stops and w not in uncommons)]

    # lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w, pos='a') for w in tokens]

    # pre-processing the emails
    # using word_tokenize() and WordNetLemmatizer()

    return tokens


def process_final(nltk_processed_df):
    email = nltk_processed_df.email
    label = nltk_processed_df.label
    X_featurized = [Counter(i) for i in email]
    allDataProcessed = [(X_featurized[i], label[i]) for i in range(len(email))]

    # randomizing using shuffle
    shuffle(allDataProcessed)

    # manually splitting into test and train data
    trainData, testData = allDataProcessed[:int(len(allDataProcessed) * 0.7)], allDataProcessed[
                                                                               int(len(allDataProcessed) * 0.7):]
    return trainData,testData


def naive_baye_clasi_model(nltk_processed_df1):
    train_set, test_set = process_final(nltk_processed_df1)
    model_nltkNaiveBayes = nltk.classify.NaiveBayesClassifier.train(train_set)
    print("NaiveBayesClassifier Training:", model_nltkNaiveBayes)
    model_nltkNaiveBayes.show_most_informative_features(15)
    testing_accuracy = nltk.classify.accuracy(model_nltkNaiveBayes, test_set)
    print("Accuracy with NLTK's Naive Bayes classifier is:", testing_accuracy)
    return model_nltkNaiveBayes


def spam_or_ham_data(path):
    pathwalk = walk(path)
    newdata = []
    for root, dr, file in pathwalk:
        for obj in file:
            with open(root + '/' + obj, encoding='latin1') as ip:
                newdata.append(" ".join(ip.readlines()))

    print("Check new data: ", type(newdata))

    labels = [''] * len(newdata)

    raw_df = pd.DataFrame({"email": newdata,
                           "label": labels})

    print("Check new data: ", raw_df.head())
    nltk_processed_new = pd.DataFrame()
    nltk_processed_new['email'] = [process_clean(e) for e in raw_df.iloc[:, 0]]
    print("New Data Processed:", nltk_processed_new.head(5))
    return nltk_processed_new


if __name__ == "__main__":
    unprocessed_data = preview_datasets()

    # pre-processing the emails
    # using word_tokenize() and WordNetLemmatizer()
    nltk_processed_df = pd.DataFrame()
    nltk_processed_df['email'] = [process_clean(e) for e in unprocessed_data.iloc[:, 0]]
    # label encoding the labels

    label_encoder = sk.preprocessing.LabelEncoder()
    nltk_processed_df['label'] = label_encoder.fit_transform(unprocessed_data.iloc[:, 1])

    # checking how the processed data looks like
    print("Processed Data", nltk_processed_df.head(5))
    print("Processed Data", type(nltk_processed_df))

    classifier = naive_baye_clasi_model(nltk_processed_df)

    new_data_final = spam_or_ham_data('BG/2004')
    #classifier.classify(new_data_final)










