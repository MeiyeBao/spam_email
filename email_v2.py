# importing system libraries

from os import walk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pandas as pd
from sklearn import metrics


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


# data train-test split
def split_dataset(data):
    z = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2)

    return z_train, z_test, y_train, y_test


# Converting String to Integer
def string_to_int(training_set):
    cv1 = CountVectorizer()
    features1 = cv1.fit_transform(training_set)

    return cv1, features1


def build_model(features1, y_train1):
    model1 = svm.SVC()

    # Train the model using the training sets
    model1.fit(features1, y_train1)

    # Predict the response for test dataset
    y_pred = model1.predict(features1)

    return model1, y_pred


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


def spam_or_ham_data(path):
    pathwalk = walk(path)
    newdata = []
    for root, dr, file in pathwalk:
        for obj in file:
            with open(root + '/' + obj, encoding='latin1') as ip:
                newdata.append(" ".join(ip.readlines()))

    print("Check new data: ", type(newdata))

    raw_df = pd.DataFrame({"email": newdata})

    return raw_df


if __name__ == "__main__":
    unprocessed_data = preview_datasets()

    email_train, email_test, label_train, label_test = split_dataset(unprocessed_data)
    cv,train_int = string_to_int(email_train)
    model, y_pred = build_model(train_int, label_train)
    features_test = cv.transform(email_test)
    print("y pred:",y_pred)
    print("Accuracy for test dataset: {}".format(model.score(features_test, label_test)))

    # predict for new datasets
    new_data_final = spam_or_ham_data('BG')
    features_new = cv.transform(new_data_final)

    print(model)
    # make predictions on test set
    label_pred = model.predict(features_new)

    print("Spam or Ham:",label_pred)








