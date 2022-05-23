# SeekTruth.py : Classify text objects into two categories
#
# Code by: Harsh Srivastava <hsrivas>
#        : Ritwik Budhiraja <rbudhira>
#        : Yash Kalpesh Shah <yashah>
#
# Based on skeleton code by D. Crandall, October 2021
#

import re
import sys
import math


def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier(train_data, test_data,stop_words_list):

    smoothing_factor = 1
    final_test_labels = []
    class1_dict = {}
    class2_dict = {}
    class1_words = []
    class2_words = []
    len_class1 = 0
    len_class2 = 0

    # Made the code dynamic by making use of classes.
    class1 = train_data["classes"][0]
    class2 = train_data["classes"][1]

    # Calculated the length of Class1 and Class2 entries.
    for vals in train_data["labels"]:
        if vals == class1:
            len_class1 += 1
        else:
            len_class2 += 1

    # Calculated prior probabilities.
    prob_class1 = len_class1/len(train_data["objects"])
    prob_class2 = len_class2/len(train_data["objects"])

    # Made a list of all the words belonging to Class1.
    class1_objects = [train_data["objects"][x] for x in filter(lambda x: train_data["labels"][x] == class1, range(len(train_data["labels"])))]
    for sentence in class1_objects:
        # Used regex to split sentences into words on the basis of non-alphanumeric characters.
        # Converted sentences into lowercase to avoid any discrepancies.
        class1_words = class1_words + re.split(r"[\b\W\b]+", sentence.lower())

    # Made a list of all the words belonging to Class2.
    class2_objects = [train_data["objects"][x] for x in filter(lambda x: train_data["labels"][x] == class2, range(len(train_data["labels"])))]    
    for sentence in class2_objects:
        class2_words = class2_words + re.split(r"[\b\W\b]+", sentence.lower())

    # Removed the stop/neutral words from Class1 words and Class2 words.
    class1_words = [item for item in class1_words if item not in stop_words_list]
    class2_words = [item for item in class2_words if item not in stop_words_list]

    # Created a dictionary of all the unique Class1 words and storing their likelihood probabilities.
    for word in class1_words:
        if word not in class1_dict:
            # Introduced a smoothing factor in the formula to take care of missing words.
            # https://towardsdatascience.com/laplace-smoothing-in-naïve-bayes-algorithm-9c237a8bdece --- Implemented Laplace smoothing.
            class1_dict[word] = (class1_words.count(word) + smoothing_factor)/(len(class1_words) + (smoothing_factor * len(train_data["classes"])))
   
    # Created a dictionary of all the unique Class2 words and storing their likelihood probabilities.
    for word in class2_words:
        if word not in class2_dict:
            class2_dict[word] = (class2_words.count(word) + smoothing_factor)/(len(class2_words) + (smoothing_factor * len(train_data["classes"])))



    # This is where we start working on the test file. 
    for sentence in test_data["objects"]:
        
        # Took log sum instead of product to improve the calculation and avoid making the operands smaller.
        # Initialized the log sum of the classes to the prior probabilities of the classes.
        log_sum_class1 = math.log(prob_class1)
        log_sum_class2 = math.log(prob_class2)
        words = re.split(r"[\b\W\b]+", sentence.lower())
        words = [item for item in words if item not in stop_words_list]
        for word in words:
            if word in class1_dict:
                log_sum_class1 += math.log(class1_dict[word])
            else:
                # Took care of the missing words here.
                log_sum_class1 += math.log(smoothing_factor/(len(class1_words) + (smoothing_factor * len(train_data["classes"]))))
            if word in class2_dict:
                log_sum_class2 += math.log(class2_dict[word])
            else:
                log_sum_class2 += math.log(smoothing_factor/(len(class2_words) + (smoothing_factor * len(train_data["classes"]))))
        final_test_labels.append(class1 if log_sum_class1 > log_sum_class2 else class2)


    return final_test_labels


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv


    # Imported a text file of the stop/neutral words.
    # https://github.com/igorbrigadir/stopwords/blob/master/en/terrier.txt --- This is where we got the text file from.
    with open('stop_words2.txt') as f:
        contents = f.read()
    stop_words_list = contents.replace("\n"," ")
    stop_words_list = stop_words_list.split()


    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)

    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized,stop_words_list)

# calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
