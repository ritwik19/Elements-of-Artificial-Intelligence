#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors:
# - Harsh Srivastava <hsrivas>
# - Ritwik Budhiraja <rbudhira>
# - Yash Shah <yashah>
#
# (based on skeleton code by D. Crandall, Oct 2020)
#

import os
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
from numpy.core.defchararray import replace
from numpy.lib.arraypad import _view_roi
import math
import json

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work.
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[2] ]))

def letter_to_boolean_matrix(letter):
    letter = [[c for c in x] for x in letter]
    letter = np.asarray(letter)
    letter[letter == '*'] = 1
    letter[letter == ' '] = 0
    return letter.astype('int')

def boolean_matrix_to_letter(letter):
    letter = np.asarray(letter)
    letter = letter.astype('str')
    letter[letter == 'True'] = '*'
    letter[letter == 'False'] = ' '
    return ["".join(x) for x in letter]

def print_slice(slice):
    for i in range(25):
        for j in range(14):
            if not slice[i][j]:
                print("\033[44;37m:\033[0m", end="")
            else:
                print("\033[41;37m:\033[0m", end="")
        print()

class ReadingText:
    LOWEST_PROBABILITY = math.pow(10, -10)

    def __init__(self, training_letters, test_letters) -> None:
        self.training_letters = training_letters
        self.training_letters_list = list(self.training_letters.keys())
        self.test_letters = test_letters
        self.letter_index = { self.training_letters_list[i]: i for i in range(len(self.training_letters)) }
        self.initial_probability = np.array([ 0.0 for x in self.training_letters_list ])

        self.ll_transition = {}

    def get_training_index(self, letter):
        return self.letter_index[letter]

    def simple_calculate_match_props(self, source, target):
        mask_xor = np.bitwise_xor(source, target)
        count_non_match_pixels = np.sum(mask_xor)
        return 1 - count_non_match_pixels / (mask_xor.shape[0] * mask_xor.shape[1])

    def hmm_calculate_match_props(self, source, target):
        match, source_miss, target_miss = 0, 0, 0
        for r in range(25):
            for c in range(14):
                match += source[r][c] == target[r][c]
                source_miss += source[r][c] != target[r][c] and source[r][c] == 1
                target_miss += target[r][c] != source[r][c] and target[r][c] == 1
        match, source_miss, target_miss = match / 350, source_miss / 350, target_miss / 350
        # print("(2.0 - 0.4 + 0.8):", (match * 2.0 - source_miss * 0.4 + target_miss * 0.8) / (2.0 - 0.4 + 0.8))
        # print("(2.0 - 0.4 + 0.8):", (match * 2.0 - source_miss * 0.8 + target_miss * 0.4) / (2.0 - 0.8 + 0.4))
        # print("(2.0 + 0.4 - 0.8):", (match * 2.0 + source_miss * 0.4 - target_miss * 0.8) / (2.0 + 0.4 - 0.8))
        # print("(2.0 + 0.8 - 0.4):", (match * 2.0 + source_miss * 0.8 - target_miss * 0.4) / (2.0 + 0.8 - 0.4))
        return (match * 2.0 - source_miss * 0.4 + target_miss * 0.8) / (2.0 - 0.4 + 0.8)

    def calculate_match_table(self, match_method):
        self.mr_emission = {}

        for test_letter_index in range(len(self.test_letters)):
            test_letter_image = self.test_letters[test_letter_index]
            test_letter_image_bool = letter_to_boolean_matrix(test_letter_image)
            training_letters = list(self.training_letters.items())

            self.mr_emission[test_letter_index] = np.zeros(len(training_letters))

            option = ""
            for t_i in range(len(training_letters)):
                training_letter, training_letter_image = training_letters[t_i]
                training_letter_image_bool = letter_to_boolean_matrix(training_letter_image)

                match_props = match_method(training_letter_image_bool, test_letter_image_bool)
                self.mr_emission[test_letter_index][t_i] = match_props

                # if option != "a":
                #     print("Test Letter:")
                #     print_slice(test_letter_image_bool)
                #     print("Training Letter:")
                #     print_slice(training_letter_image_bool)
                #     option = input()

            # self.mr_emission[test_letter_index] /= np.sum(self.mr_emission[test_letter_index])

    def calculate_transition_probabilities(self, text_train_file):
        # Initialise transition matrix for basic training letters
        self.ll_transition = np.zeros((len(self.training_letters), len(self.training_letters)))

        # Load dumped training data if already present
        if os.path.exists(text_train_file + ".model.cache"):
            raw = json.loads(open(text_train_file + ".model.cache", "r").read())
            self.initial_probability = np.array(raw["initial"])
            self.ll_transition = np.array(raw["transition"])

        # In case the file is freshly read, process it
        #   - This usually takes a while!
        else:
            # Open the file handle
            with open(text_train_file, 'r') as words_file_handle:
                # Process the file line by line
                lines = words_file_handle.readlines()

                # Iterate through each line
                for line in lines:
                    line = line.strip()
                    line = line.replace("`", "'")
                    line = line.replace("''", "\"")

                    # In case line is zero length, skip it
                    if len(line) == 0:
                        continue

                    # Get the words/tokens
                    line = " ".join(line.split(' ')[::2]).strip()
                    # for c in "),.-!?\"' ":
                    #     line = line.replace(" " + c, c)

                    # Remove characters not in training set
                    line = "".join([c for c in line if c in self.training_letters_list])
                    line = line.replace("  ", " ")
                    line = line.replace(" .", ".")
                    line = line.replace(" ,", ",")
                    line = line.replace(" ;", ";")
                    line = line.lower()

                    # Build the initial probability for start of sentences
                    if line[0][0] in self.letter_index:
                        self.initial_probability[self.letter_index[line[0][0]]] += 1

                    # Now process each word, by each letter pair to determine the transition probability
                    # for word in words:
                    #     word = word.strip().replace("``", "\"")
                    #     # word = word.strip().lower().replace("``", "\"")
                    for i in range(len(line) - 1):
                        if line[i] in self.training_letters and line[i + 1] in self.training_letters:
                            # In case the character is a letter, which is really likely,
                            # we add all sets of upper/lower-case letter pairs, to make sure
                            # the probabilities aren't skewed for words which only differ in
                            # their capitalisation, like - 'the' or 'The'
                            # for word_i0 in [word[i], word[i].upper()]:
                            #     for word_i1 in [word[i + 1], word[i + 1].upper()]:
                            self.ll_transition[self.get_training_index(line[i])][self.get_training_index(line[i + 1])] += 1
                self.initial_probability /= len(lines)

            # Divide the counts to get the 'probability' for each transition state
            for training_letter_index in range(len(self.training_letters)):
                sum = np.sum(self.ll_transition[training_letter_index])
                if sum != 0:
                    self.ll_transition[training_letter_index] /= sum

            # Dump the loaded probabilities with the source file name
            with open(text_train_file + ".model.cache", "w") as train_transition_handle:
                train_transition_handle.write(json.dumps({
                    "initial": np.ndarray.tolist(self.initial_probability),
                    "transition": np.ndarray.tolist(self.ll_transition)
                }))

    def simple_bayes(self):
        self.calculate_match_table(self.simple_calculate_match_props)

        predicted_result = ""

        for test_letter_index in range(len(self.test_letters)):
            max_index = np.argmax(self.mr_emission[test_letter_index][:])
            predicted_result += self.training_letters_list[max_index]

        return predicted_result

    def hmm_map(self, text_train_file, predicted_simple_bayes):
        # This implementation is based on the material provided during the class activity for Viterbi algorithm
        # Ref: https://iu.instructure.com/courses/2027431/files/128919777?module_item_id=25300577

        # Re-calculate the emission probabilties
        self.calculate_match_table(self.hmm_calculate_match_props)

        # The viterbi table will be N * N for N letters
        v_table = np.zeros((len(self.training_letters), len(self.test_letters)))
        which_table = [["" for _ in range(len(self.test_letters))] for _ in range(self.ll_transition.shape[0])]

        for t_i in range(len(self.training_letters)):
            v_table[t_i][0] = self.mr_emission[0][t_i] #* self.initial_probability[t_i]

        i = 0
        for i in range(1, len(self.test_letters)):
            for t_i in range(len(self.training_letters)):
                t_i2 = t_i if t_i > 25 else (t_i - 26)
                (which_table[t_i][i], v_table[t_i][i]) = max([
                    (self.training_letters_list[s0], v_table[s0][i - 1] * self.ll_transition[t_i2][s0])
                        for s0 in range(len(self.training_letters))], key=lambda l:l[1])
                v_table[t_i][i] *= self.mr_emission[i][t_i]

        viterbi_sequence = [""] * len(predicted_simple_bayes)
        viterbi_sequence[len(predicted_simple_bayes) - 1] = self.training_letters_list[np.argmax([v_table[tag_i][i] for tag_i in range(len(self.training_letters_list))])]

        for i in range(len(predicted_simple_bayes) - 2, -1, -1):
            viterbi_sequence[i] = which_table[self.letter_index[viterbi_sequence[i + 1]]][i + 1]

        return "".join(viterbi_sequence)


# The final two lines of your output should look something like this:
# print("Simple: " + "Sample s1mple resu1t")
# print("   HMM: " + "Sample simple result")
model = ReadingText(train_letters, test_letters)
predicted_simple_bayes = model.simple_bayes()
print("Using Simple Bayes: '" + predicted_simple_bayes + "'")
model.calculate_transition_probabilities(train_txt_fname)
print("Using HMM MAP: '" + model.hmm_map(train_txt_fname, predicted_simple_bayes) + "'")