###################################
# CS B551 Spring 2021, Assignment #3
#
# Authors
# - Harsh Srivastava <hsrivas>
# - Ritwik Budhiraja <rbudhira>
# - Yash Shah <yashah>
#
# (Based on skeleton code by D. Crandall)
#

from numpy.core.fromnumeric import transpose
import math
import numpy as np

# Note: For this file we assume-
# Any, S = Tag
#      W = Word
class Solver:
    # Assign a really low probability for keys that are not found in the probability tables
    LOWEST_PROBABILITY = math.pow(10, -10)

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling.
    def posterior(self, model, sentence, label):

        # Blank variable (calculations are done in log mode: add and subtract only)
        product = 0.0

        # Simple model posterior calculation
        if model == "Simple":
            # For each word we have a simple emission probabliity that factors as-
            #     P(Si | Wi) = P(Si) * P(Wi | Si)
            #
            # P(Wi | Si) means we need a table from S to W mapping
            for i in range(len(sentence)):
                # Get the Si -> Wi probability => P(Wi | Si)
                P_emission_i = self.tw_emission[self.tag_index[label[i]]].get(sentence[i], Solver.LOWEST_PROBABILITY)

                # Get the P(Si)
                P_tag_i = self.tag_probability[self.tag_index[label[i]]]

                # Add the same to final product
                product += math.log(P_emission_i) + math.log(P_tag_i)

        # HMM model posterior calculation
        elif model == "HMM":
            # For HMM we have the following factorization of the probability
            #     P(Si | Wi) = P(Wi | Si) * P(Si)
            #                = P(Wi | Si) * P(Si | Si-1) * P(Si-1)

            # However, for the zero-th case we can simply take the P(Wi | Si) and P(Si) since there is no Si-1 for i = 0
            product += math.log(self.tw_emission[self.tag_index[label[0]]].get(sentence[0], Solver.LOWEST_PROBABILITY)) \
                     + math.log(self.tag_probability[self.tag_index[label[0]]])

            # Now iterate from i = 1 till the end and add all log probabilities
            for i in range(1, len(sentence)):
                # Get the P(Wi | Si) using this
                P_emission_i = self.tw_emission[self.tag_index[label[i]]].get(sentence[i], Solver.LOWEST_PROBABILITY)

                # Get the transition from Si-1 to Si
                P_transition_i = self.tt_transition[self.tag_index[label[i - 1]]][self.tag_index[label[i]]]

                # Sum the log probabilities
                product += math.log(P_transition_i) + math.log(P_emission_i) + math.log(self.tag_probability[self.tag_index[label[i - 1]]])
        elif model == "Complex":
            product += math.log(self.tw_emission[self.tag_index[label[0]]].get(sentence[0], Solver.LOWEST_PROBABILITY)) \
                     + math.log(self.tag_probability[self.tag_index[label[0]]])
            if len(sentence) > 1:
                product += math.log(self.word_tag_pair_table.get(sentence[1], {}).get((label[0], label[1]), Solver.LOWEST_PROBABILITY)) \
                        + math.log(self.tt_transition[self.tag_index[label[0]]][self.tag_index[label[1]]])
                for i in range(2, len(sentence)):
                    P_tag_pair_to_tag = math.log(self.tag_pair_to_tag_table.get((label[i - 2], label[i - 1]), {})
                                                                        .get(label[i], Solver.LOWEST_PROBABILITY)) \
                                    + math.log(self.tt_transition[self.tag_index[label[i - 2]]][self.tag_index[label[i - 1]]])

                    P_tag_pair_to_word = math.log(self.word_tag_pair_table.get(sentence[1], {})
                                                                        .get((label[i - 1], label[i]), Solver.LOWEST_PROBABILITY)) \
                                    + math.log(self.tt_transition[self.tag_index[label[i - 1]]][self.tag_index[label[i - 0]]])

                    product += P_tag_pair_to_tag + P_tag_pair_to_word
        else:
            print("Unknown algo!")
        return product

    def sanitize(self, word):
        word = word.strip()
        if word.startswith("'") and word.endswith("'"):
            return word
        # elif "'" in word:
        #     splits = word.split("'")
        #     try:
        #         i = int(splits[-1])
        #     except Exception as e:

        #     return self.sanitize("'".join(splits[:-1]))
        else:
            return word

    # Do the training!
    #
    def train(self, data):
        self.tags = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", "x", "."]
        self.tag_index = { self.tags[i]: i for i in range(len(self.tags)) }
        self.tt_transition = np.zeros((len(self.tags), len(self.tags)))
        self.tw_emission = [{} for _ in range(len(self.tags))]
        self.initial_probability = np.zeros(len(self.tags))
        self.word_tag_pair_table = {}
        self.tag_pair_to_tag_table = {}
        self.tag_probability = np.zeros(len(self.tags))

        total_words = 0

        for sentence_data in data:
            tag_current_minus_2 = None
            tag_current_minus_1 = None
            for i in range(len(sentence_data[0])):
                if i != 0:
                    tag_current_minus_1 = sentence_data[1][i - 1]
                    curr = sentence_data[1][i]
                    self.tt_transition[self.tag_index[tag_current_minus_1]][self.tag_index[curr]] += 1

                total_words += 1

                word_current = self.sanitize(sentence_data[0][i])
                tag_current = sentence_data[1][i]
                self.tag_probability[self.tag_index[tag_current]] += 1

                # Fill in the emission probabilties
                if word_current not in self.tw_emission[self.tag_index[tag_current]]:
                    self.tw_emission[self.tag_index[tag_current]][word_current] = 0
                self.tw_emission[self.tag_index[tag_current]][word_current] += 1

                # Prepare data for MCMC sampling
                if word_current not in self.word_tag_pair_table:
                    self.word_tag_pair_table[word_current] = []
                self.word_tag_pair_table[word_current].append((tag_current_minus_1, tag_current))

                # Create the tag pair to tag mapping for mcmc
                if (tag_current_minus_2, tag_current_minus_1) not in self.tag_pair_to_tag_table:
                    self.tag_pair_to_tag_table[(tag_current_minus_2, tag_current_minus_1)] = {}
                if tag_current not in self.tag_pair_to_tag_table[(tag_current_minus_2, tag_current_minus_1)]:
                    self.tag_pair_to_tag_table[(tag_current_minus_2, tag_current_minus_1)][tag_current] = 0
                self.tag_pair_to_tag_table[(tag_current_minus_2, tag_current_minus_1)][tag_current] += 1

                tag_current_minus_2 = tag_current_minus_1
                tag_current_minus_1 = tag_current

            # Sentence beginning
            tag_current = sentence_data[1][0]
            self.initial_probability[self.tag_index[tag_current]] += 1

        for tag_current in range(self.tt_transition.shape[0]):
            sum_count = sum(self.tt_transition[tag_current])
            for target_tag_i in range(self.tt_transition.shape[1]):
                self.tt_transition[tag_current][target_tag_i] = self.tt_transition[tag_current][target_tag_i] / sum_count
                if self.tt_transition[tag_current][target_tag_i] == 0.0:
                    self.tt_transition[tag_current][target_tag_i] = Solver.LOWEST_PROBABILITY

        # Calculate the emission probabilties
        for tag_current in range(len(self.tw_emission)):
            sum_count = sum([v for (k, v) in self.tw_emission[tag_current].items()])
            for word_current in self.tw_emission[tag_current]:
                self.tw_emission[tag_current][word_current] = self.tw_emission[tag_current][word_current] / sum_count
                if self.tw_emission[tag_current][word_current] == 0.0:
                    self.tw_emission[tag_current][word_current] = Solver.LOWEST_PROBABILITY

        self.initial_probability[tag_current] = self.initial_probability[tag_current] / len(data)

        for word_current in self.word_tag_pair_table:
            prob = {}
            for pair in self.word_tag_pair_table[word_current]:
                if pair not in prob:
                    prob[pair] = 0
                prob[pair] += 1
            prob = { x: prob[x]/len(self.word_tag_pair_table[word_current]) for x in prob }
            self.word_tag_pair_table[word_current] = prob

        for pair in self.tag_pair_to_tag_table:
            sum_prob = sum([self.tag_pair_to_tag_table[pair][x] for x in self.tag_pair_to_tag_table[pair]])
            prob = { x: self.tag_pair_to_tag_table[pair][x] / sum_prob for x in self.tag_pair_to_tag_table[pair] }
            self.tag_pair_to_tag_table[pair] = prob

        self.tag_probability = self.tag_probability / total_words

    def simplified(self, sentence):
        predictions = []
        for word in sentence:
            word = self.sanitize(word)
            max_i = max([(self.tags[tag_i], self.tw_emission[tag_i][word]) for tag_i in range(len(self.tw_emission)) if word in self.tw_emission[tag_i]], key=lambda x: x[1], default=None)
            if max_i == None:
                predictions.append("x")
            else:
                predictions.append(max_i[0])
        return predictions

    def hmm_viterbi(self, sentence):
        # This implementation is based on the material provided during the class activity for Viterbi algorithm
        # Ref: https://iu.instructure.com/courses/2027431/files/128919777?module_item_id=25300577

        # The viterbi table will be N * N for N letters
        v_table = np.zeros((self.tt_transition.shape[0], len(sentence)))
        lookup = [["" for _ in range(len(sentence))] for _ in range(self.tt_transition.shape[0])]

        for tag_i in range(len(self.tags)):
            v_table[tag_i][0] = self.initial_probability[tag_i] * self.tw_emission[tag_i].get(sentence[0], Solver.LOWEST_PROBABILITY)

        i = 0
        for i in range(1, len(sentence)):
            for tag_i in range(len(self.tags)):
                (lookup[tag_i][i], v_table[tag_i][i]) = max([
                    (self.tags[s0], v_table[s0][i-1] * self.tt_transition[s0][tag_i]) for s0 in range(len(self.tags)) ], key=lambda l:l[1] ) 
                v_table[tag_i][i] *= self.tw_emission[tag_i].get(sentence[i], Solver.LOWEST_PROBABILITY)

        viterbi_sequence = [""] * len(sentence)
        viterbi_sequence[len(sentence) - 1] = self.tags[np.argmax([v_table[tag_i][i] for tag_i in range(len(self.tags))])]

        for i in range(len(sentence) - 2, -1, -1):
            viterbi_sequence[i] = lookup[self.tag_index[viterbi_sequence[i + 1]]][i + 1]

        return viterbi_sequence

    def complex_mcmc(self, sentence):
        sampling_count = 1000

        def sample_tag(tag_2, tag_1):
            tags_prob = self.tag_pair_to_tag_table[(tag_2, tag_1)].items()
            return np.random.choice([x[0] for x in tags_prob], p=[x[1] for x in tags_prob])

        def sample_word(tag_current_minus_2, tag_current_minus_1, current_word):
            if current_word not in self.word_tag_pair_table:
                return ("x", Solver.LOWEST_PROBABILITY)
            tag = sample_tag(tag_current_minus_2, tag_current_minus_1)
            tag_pair_table = self.word_tag_pair_table[current_word]
            return (tag, tag_pair_table.get((tag_current_minus_1, tag), Solver.LOWEST_PROBABILITY))

        predicted_tags_list = [[] for _ in sentence]
        final_prediction = []

        for word_index in range(len(sentence)):
            tag_2 = None
            tag_1 = None

            predicted_tag_infos = []

            # Populate predictions and samples
            for _ in range(sampling_count):
                # Sample the word
                tag_info = sample_word(tag_2, tag_1, sentence[word_index])
                predicted_tag_infos.append(tag_info)

                tag_2 = tag_1
                tag_1 = tag_info[0]

            predicted_tags = [x[0] for x in predicted_tag_infos]

            # Possible tags list for all unique list
            possible_tags_probability = list(dict(sorted(list(set(predicted_tag_infos)), key=lambda x: x[1])).items())

            # Calculate the possible tag probability, get arg max over it
            max_tag = np.argmax([predicted_tags.count(word_prob_pair[0]) * word_prob_pair[1]
                                    for word_prob_pair in possible_tags_probability])
            final_prediction.append(possible_tags_probability[max_tag])

        final_prediction = [x[0] for x in final_prediction]
        return final_prediction



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

