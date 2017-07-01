import word2vec, sys, os, math
import numpy as np
import scipy.spatial as spatial
import cPickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm #used in circular_walk_graph
#import time #used in temporary testing of get_pisa_scores

''' Files used by this class:
        canon_adj.txt        canon_adj_pl.txt
        canon_hypernym.txt   canon_hypernym_pl.txt
        canon_meronym.txt    canon_meronym_pl.txt
        canon_verbs.txt      canon_verbs_pl.txt

        postagged_wikipedia_for_word2vec.bin            (word2vec-compatible file using all pos-tagged Wikipedia, Jan 2016)
        postagged_wikipedia_for_word2vec_30kn3kv.pkl    (scholar-compatible file using top 30k nouns, 3k verbs, same corpus)
        postag_distributions_for_scholar.txt            (pos-tag distributions for all words in Wikipedia)
        postag_distributions_for_scholar_30kn3kv.txt    (pos-tag distributions for top 30k nouns, 3k verbs, same corpus)
'''


class Scholar:

    # Initializes the class
    def __init__(self, slim=False, pisa_scale=1):
        self.slim = slim
        self.pisa_scale = pisa_scale
        if self.slim:
            self.word2vec_bin_loc = 'scholar/postagged_wikipedia_for_word2vec_30kn3kv.pkl'
            self.tag_distribution_loc = 'scholar/postag_distributions_for_scholar_30kn3kv.txt'
        else:
            self.word2vec_bin_loc = 'scholar/postagged_wikipedia_for_word2vec.bin'
            self.tag_distribution_loc = 'scholar/postag_distributions_for_scholar.txt'
        self.number_of_results = 10
        self.number_analogy_results = 20
        self.autoAddTags = True
        self.load_word2vec(self.word2vec_bin_loc)
        # This is a list of the tags as organized in the text file
        self.tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                         'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                         'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                         'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                         'WDT', 'WP', 'WP$', 'WRB']
        self.load_tag_counts(self.tag_distribution_loc)

    # Return a list of words from a file
    def load_desired_vocab(self, filename):
        text = open(filename)
        word_list = []
        for line in text:
            word_list.append(line.replace('\n', ''))
        return word_list

    # Loads the word2vec model from a specified file
    def load_word2vec(self, model_filename):
        if self.slim:
            self.model = pkl.load(open(model_filename, 'r'))
        else:
            self.model = word2vec.load(model_filename)

    # Loads the part of speech tag counts into a dictionary (words to tag string delimited by '-'s)
    def load_tag_counts(self, tag_count_filename):
        # Read in the tag information for each word from the file
        with open(tag_count_filename) as f:
            word_tag_dist = f.read()

        # Save each word to a list of tags in a global dictionary
        self.word_to_tags = {}
        for line in word_tag_dist.split():
            pieces = line.split('.')
            word = pieces[0]
            tags = pieces[1].split('-')
            tags = map(int, tags)
            self.word_to_tags[word] = tags

    # Return the cosine similarity of vectors for two specified words
    def get_cosine_similarity(self, word1, word2):
        vec1 = self.model.get_vector(word1)
        vec2 = self.model.get_vector(word2)
        dividend = np.dot(vec1, vec2)
        divisor = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        result = dividend / divisor
        return result

    # Return the vector for a word, else return None
    def get_vector(self, word):
        if self.exists_in_model(word):
            return self.model.get_vector(word)
        return None

    # Return the angle between two vectors (assumes a hypersphere)
    #   Angle returned is a simple 2D angle, not a multidimensional set of angles.
    def angle(self, vec1, vec2):
        unit_vec1 = vec1 / np.linalg.norm(vec1)
        unit_vec2 = vec2 / np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))

    # Return the angle between two words
    def get_angle(self, word1, word2):
        vec1 = self.model.get_vector(word1)
        vec2 = self.model.get_vector(word2)
        return self.angle(vec1, vec2)

    # Return the analogy results for a list of words (input: "king -man woman")
    def analogy(self, words_string):
        positives, negatives = self.get_positives_and_negatives(words_string.split())
        return self.get_results_for_words(positives, negatives)

    # Takes a list of words (ie 'king woman -man') and separates them into two lists (ie '["king", "woman"], ["man"]')
    def get_positives_and_negatives(self, words):
        positives = []
        negatives = []
        for x in range(len(words)):
            word_arg = words[x]
            if word_arg.startswith('-'):
                negatives.append(word_arg[1:])
            else:
                positives.append(word_arg)
        return positives, negatives

    # Returns the results of entering a list of positive and negative words into word2vec
    def get_results_for_words(self, positives, negatives):
        indexes, metrics = self.model.analogy(pos=positives, neg=negatives, n=self.number_analogy_results)
        results = self.model.generate_response(indexes, metrics).tolist()
        return self.format_output(results)

    # Changes the output from a list of tuples (u'man', 0.816015154188), ... to a list of single words
    def format_output(self, output):
        words = []
        for word_value in output:
            words.append(str(word_value[0]))
        return words

    # Returns a list of the words in a tagged sentence ordered by salience (as determined by Word2Vec)
    def get_words_by_salience(self, sentence):
        sentence = sentence.split()
        word_vectors = []
        # Get the vectors for every word in the sentence
        for tagged_word in sentence:
            word_vectors.append(self.model[tagged_word])
        word_salience = {}
        # For every word in the sentence...
        for word_index in range( len(sentence) ):
            total_vector = np.array([0.0] * 100)
            # Add up the vectors for every other word in the sentence...
            for vector_index in range( len(word_vectors) ):
                if word_index != vector_index:
                    total_vector += word_vectors[vector_index]
            # Find the average for those vectors
            average_vector = total_vector / float( len(word_vectors) - 1 )
            # Take the difference of the average vector and the current word vector
            difference_list = ( average_vector - word_vectors[word_index] ).tolist()
            difference_scalar = 0
            # For every scalar in the difference vector...
            for difference_number in difference_list:
                # Add that squared number to a single scalar
                difference_scalar += math.pow(difference_number, 2)
            # The square root of that single scalar is the key in a dictionary
            word_salience[ math.sqrt(difference_scalar) ] = sentence[word_index]
        words_sorted_by_salience = []
        # Add words in order of lowest salience to highest
        for key in sorted(word_salience.iterkeys()):
            words_sorted_by_salience.append(word_salience[key])
        # Reverse the list
        words_sorted_by_salience.reverse()
        return words_sorted_by_salience

    ''' # No idea why this was here... Daniel?
    def get_vector(self, word):
        return self.model[word]
    '''

    def get_canonical_results_for_nouns_hyper(self, noun, query_tag, canonical_tag_filename, plural, number_of_user_results):
        if self.autoAddTags:
            noun += '_NNS' if plural else '_NN'
        canonical_pairs = open(canonical_tag_filename)
        result_map = {}
        # For every line in the file of canonical pairs...
        for line in canonical_pairs:
            # ...split into separate words...
            words = line.split()
            if plural:
                if query_tag == 'VB' or query_tag == 'JJ':
                    query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NNS ' + noun
                elif query_tag == 'HYPER':
                    query_string = words[0] + '_NNS -' + words[1] + '_NNS ' + noun
                elif query_tag == 'HYPO':
                    query_string = words[1] + '_NNS -' + words[0] + '_NNS ' + noun
                elif query_tag == 'PARTS':
                    query_string = '-' + words[0] + '_NNS ' + words[1] + '_NNS ' + noun
                elif query_tag == 'WHOLE':
                    query_string = '-' + words[1] + '_NNS ' + words[0] + '_NNS ' + noun
            else:
                if query_tag == 'VB' or query_tag == 'JJ':
                    query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NN ' + noun
                elif query_tag == 'HYPER':
                    query_string = words[0] + '_NN -' + words[1] + '_NN ' + noun
                elif query_tag == 'HYPO':
                    query_string = words[1] + '_NN -' + words[0] + '_NN ' + noun
                elif query_tag == 'PARTS':
                    query_string = '-' + words[0] + '_NN ' + words[1] + '_NN ' + noun
                elif query_tag == 'WHOLE':
                    query_string = '-' + words[1] + '_NN ' + words[0] + '_NN ' + noun

            # ...performs an analogy using the words...
            try:
                result_list = self.analogy(query_string)
            except:
                result_list = []
            # ...and adds those results to a map (sorting depending on popularity, Poll method)
            for result in result_list:
                if result in result_map.keys():
                    result_map[result] += 1
                else:
                    result_map[result] = 1
        final_results = []
        current_max = number_of_user_results
        # While we haven't reached the requested number of results and the number of possible matches is within reason...
        while len(final_results) < number_of_user_results and current_max > 0:
            # ...for every key in the results...
            for key in result_map.keys():
                # ...if the number of times a result has been seen equals the current 'number of matches'...
                if result_map[key] == current_max:
                    # ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
                    final_results.append(key)
            current_max -= 1
        if len(final_results) >= number_of_user_results:
            return final_results[0:number_of_user_results]
        return final_results

    # Returns the canonical results for verbs
    def get_verbs(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'VB', 'scholar/canon_verbs.txt', False, number_of_user_results)

    # Returns the canonical results for nouns
    def get_nouns(self, verb, number_of_user_results):
        return self.get_canonical_results_for_verbs(verb, 'scholar/canon_verbs.txt', False, number_of_user_results)

    # Returns the canonical results for adjectives
    def get_adjectives(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'JJ', 'scholar/canon_adj.txt', False, number_of_user_results)

    # Returns the canonical results for hypernyms (generalized words)
    def get_hypernyms(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'HYPER', 'scholar/canon_hypernym.txt', False, number_of_user_results)

    # Returns the canonical results for hyponyms (specific words)
    def get_hyponyms(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'HYPO', 'scholar/canon_hypernym.txt', False, number_of_user_results)

    # Returns the canonical results for parts of the given noun
    def get_parts(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'PARTS', 'scholar/canon_meronym.txt', False, number_of_user_results)

    # Returns the canonical results for things the noun could be a part of
    def get_whole(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'WHOLE', 'scholar/canon_meronym.txt', False, number_of_user_results)

    # Returns the canonical results for verbs (plural)
    def get_verbs_plural(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'VB', 'scholar/canon_verbs_pl.txt', True, number_of_user_results)

    # Returns the canonical results for nouns (plural)
    def get_nouns_plural(self, verb, number_of_user_results):
        return self.get_canonical_results_for_verbs(verb, 'scholar/canon_verbs.txt', True, number_of_user_results)

    # Returns the canonical results for adjectives (plural)
    def get_adjectives_plural(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'JJ', 'scholar/canon_adj_pl.txt', True, number_of_user_results)

    # Returns the canonical results for hypernyms (generalized words) (plural)
    def get_hypernyms_plural(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'HYPER', 'scholar/canon_hypernym_pl.txt', True, number_of_user_results)

    # Returns the canonical results for hyponyms (specific words) (plural)
    def get_hyponyms_plural(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'HYPO', 'scholar/canon_hypernym_pl.txt', True, number_of_user_results)

    # Returns the canonical results for parts of the given noun (plural)
    def get_parts_plural(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'PARTS', 'scholar/canon_meronym_pl.txt', True, number_of_user_results)

    # Returns the canonical results for things the noun could be a part of (plural)
    def get_whole_plural(self, noun, number_of_user_results):
        return self.get_canonical_results_for_nouns(noun, 'WHOLE', 'scholar/canon_meronym_pl.txt', True, number_of_user_results)

    # Returns canonical results for specified relationships between words
    # As an aside, this is simply returning the results of all the analogies from all the canonical pairs.
    # Occasionally it returns unexpected tags (ie user requested a list of adjectives related to a noun,
    # and got mostly adjectives but also one preposition). Be aware of this if it matters.
    def get_canonical_results_for_nouns(self, noun, query_tag, canonical_tag_filename, plural, number_of_user_results):
        if self.autoAddTags:
            noun += '_NNS' if plural else '_NN'
        canonical_pairs = open(canonical_tag_filename)
        result_map = {}
        # For every line in the file of canonical pairs...
        for line in canonical_pairs:
            # ...split into separate words...
            words = line.split()
            if plural:
                if query_tag == 'VB' or query_tag == 'JJ':
                    query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NNS ' + noun
                elif query_tag == 'HYPER':
                    query_string = words[0] + '_NNS -' + words[1] + '_NNS ' + noun
                elif query_tag == 'HYPO':
                    query_string = words[1] + '_NNS -' + words[0] + '_NNS ' + noun
                elif query_tag == 'PARTS':
                    query_string = '-' + words[0] + '_NNS ' + words[1] + '_NNS ' + noun
                elif query_tag == 'WHOLE':
                    query_string = '-' + words[1] + '_NNS ' + words[0] + '_NNS ' + noun
            else:
                if query_tag == 'VB' or query_tag == 'JJ':
                    query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NN ' + noun
                elif query_tag == 'HYPER':
                    query_string = words[0] + '_NN -' + words[1] + '_NN ' + noun
                elif query_tag == 'HYPO':
                    query_string = words[1] + '_NN -' + words[0] + '_NN ' + noun
                elif query_tag == 'PARTS':
                    query_string = '-' + words[0] + '_NN ' + words[1] + '_NN ' + noun
                elif query_tag == 'WHOLE':
                    query_string = '-' + words[1] + '_NN ' + words[0] + '_NN ' + noun

            # ...performs an analogy using the words...
            try:
                result_list = self.analogy(query_string)
            except:
                result_list = []
            # ...and adds those results to a map (sorting depending on popularity, Poll method)
            for result in result_list:
                if result in result_map.keys():
                    result_map[result] += 1
                else:
                    result_map[result] = 1
        final_results = []
        current_max = number_of_user_results
        # While we haven't reached the requested number of results and the number of possible matches is within reason...
        while len(final_results) < number_of_user_results and current_max > 0:
            # ...for every key in the results...
            for key in result_map.keys():
                # ...if the number of times a result has been seen equals the current 'number of matches'...
                if result_map[key] == current_max:
                    # ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
                    final_results.append(key)
            current_max -= 1
        if len(final_results) >= number_of_user_results:
            return final_results[0:number_of_user_results]
        return final_results

    # Returns canonical results for specified relationships between words
    # As an aside, this is simply returning the results of all the analogies from all the canonical pairs.
    # Occasionally it returns unexpected tags (ie user requested a list of adjectives related to a noun,
    # and got mostly adjectives but also one preposition). Be aware of this if it matters.
    def get_canonical_results_for_verbs(self, verb, canonical_tag_filename, plural, number_of_user_results):
        canonical_pairs = open(canonical_tag_filename)
        result_map = {}
        # For every line in the file of canonical pairs...
        for line in canonical_pairs:
            # ...split into separate words...
            words = line.split()
            if plural:
                query_string = words[1] + '_NNS' + ' -' + words[0] + '_VB ' + verb + '_VB'
            else:
                query_string = words[1] + '_NN' + ' -' + words[0] + '_VB ' + verb + '_VB'

            # ...performs an analogy using the words...
            try:
                result_list = self.analogy(query_string)
            except:
                result_list = []
            # ...and adds those results to a map (sorting depending on popularity, Poll method)
            for result in result_list:
                if result in result_map.keys():
                    result_map[result] += 1
                else:
                    result_map[result] = 1
        final_results = []
        current_max = number_of_user_results
        # While we haven't reached the requested number of results and the number of possible matches is within reason...
        while len(final_results) < number_of_user_results and current_max > 0:
            # ...for every key in the results...
            for key in result_map.keys():
                # ...if the number of times a result has been seen equals the current 'number of matches'...
                if result_map[key] == current_max:
                    # ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
                    final_results.append(key)
            current_max -= 1
        if len(final_results) >= number_of_user_results:
            return final_results[0:number_of_user_results]
        return final_results

    def get_most_common_words(self, pos_tag, number_of_results):
        # If the tag doesn't exist, return nothing
        if pos_tag not in self.tag_list or not os.path.exists(self.tag_distribution_loc):
            return []

        # Get the index of the specific tag requested in the list above
        tag_index = self.tag_list.index(pos_tag)

        # Read in the tag information for each word from the file
        with open(self.tag_distribution_loc) as f:
            word_tag_dist = f.read()

        tag_to_word = {}

        # For each of the lines in the text file... (dog.0-0-0-0-0-4-0-0-90-3-0-0-etc.)
        for line in word_tag_dist.split():
            pieces = line.split('.')
            word = pieces[0]
            tags = pieces[1].split('-')
            current_tag = int(tags[tag_index])
            # Add to the dictionary of tag numbers to words
            try:
                tag_to_word[current_tag].append(word)
            except:
                tag_to_word[current_tag] = []
                tag_to_word[current_tag].append(word)

        common_words = []
        taglist = tag_to_word.keys()
        # Sort the list of tag numbers from lowest to highest
        if (sys.version_info > (3, 0)):
            taglist = sorted(taglist, key=lambda k: int(k))
        else:
            taglist.sort()
        # Reverse the list (to highest to lowest)
        taglist.reverse()
        # Add the words for each tag number to a list
        for tag in taglist:
            common_words += tag_to_word[tag]

        # Only return the number of results specified by the user
        return common_words[:number_of_results]

    def get_words_by_rarity(self, sentence):
        # Clean up input sentence (remove punctuation and unnecessary white space)
        sentence = sentence.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').replace(':', ' ').replace(';', ' ').replace('-', ' ')
        while '  ' in sentence:
            sentence = sentence.replace('  ', ' ')
        # Create dictionary of words to their popularities
        word_to_pop = {}
        for word in sentence.split():
            word_to_pop[word] = self.get_word_popularity(word)
        # Return list of words sorted by popularities
        return sorted(word_to_pop, key=word_to_pop.__getitem__)

    # Returns the popularity of a word (without a tag)
    def get_word_popularity(self, word):
        try:
            popularity = 0
            for tag_amount in self.word_to_tags[word]:
                popularity += int(tag_amount)  # int(self.word_to_tags[word][tag_amount])
            return popularity
        except:
            if (sys.version_info > (3, 0)):
                return math.inf
            else:
                return float('inf')

    # Returns the most common tag for a specific word
    def get_most_common_tag(self, word):
        word_tags = self.word_to_tags[word]
        return self.tag_list[word_tags.index(max(word_tags))]

    # Returns True if the word/tag pair exists in the Wikipedia corpus
    def exists_in_model(self, word):
        try:
            vector = self.model.get_vector(word)
            return True
        except:
            return False


###############################################################################
#   The Yarax / Pisa Expansion Set
###############################################################################

    """
        The following functionalities were added by
        Nathan Tibbetts, Zachary Brown, and Nancy Fulda

        Many of them, particularly when finding the nearest word to a given
        vector, are dependent upon an edited version of
        word2vec's wordvectors.py, not included here.
        For more information, contact Daniel Ricks, author of scholar.
    """

#-----------------------------------------------------------------------------#
#   Utilities and Helper Functions
#-----------------------------------------------------------------------------#


    def yarax(self, vec_x, vec_dir, theta):
        '''
            Distance Respecting Hypersphere Traversal
            Arc-tracer, instead of vector estimation.
            -----------------------------------------
            vec_x: the word vector to apply an analogy to--a normalized vector.
                This is our starting point.
            vec_ref: the vector of the analogy - length does not matter.
            theta: the angle traversed around the hypersphere
                in the direction of the reference vector,
                starting from the tip of vec_x. In radians.
            returns: the vector resulting from the angular traversal.
            Methodology: y = A^(-1) * R * A * x, where:
                x is our starting vector,
                A is the basis of the plane we want to rotate on,
                    made from vec_x and vec_dir, with vec_dir orthonormalized,
                R is the simple 2D rotation matrix,
                A^(-1) is A inverse -
                    achieved by transposing A, since A is orthonormal, and
                y is the rotated vector, now in the original basis. Returns.
            NOTE: reversing direction is not as simple as a negative angle,
                simply because if we have rotated past a certain point,
                our direction vector will already be pointing backwards!
        '''
        # Gram-Schmidt on second row of basis of plane, vec_dir:
        #    Orthogonalize 2nd row to 1st,
        row2 = vec_dir - np.dot(vec_x, vec_dir)*vec_x
        row2 /= np.linalg.norm(row2)  # ...and normalize it.
        return np.dot(
            np.vstack((vec_x, row2)).T,  # The basis of the plane to rotate in,
                # This is A^(-1), or A^T because it is orthonormal (truncated).
            np.array([np.cos(theta), np.sin(theta)]).T) # Truncated R;
                # This is R*A*x, where A*x = [1,0] because it is the
                #   representation of a vector in a plane created from itself
                #   and an orthogonal vector.
                # Thus only the left half of the original R remains.

        # OLD:
        '''row2 = vec_dir - np.dot(vec_x, vec_dir)*vec_x
        row2 /= np.linalg.norm(row2)  # ...and normalize it.
        return np.dot(
            np.vstack((vec_x, row2)).T,  # The basis of the plane to rotate in.
            np.dot(
                # The rotational matrix:
                [[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta),  np.cos(theta)]],
                # The representation of our vector to rotate,
                #    in the basis of the plane to rotate it in:
                [1, 0]))'''


    def average_vector(self, vec_list):
        ''' Finds an average vector, given a list of vectors.'''
        return sum(vec_list) / len(vec_list)


    # Helper function to deal with returned word indecies instead of words:
    def wordify(self, dual_set):
        # Takes the output set of vectors and similarities
        #     and converts indecies to words.
        words = dual_set[0].tolist()
        similarities = dual_set[1]
        num_words = np.size(words)
        for word in range(num_words):
            words[word] = self.model.vocab[words[word]]
        return np.array(words), similarities

        #dual_set[0] = dual_set[0].tolist()
        #dual_set[0] = map(self.model.vocab[], dual_set[0]
        #OR LIST COMPREHENSION!


#-----------------------------------------------------------------------------#
#   Scoring Methods
#-----------------------------------------------------------------------------#


    #spot_to_score is the vec of the word you're measuring the "distance" to.
    #center_of_search is the vector of the word you got to by applying the
    #   analogy vector to your source word
    #direction_of_travel is the analogy vector

    def pisa_score(self, spot_to_score, center_of_search, direction_of_travel):
        #search in lopsided rings that stretch in the direction of travel
        #(You know, like the leaning tower of Pisa...)

        supplementary_distance = spatial.distance.cosine(
            center_of_search + direction_of_travel, # vec_to_source
            center_of_search - spot_to_score)       # vec_to_word

        score = (self.pisa_scale * (2.0-supplementary_distance) +
                 # base_distance:
                 spatial.distance.cosine(spot_to_score,center_of_search))

        return score
        #NOTES: 2.0 is not dependent on vectors being normalized.
        #   Scale factor adjusts length, not angle, measure.
        #       Overall scale doesn't matter because it's relative to itself.
        #   Why depend on direction of travel? Can use cosine_dist sums
        #       of all source words independent of direction of travel...
        #       although being dependent on direction of travel is interesting,
        #       as we know that's essentially what yarax does.


    def get_pisa_scores(self, db_words, center_of_search, direction_of_travel,
                        scale='default'):
        if scale == 'default':
            scale = self.pisa_scale
        #CALCULATE COSINE SIMILARITIES OF ALL VECTORS TO SOURCE WORD
        #(We normalize center_of_search in case it doesn't have unit length,
        #but we assume db_words are normalized already...)
        base_distances = 1 - (np.dot(db_words,center_of_search.T) /
                              np.linalg.norm(center_of_search))
        #OR, IN CASE SPACE ISN'T NORMALIZED:
        #base_distances = 1 - np.dot(db_words,center_of_search.T)/(
        #    np.linalg.norm(db_words,axis=1)*np.linalg.norm(center_of_search))

        vector_to_source = center_of_search + direction_of_travel
        vectors_to_words = center_of_search - db_words
        supplementary_distances = (
            1 - (np.dot(vectors_to_words, vector_to_source.T) /
                 (np.linalg.norm(vectors_to_words,axis=1) *
                  np.linalg.norm(vector_to_source))))

        return scale * (2-supplementary_distances) + base_distances


    def get_closest_words_pisa(self, vec_from_a, vec_to_b, vec_on_c, end_vec,
                               num_words=10,
                               pisa="diagonal", scale='default'):
        if scale == 'default':
            scale = self.pisa_scale
        # end_vec is the result of the analogy, and the center of our search.
        if pisa == "diagonal":
            # Move away from all source words.
            # Note: if we used (b-a)+(c-a) this would be slightly different
            #   when using analogies other than Mikolov's vector addition.
            affordance_vec = end_vec - vec_from_a
        elif pisa == "source":
            # Move away from c, the start of the current comparison.
            affordance_vec = end_vec - vec_on_c
        elif pisa == "analogous":
            # Move away from b, the analogical equivalent.
            affordance_vec = end_vec - vec_to_b
        elif pisa == "origin":
            # Move in direction from c to d, but starting from c.
            affordance_vec = end_vec - vec_on_c
        else:
            raise ValueError("Skipped origin call?")
            print "Unrecognized Pisa type \"" + pisa + "\"."
            return np.array([]), np.array([])

        '''
        distances = []
        for v in self.model.vectors:
            # Normal distance metric: (can use this function to replace other)
            #distances.append(spatial.distance.cosine(v, vec_d))
            distances.append(self.pisa_score(v, end_vec, affordance_vec))
            # DO MATH INSTEAD OF FOR, OR LIST COMPREHENSION AT LEAST'''
        if pisa == "origin":
            distances = self.get_pisa_scores(self.model.vectors, vec_on_c,
                                             affordance_vec, scale=scale)
        else:
            distances = self.get_pisa_scores(self.model.vectors, end_vec,
                                             affordance_vec, scale=scale)
                                             #.tolist()

        found_words = []
        found_distance = []
        for count in range(0, num_words):
            #OLDER AND SLOW:
            #min_dist = min(distances)
            #index = distances.index(min_dist)
            #OLD AND LESS FAST:
            #min_dist = np.amin(distances)
            #index = np.where(distances==min_dist)[0][0]
            #found_words.append(self.model.vocab[index])
            #found_distance.append(min_dist)
            index = np.argmin(distances)
            found_words.append(self.model.vocab[index])
            found_distance.append(distances[index])
            distances[index] = 1000 # Now not min.

        return np.array(map(str,found_words)), np.array(found_distance)


    # Wrapper for other functions
    def get_closest_words(self, vec, num_words=1,
                          exclude=False, exc_list=[], pisa="none"):
        if pisa == "none":
            if exclude:
                return self.model.get_closest_words_excluding(
                    vec, exc_list, num_words)
            else: return self.model.get_closest_words(vec, num_words)
        else: #####STUFF!!!
            pass

#-----------------------------------------------------------------------------#
#   Analogies
#-----------------------------------------------------------------------------#


    def normal_analogy(self, from_a, to_b, on_c,
                       num_words=1, exclude=True, pisa="none"):
        '''
            Expanded function for for the original analogy designed by Mikolov.
            The relationship from a to b, applied on c.
            In Mikolov's code, pos = [to_b, on_c], neg = [from_a],
                or: b - a + c = d.
                Parameter order matches analogical reasoning, "a:b::c:d".
            -------------------------------------------------------------------
            num_words = the number of closest results to return.
            exclude = whether or not to exclude the source words.
                Note: this parameter ignored if using Pisa distance scoring.
            pisa = the type of pisa scoring to use, namely:
                "none",
                "source" to move away in the direction from c to d,
                "diagonal" to move away in the direction from a to d.
        '''
        vec_a = self.model.get_vector(from_a)
        vec_b = self.model.get_vector(to_b)
        vec_c = self.model.get_vector(on_c)
        #analogy_vec = vec_b - vec_a
        end_vec = vec_b - vec_a + vec_c
        """
        """
        end_vec /= np.linalg.norm(end_vec)#FIX THIS!!!

        if pisa != "none":
            return self.get_closest_words_pisa(vec_a, vec_b, vec_c, end_vec,
                                               num_words=num_words, pisa=pisa)
        elif exclude:
            #return self.wordify(self.model.get_closest_words_excluding(
            #    end_vec, [vec_a, vec_b, vec_c], num_words))
            return self.wordify(self.model.analogy(
                [to_b, on_c], [from_a], num_words))
        else:
            return self.wordify(self.model.get_closest_words(end_vec, num_words))


    def yarax_analogy(self, from_a, to_b, on_c,
                      num_words=5, exclude=True,
                      mode="relative", angle_scale=1, pisa="none"):
        '''
            Analogy with the Yarax Method
            -----------------------------
            Options: Can set mode="relative", mode="degrees", or mode="radians"
                     Can set exclude=True or exclude=False.
                     Can set pisa="none", "source", "diagonal", or "analogous".
            Note: exclude argument ignored if pisa is used.
        '''
        vec_a = self.model.get_vector(from_a)
        vec_b = self.model.get_vector(to_b)
        vec_c = self.model.get_vector(on_c)
        analogy_dir = vec_b - vec_a
        if mode == "radians": analogy_angle = angle_scale
        elif mode == "degrees": analogy_angle = angle_scale*np.pi/180.0
        elif mode == "relative":
            analogy_angle = self.get_angle(from_a, to_b)*angle_scale
        else: raise Exception("Unrecognized angle mode.")
        end_vec = self.yarax(vec_c, analogy_dir, analogy_angle)
        end_vec /= np.linalg.norm(end_vec)

        if pisa != "none":
            return self.get_closest_words_pisa(vec_a, vec_b, vec_c, end_vec,
                                               num_words=num_words, pisa=pisa)
        elif exclude:
            if self.slim == True: # This branch other part of patch:
                results = self.wordify(
                    self.model.get_closest_words(end_vec, num_words+3))
                trimmed = ([word for word in results[0]
                            if word not in [from_a, to_b, on_c]],
                           [results[1][i] for i in range(len(results[1]))
                            if results[0][i] not in [from_a, to_b, on_c]])
                return (np.array(trimmed[0][:num_words:]),
                        np.array(trimmed[1][:num_words:]))
            else: # This branch is the original return:
                return self.wordify(self.model.get_closest_words_excluding(
                    end_vec, [vec_a, vec_b, vec_c], num_words))
        else: # The real original return...
            return self.wordify(
                self.model.get_closest_words(end_vec, num_words))


    def hydra_analogy(self, from_a, to_b, on_c, like_e="", is_to_f="",
                           yarax=True, num_checks=5, exclude=True,
                           pisa="none"):
        '''
            A reinforced analogy that reapplies resulting potential analogies
            to the original or a second to check which result seems closest.
        '''
        if like_e == "":
            like_e = from_a
        if is_to_f == "":
            is_to_f = to_b
        end_vec = []
        distance = []

        if yarax:
            is_to_d = self.yarax_analogy(from_a, to_b, on_c,
                                         num_words=num_checks, exclude=exclude,
                                         mode="relative", angle_scale=1,
                                         pisa=pisa)[0]
            for i in range(len(is_to_d)):
                analogy_dir = (self.model.get_vector(is_to_d[i]) -
                               self.model.get_vector(on_c))
                analogy_len = self.get_angle(on_c, is_to_d[i])
                end_vec.append(self.yarax(
                    self.model.get_vector(like_e), analogy_dir, analogy_len))
                end_vec[i] /= np.linalg.norm(end_vec[i])
                distance.append(self.angle(end_vec[i],
                                self.model.get_vector(is_to_f)))
        else:
            is_to_d = self.normal_analogy(from_a, to_b, on_c,
                                          num_words=num_checks,
                                          exclude=exclude, pisa=pisa)[0]
            for i in range(len(is_to_d)):
                analogy_vec = (self.model.get_vector(is_to_d[i]) -
                               self.model.get_vector(on_c))
                end_vec.append(self.model.get_vector(like_e) + analogy_vec)
                end_vec[i] /= np.linalg.norm(end_vec[i])
                distance.append(self.angle(end_vec[i],
                                self.model.get_vector(is_to_f)))

        #index_of_min = distance.index(min(distance))
        #return [is_to_d[index_of_min]], [distance[index_of_min]]
        pairs = zip(is_to_d, distance)
        pairs = sorted(pairs, key=lambda f:f[1])
        is_to_d, distance = zip(*pairs)
        return np.array(map(str,is_to_d)), np.array(distance)


    def two_way_yarax_analogy(self, from_a, to_b, on_c,
                              num_words=1, exclude=True, pisa="none"):
        ''' Do yarax both ways possible on given analogy,
            then find the average between the two answers.
            Note: with the normal analogy, they would be identical.'''
        vec_a = self.model.get_vector(from_a)
        vec_b = self.model.get_vector(to_b)
        vec_c = self.model.get_vector(on_c)
        analogy_dir_1 = vec_b - vec_a
        analogy_dir_2 = vec_c - vec_a
        analogy_angle_1 = self.angle(vec_a,vec_b)
        analogy_angle_2 = self.angle(vec_a,vec_c)
        end_vec_1 = self.yarax(vec_c,analogy_dir_1,analogy_angle_1)
        end_vec_2 = self.yarax(vec_b,analogy_dir_2,analogy_angle_2)
        end_vec_1 /= np.linalg.norm(end_vec_1)
        end_vec_2 /= np.linalg.norm(end_vec_2)
        end_avg = end_vec_1 + end_vec_2
        end_avg /= np.linalg.norm(end_avg)
        #Original:
        #   return self.wordify(self.model.get_closest_words(end_avg, num_words))
        if pisa != "none":
            return self.get_closest_words_pisa(vec_a, vec_b, vec_c, end_avg,
                                               num_words=num_words, pisa=pisa)
        elif exclude:
            if self.slim == True: # This branch other part of patch:
                results = self.wordify(
                    self.model.get_closest_words(end_avg, num_words+3))
                trimmed = ([word for word in results[0]
                            if word not in [from_a, to_b, on_c]],
                           [results[1][i] for i in range(len(results[1]))
                            if results[0][i] not in [from_a, to_b, on_c]])
                return (np.array(trimmed[0][:num_words:]),
                        np.array(trimmed[1][:num_words:]))
            else: # This branch is the original return:
                return self.wordify(self.model.get_closest_words_excluding(
                    end_avg, [vec_a, vec_b, vec_c], num_words))
        else:
            return self.wordify(
                self.model.get_closest_words(end_avg, num_words))


    def yarax_intersect_analogy(self, from_a, to_b, on_c,
                                num_words=1, exclude=True, pisa="none"):
        ''' Do yarax from both directions, using as angle the place where the
            two traced arcs would intersect, whether closer or farther.'''
        raise NotImplementedError("Intersect not yet implemented.")


#-----------------------------------------------------------------------------#
#   Analogy Toolset
#-----------------------------------------------------------------------------#


    def compare_analogies(self, word_from, word_to, apply_to_word,
                          num_words=10, exclude=True,
                          mode="relative", angle_scale=1, pisa="none"):
        words_y, sims_y = self.yarax_analogy(word_from, word_to, apply_to_word,
            num_words, exclude, mode, angle_scale, pisa)
        words_n, sims_n = self.normal_analogy(word_from, word_to,
            apply_to_word, num_words, exclude, mode, angle_scale, pisa)
        print
        print "YARAX:\tNORMAL:\t\tCOS_DIF*1K:"
        for w in range(np.size(words_y)):
            if words_y[w] == words_n[w]:
                print words_y[w], "\t\t", 1000*(sims_y[w]-sims_n[w])
            else:
                print words_y[w], words_n[w]


    def analogy_convergence(self, from_a, to_b, num_words=10):
        ''' Finds where continued normal vector addition analogies would
            converge to if taken repeatedly, by simply
            normalizing the analogy vector and searching nearby.'''
        end_vec = self.model.get_vector(to_b) - self.model.get_vector(from_a)
        return self.wordify(self.model.get_closest_words(
            end_vec / np.linalg.norm(end_vec),
            num_words))


    def analogical_walk(self, word_from,    word_to,         apply_to_word,
                        num_words=5,        mode="radians",  pisa="none",
                        start=0,            stop=2*np.pi,    step=np.pi/24.0):
                        # Note that in relative mode, these last three
                        #   are measured in multiples of the analogical angle.
                        # Start inclusive. Stop also, if step lands on stop.
        words = []
        for i in range(int(start/step), int(stop/step+1), 1):
            words.append(self.yarax_analogy(
                word_from, word_to, apply_to_word, num_words=num_words,
                exclude=False, mode=mode, angle_scale=i*step,
                pisa=pisa)[0].tolist())
        words = np.array(words)
        for x in range(len(words)):
            string = ""
            for word in words[x]:
                string += word + "  "
            print string
        # return np.array(words)


    def circular_walk_graph(self, from_a, to_b, on_c,
                            num_closest=3, pisa="none"):
        '''
            Walks around the hypersphere in a circle defined by the analogy,
                taking note of all words appearing within the n closest at any
                time. Then walks around again, tracking their angular distance
                from every point along that circular walk, and graphing them.
            Good for visualizing analogical progression, as well as the
                supercluster in the space, and minor clusters.
            Whatever line is lowest at any given point is the "closest" word.
            Also takes note of where normal and Yarax analogies would fall,
                drawing a vertical line at that position in the walk.
            x-axis: my position, in degrees, on my circular walk.
            y-axis: angle (centered at origin of sphere) from self to features.
        '''
        words = []
        angles = []
        color_tags = ['k-','g-','b-','m-','r-']
        colors = ['black','green','blue','magenta','red']
        vec_a = self.get_vector(from_a)
        vec_b = self.get_vector(to_b)
        vec_c = self.get_vector(on_c)
        analogy_dir = vec_b - vec_a

        # Walk around circle to collect words we want to watch:
        for i in tqdm(range(0,360,3), desc="Initial Collection Walk"):
            next_group = self.yarax_analogy(from_a, to_b, on_c,
                num_words=num_closest, exclude=False,
                mode="degrees", angle_scale=i, pisa=pisa)[0]
            for j in next_group:
                if j not in words:
                    words.append(j)
                    angles.append(i) # Can use this as label locations as well.

        # GRAPH CONSTRUCTION:
        move = 0 # An offset for the labels
        indecies = range(len(words))
        # This is our baseline - the vectors at each angle along the way:
        position_vecs = [self.yarax(vec_c, analogy_dir,
                            i*np.pi/180.0) for i in range(360)]
        word_vecs = [self.get_vector(j) for j in words]
        if pisa == "diagonal":
            graphs = [[self.pisa_score(word_vecs[w], position_vecs[i],
                                       position_vecs[i] - vec_a)
                       for i in range(360)] for w in tqdm(indecies,
                                            desc="Diagonal Pisa Measuring")]
        elif pisa == "source":
            graphs = [[self.pisa_score(word_vecs[w], position_vecs[i],
                                       position_vecs[i] - vec_c)
                       for i in range(360)] for w in tqdm(indecies,
                                            desc="Source Pisa Measuring")]
        elif pisa == "analogous":
            graphs = [[self.pisa_score(word_vecs[w], position_vecs[i],
                                       position_vecs[i] - vec_b)
                       for i in range(360)] for w in tqdm(indecies,
                                            desc="Analogous Pisa Measuring")]
        else:
            if pisa != "none":
                print "Unrecognized pisa type!"
                return
            move = 2
            graphs = [[self.angle(position_vecs[i], word_vecs[w])*180/np.pi
                       for i in range(360)] for w in tqdm(indecies,
                                            desc="Angle Measuring")]

        # Find the troughs, so we can put word labels there:
        for w in indecies:
            proximity = 180
            for n in range(360):
                if graphs[w][n] < proximity:
                    proximity = graphs[w][n]
                    angles[w] = n # The angles at which to place word labels.

        # Find the places where our analogies would end up:
        yarax_real = self.get_angle(from_a, to_b)*180/np.pi
        normal_real = self.angle(vec_c + analogy_dir, vec_c)*180/np.pi

        # PLOTTING AND GRAPHING:
        plt.title("Angular Distances of Top " + str(num_closest) +
                  " Words Passing Near Walk Ring")
        plt.xlabel("1-Degree steps along ring of " +
                   to_b + " - " + from_a + " + " + on_c)
        plt.ylabel("Degrees away from ring")
        # Plot the real Yarax end spot as a vertical line.
        plt.plot([yarax_real]*181, range(181), 'k-', lw=1)
        plt.annotate("Yarax", xy=(yarax_real,6), color='black', fontsize=9)
        # Plot the real normal analogy end spot as a vertical line:
        plt.plot([normal_real]*181, range(181), 'k-', lw=1)
        plt.annotate("Normal", xy=(normal_real,3), color='black', fontsize=9)
        # Plot the actual graphs:
        for w in indecies: # Plot curves
            plt.plot(range(360), graphs[w], color_tags[w % len(color_tags)],
                     linewidth=1)
        for w in indecies: #Plot labels
            plt.annotate(words[w], xy=(angles[w], graphs[w][angles[w]] + move),
                         color=colors[w % len(colors)], fontsize=9)
        #plt.annotate(words[0], xy=(angles[0], graphs[0][angles[0]] + 1),
        #             color=colors[0], fontsize=9) # For the first word again.
        # Fixed boundaries:
        if pisa != "none":
            plt.ylim(0, 5)
        else: plt.ylim(0, 180)
        plt.xlim(0,359)
        plt.show()
