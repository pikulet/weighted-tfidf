#!/usr/bin/python
import sys
import getopt
import nltk
from nltk.stem.porter import *
import os
import linecache
import pickle
from math import log, sqrt

########################### DEFINE CONSTANTS ###########################
PORTER_STEMMER = PorterStemmer()
END_LINE_MARKER = '\n'
STOP_WORD_REMOVAL = False
REMOVE_NUMBERS = False
STOP_WORD_FILE = 'stopwords'
STOP_WORDS = set()

######################## COMMAND LINE ARGUMENTS ########################

### Read in the input files as command-line arguments
###
def read_files():
    def usage():
        print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

    input_directory = output_file_dictionary = output_file_postings = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i': # input directory
            input_directory = a
        elif o == '-d': # dictionary file
            output_file_dictionary = a
        elif o == '-p': # postings file
            output_file_postings = a
        else:
            assert False, "unhandled option"

    if input_directory == None or output_file_postings == None or output_file_dictionary == None:
        usage()
        sys.exit(2)

    return input_directory, output_file_dictionary, output_file_postings

############################## DOCUMENT INDEXING ##############################

### Driver function
###
def main():
    input_directory, output_file_dictionary, output_file_postings = read_files()

    check_stop_words()
           
    dictionary = Dictionary(output_file_dictionary)
    postings = PostingList(output_file_postings)
    length = dict()
    
    # search directory for files
    all_files = [int(f) for f in os.listdir(input_directory)]   # dictionary used, sorting not necessary

    for docID in all_files:
        process_document(input_directory, dictionary, postings, length, docID)
        
    # calculate idf using the t-scheme
    convert_idf(dictionary, len(all_files))
    
    # save the postings to the disk, update offset for each term in dictionary
    postings.save_to_disk(length, dictionary)
    dictionary.save_to_disk()

### Process a particular document
###
def process_document(input_directory, dictionary, postings, length, docID):
    file_name = str(docID)
    file_location = os.path.join(input_directory, file_name)

    vector = dict()                 # term vector for this document
    line_num = 1                    # start at the first line
        
    # retrieve data on first line
    line = linecache.getline(file_location, line_num)

    while len(line) != 0:   # iterate through whole document until empty line is encountered
        # add data information
        add_vector_count(line, vector)            
        line_num += 1
        line = linecache.getline(file_location, line_num)

    # Calculate tf using the l-scheme
    convert_tf(vector)
    
    length[docID] = get_length(vector)
    build_list(dictionary, postings, docID, vector)

### Process the line, add the term count to the vector
###
def add_vector_count(line, vector):
    sentences = nltk.sent_tokenize(line)            # split line into sentences

    for s in sentences:
        words = nltk.word_tokenize(s)               # split sentences into words
        for w in words:
            if STOP_WORD_REMOVAL and w in STOP_WORDS:   # ignore stop words
                continue
            if REMOVE_NUMBERS and hasNumbers(w):        # ignore numbers
                continue

            t = normalise_term(w)
            
            try:
                vector[t] += 1
            except KeyError:
                vector[t] = 1

### Add the data to our dictionary and postings record.
###
def build_list(dictionary, postings, docID, vector):  
    for t, tf in vector.items():
            
        # add data to dictionary and postings
        if dictionary.has_term(t):
            dictionary.add_frequency(t)

            termID = dictionary.get_termID(t)
            postings.add_docID_to_posting(docID, termID, tf)    # create posting list entry

        else:
            # term currently does not exist in dictionary
            termID = postings.add_new_term(docID, tf)           # returns index of added set, which is the new termID
            dictionary.add_term(t, termID)
            
########################## TERM NORMALISATION MODULE ##########################

### Checks for stop words, and retrieves them if necessary.
###
def check_stop_words():
    # retrieve stop words
    global STOP_WORDS    
    if STOP_WORD_REMOVAL:
        with open(STOP_WORD_FILE, 'r') as f:
            STOP_WORDS = set(f.read().splitlines())
            
### Checks if a term has digits
###
def has_numbers(t):
    return any(char.isdigit() for char in t)

### Normalise a token to standard form
### Here, we apply the porter stemmer and case folding
###
def normalise_term(t):
    return PORTER_STEMMER.stem(t.lower())

########################## COSINE SIMILARITY MODULE ##########################
            
### Given a posting list, map all the tf(t, d) to w(t, d) using the l-scheme
### w(t, d) = 1 + log(tf(t, d))
###
def convert_tf(vector):
    for t, tf in vector.items():
        vector[t] = 1 + log(tf, 10)

### Calculate the length of a vector, represented as a list
###
def get_length(vector):
    return sqrt(sum(map(lambda x: x**2, vector.values())))

### Given a dictionary, map the document frequencies to inverse document frequencies using the t-scheme
### idf(t) = log(N/ df(t))
###
def convert_idf(dictionary, n):
    for term, values in dictionary.terms.items():
        df = values[1]
        idf = log(n/df)
        dictionary.set_idf(term, idf)

########################## HELPER DATA STRUCTURES ##########################

### A dictionary class that keeps track of terms --> termID/ term_offset, document_frequency/ idf
### termID is a sequential value to access the posting list of the term at indexing time
### term_offset is the exact position (in bytes) of the term posting list in postings.txt
###
class Dictionary():
    def __init__(self, file):
        self.terms = {} # every term maps to a tuple of termID/ term_offset, document_frequency/idf
        self.file = file

    def add_term(self, t, termID):
        self.terms[t] = [termID, 1]

    def has_term(self, t):
        return t in self.terms

    def get_terms(self):
        return self.terms

    def add_frequency(self, t):
        self.terms[t][1] += 1

    def get_termID(self, t):
        return self.terms[t][0]

    def set_idf(self, t, idf):
        self.terms[t][1] = idf

    def set_offset(self, t, offset):
        self.terms[t][0] = offset

    def save_to_disk(self):
        with open(self.file, 'wb') as f:
            pickle.dump(self.terms, f)

### A Postings class that collects all the posting lists.
### Each posting list is a dictionary mapping docIDs to term frequencies
###
class PostingList():
    def __init__(self, file):
        self.file = file
        self.postings = []
        self.currentID = -1

    def add_new_term(self, docID, tf):
        new_posting = { docID : tf }    # new term posting with one entry
        self.postings.append(new_posting)

        self.currentID += 1
        return self.currentID       # return the index of the new posting list (termID)

    # Create a new entry in the posting list
    def add_docID_to_posting(self, docID, termID, tf):
        posting = self.postings[termID]
        posting[docID] = tf
            
    # Saves the posting lists to file, and update offset value in the dictionary
    def save_to_disk(self, length, dictionary):        
        with open(self.file, 'wb') as f:
            pickle.dump(length, f)

            for t in dictionary.get_terms():
                termID = dictionary.get_termID(t)
                posting = self.postings[termID]

                dictionary.set_offset(t, f.tell())  # update dictionary with current byte offset
                pickle.dump(posting, f)

if __name__ == "__main__":
    main()
