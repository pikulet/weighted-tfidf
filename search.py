#!/usr/bin/python
import re
import nltk
import sys
import getopt
import pickle
import index
import operator
import heapq

from math import log, sqrt
from index import normalise_term, END_LINE_MARKER

########################### DEFINE CONSTANTS ########################### 
NUM_RANKED_ENTRIES = 10
TF_MARKER = ","

######################## COMMAND LINE ARGUMENTS ########################

### Read in the input files as command-line arguments
###
def read_files():
    dictionary_file = postings_file = file_of_queries = file_of_output = None

    def usage():
        print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")
	
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file  = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
        usage()
        sys.exit(2)

    return dictionary_file, postings_file, file_of_queries, file_of_output

######################## FILE READING FUNCTIONS ########################

### Retrieve a dictionary mapping docIDs to normalised document lengths
###
def get_lengths():
    postings_file.seek(0)
    length_dict = pickle.load(postings_file)
    return length_dict

### Retrieve a dictionary format given the dictionary file
###
def get_dictionary():
    with open(dictionary_file, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary

### Retrieve all the queries which need to be processed
###
def get_queries():
    with open(file_of_queries, 'r') as f:
        queries = f.read().splitlines()
    return queries

### Retrieve the posting list for a particular dictionary term
###
def get_posting_list(t):
    try:
        offset, idf = dictionary[t]
        postings_file.seek(offset)
        data = pickle.load(postings_file)
        return idf, data
    except KeyError:
        # term does not exist in dictionary
        return 0, dict()

########################### QUERY PROCESSING ###########################

### Processing a query
###
def process_query(q):
    q = [normalise_term(t) for t in q.split()]
    q = dict(zip(q, map(lambda x: 1 + log(q.count(x), 10), q)))
    
    scores = dict()

    # linear combination of term scores
    for t in q:
        term_idf, posting = get_posting_list(t)
        if term_idf == 0:       # term does not exist, or appears in all documents
            continue

        query_weight = q[t] *  term_idf

        for docID, document_weight in posting.items():
            term_score = query_weight * document_weight
            
            try:
                scores[docID] += term_score
            except KeyError:
                scores[docID] = term_score

    # length normalisation
    for docID in scores:
        scores[docID] /= length[docID]

    # retrieve top entries using heapq (sort by score, then docID in increasing order)
    result = heapq.nlargest(NUM_RANKED_ENTRIES, scores, key=lambda x: (scores[x], -x))
    
    return result

dictionary_file, postings_file, file_of_queries, file_of_output = read_files()
postings_file = open(postings_file, 'rb') # kept open because of frequent use

length = get_lengths()
dictionary = get_dictionary()
queries = get_queries()

with open(file_of_output, 'w') as f:
    for q in queries:
        result = process_query(q)
        f.write(' '.join([str(x) for x in result]) + END_LINE_MARKER)
        
postings_file.close()
