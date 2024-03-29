== Python Version ==

I'm using Python Version 3.6.6 for this assignment.

== General Notes about this assignment ==

### Indexing Algorithm
The algorithm loops through all the files to index them. 

Every document is read line by line, and passed through the nltk
sentence and word tokenisers. Within a document, the frequencies of
each term are tabulated and saved in a vector. The vector is then
converted to a weighted-tf scheme using the ln-scheme. 

A sample vector is as follows:
{ 	"egg" 	: ln(2) ,
	"bacon"	: ln(1) }, where ln calculates the weighted-tf value.

This vector is iterated through to save the corresponding
term-docID-termFrequency information in the posting list, as well as
term-termID-documentFrequency data in the dictionary. The vector is
also used to generate the document length using the cosine scheme.

The posting list is represented as a document, since document order
is not as relevant (we are only keen in documents containing the
query terms), and also because there is a need to store the term
frequency with the docID.

When all the documents have been processed, the total document count
and corresponding document frequencies of each term in the dictionary
are known. The idf is calculated and saved back to the dictionary.

### Search Algorithm

The search algorithm first normalises the query terms to the ln-scheme
vector as shown above. This shortened query, which has collapsed
similar terms together in the vector, is iterated.

For each query term, the corresponding posting list (represented as
a python dictionary) is retrieved. If the term idf is zero, then
either the term does not exist in the corpus, or that is appears in
every document. Such terms are ignored because they do not contribute
to understanding document relevance.

For the rest of the query terms, the weight of that particular query
term is generated by multiplying the weighted-tf with the term-idf.
The documents in the posting list are the iterated through, generating
the per-term-per-docID score by multiplying the query weight and the
document weight together. These scores are added to the corresponding
score tracker for that docID.

Once all the query terms have been processed, we perform a final
normalisation of document scores using the document length, in effect
performing a cosine similarity (dot product in earlier stage) ranking.

A max heap is then used to retrieve the top ten entries to be returned.

### Saving to Disk

Compared to the previous hw2, an improvement module is the use of
multiple pickle dumps and loads. The pickle module has its own
delimiter for retrieving the dumped bytes. The posting file is now
a series of posting list dumps, with their initial byte offset
recorded for retrieval.

### Essay Questions

1. In this assignment, we didn't ask you to support phrasal queries,
which is a feature that is typically supported in web search engines.
Describe how you would support phrasal search in conjunction with the
VSM model.

## METHOD A: SUPPORT PHRASAL INDEXING
Currently, there is no record of how many query terms a document
contains. A naive solution is to do context-specific indexing. It is
possible to directly have a list of common phrases in economics, social
policies, names of politicians which can be used in a similar fashion
as stopwords. These phrases could be recorded at indexing time. 

However, it is often difficult to predict the sheer multitude of
possible phrasal searches which could occur, unless the context is
specific (say, phrasal medical terms). It would then make more sense
to explore solutions which rank documents different at search time.

## METHOD B: MERGE POSTING LISTS, THEN CALCULATE SCORES
One way is to first merge the posting lists for all the query terms.
This approach is equivalent to a boolean query accumulating the query
terms with an "AND" operation. The score is then only calculated for
the documents in the final posting list. The major flaw of such an
approach is the intolerant retrieval, because only full matches are
classified.

Should there be insufficient documents (< 10) with the exact phrase
match, then shorter phrasal queries on the biwords
("annual car" and "car sales") can be spawned. A new challenge is
also introduced which is the relative importance of biwords, and which
biword search results should be returned. Metrics like the bigram
overlap can be used to determine which documents should be scored.

2. Describe how your search engine reacts to long documents and long
queries as compared to short documents and queries. Is the
normalization you use sufficient to address the problems
(see Section 6.4.4 for a hint)?

The lnc-ltc scheme uses a cosine normalisation of the document vectors
using the Euclidean length of the vectors.

Long documents are likely to contain more distinct terms, and a 
higher frequency of the terms. News articles in the reuters data set
can be classified as non-verbose documents, that is, it is unlikely
for the same content to be repeated across the document. Instead, they
are likely to contain a lot of unique terms and ideas to convey the
news to readers.

When performing the cosine normalisation, the effect of the long
document length is damped. Relative to short documents, the Euclidean
length of long documents is not going to be very much different. As a
result, long documents have a higher score. We can consider a simple
example below:

doc1: egg egg egg
doc2: egg egg egg cat cat cat cat	(7 words)
doc3: i had an egg for breakfast today 	(7 words)
length (doc1) = 3
length (doc2) = sqrt(square(3) + square(4)) = 5
length (doc3) = sqrt(7) = 2.65

We observe that unique terms significantly reduce the document length,
skewing the scores of non-verbose documents higher. This observation
is pertinent in understanding our reuters dataset, because the news
articles can be fairly inconsistent. Document 1 is a fairly long
article with a lot of unique terms, while document 7785 is a short
record that repeats itself.

The positive skewing on long, non-verbose documents and the existence
of an inconsistency in the quality of data suggests that the lnc-ltc
scheme may not be sufficient. We can compensate for the effect of
document length by changing our normalisation method from Euclidean
normalisation to Pivoted Document Length Normalisation.

In your judgement, is the ltc.lnc scheme (n.b., not the ranking scheme
you were asked to implement) sufficient for retrieving documents from
the Reuters-21578 collection?

There are two primary differences with using ltc-lnc vs. lnc-ltc.
A. Repeated calculation of idf
For every tf, the idf is multiplied. This has essentially no effect on
the scalar multiplication in cosine similarity.

score(t, d) = l-scheme(tf(t, d)) * l-scheme(tf(t, q)) * t-scheme(t)

However, a lot of extra work is done at indexing stage to first
calculate l-scheme(tf(t, d)) * t-scheme(t). 

It would be computationally more efficient to calculate
w(t, q) = l-scheme(tf(t, q)) * t-scheme(t) for the query term, then
multiple this value to w(t, d) for the respective document weights.

B. Document length
The idf of the terms are now factored in to document length. For
example, consider this corpus of two documents:

doc1: the the the cat
doc2: the the the the cat

idf(the) = 0
idf(cat) = 0.3

tf(doc1, the) = 1 + log(3) = 1.5
tf(doc1, cat) = 1 + log(1) = 1
tf(doc2, the) = 1 + log(4) = 1.6
tf(doc2, cat) = 1 + log(1) = 1

(lnc-ltc): Length of doc1 = sqrt(sq(1.5) + sq(1)) = 1.8
(lnc-ltc): Length of doc2 = sqrt(sq(1.6) + sq(1)) = 1.9
(ltc-lnc): Length of doc1 = 0
(ltc-lnc): Length of doc2 = 0

Documents which purely contain common terms cannot be differentiated
from each other as their lengths will all be normalised to 0. This
situation is not very problematic for the reuters data set because
news articles are likely to contain unique terms and references.

3. Do you think zone or field parametric indices would be useful for
practical search in the Reuters collection? Note: the Reuters
collection does have metadata for each article but the quality of the
metadata is not uniform, nor are the metadata classifications
uniformly applied (some documents have it, some don't).

It is unlikely that zone and parametric data is useful in the Reuters
collection. More often than not, users want to look up news articles
based on the content, rather than author.

However, there is a benefit to indexing by year (vis-a-vis date),
since users may want to look up news articles from a period in time.
It is also unlikely that users will remember and lookup the exact date
of the articles, else they would just go to an archive and not a
search engine.

Nevertheless, because the metadata quality and availability is not
uniform, introducing parametric indices would create a bias towards
articles that contain the metadata. The results from such a search may
be skewed and not ideal, but it is indeed better than having a lack of
any such information.

== Files included with this submission ==

index.py -- program to index the corpus
search.py -- program to process queries
dictionary.txt -- dictionary of terms mapped to frequencies and offsets
postings.txt -- length information, postings list
stopwords -- stopwords to be removed

== Statement of individual work ==

[X] I, JOYCE YEO SHUHUI, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

== References ==
# nlargest for heap
	https://docs.python.org/3/library/heapq.html
# Verifying query results
	CHOW KENG JI + SOH JASON
	We discovered a mistake in my use of Porter Stemmer.
	Case-folding should be applied before stemming.
