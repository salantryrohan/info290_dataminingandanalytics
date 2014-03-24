INFO 290: DATA MINING AND ANALYSIS FINAL REPORT

Sayantan Mukhopadhyay
AJ Renold
Rohan Salantry
 
Problem:
 
For our final project we are working with Yelp Academic Dataset published for Kaggle Data Challenge and we are trying to predict how many ‘Useful’ votes a particular user review will get based on the supplied characteristics of the data. Given that users tend to use the reviews as the source of information before making their choice to conduct business with certain business, it is a growing challenge for users and Yelp to figure out which reviews are most meaningful for the users. As reviews can be voted to be ‘Useful’ ‘Funny’ and ‘Cool’ by users, we are using the ‘Useful’ review as the indicator to signify the importance of the review. Apart from helping to decide the ordering of the current reviews based in usefulness it’s also important to position a new review based on a predicted ‘usefulness’.
 
Challenges:
 
The main challenges we found while making use of the Review Text data to get meaningful numeric results are:
1.     The user text is very unstructured.
2.     Most of the reviews (around 70%) did not have a single ‘Useful’ vote.
3.     The reviews with or without Useful votes had very similar features.
4.  Some Business Categories have a small number of reviews (less than 100)
 
Steps 

Preprocessing
Text Preprocessing
Unigram tokens - using Regular Expressions we scrape out punctuation and numbers from the text, replacing them with spaces, then we split the text by spaces and the remaining words are used as tokens for classification.
Stop words - we tried three methods of removing stop words from our documents:
We use a stop-words-english3-google.txt to create a look-up dictionary and we remove words from documents during tokenization if they are in the dictionary
We extract stop words from our Naive Bayes classifier based on class probability. Based on class probability, our algorithm takes the intersection of the n most common words per class and uses these as stop words. This step happens after training and is used for the labeling step. We have sample output of this step in our Appendix Section a.
We remove names from the documents by creating a look-up dictionary based on the first and last names in the U.S. Census Bureau’s list of names from the 1990 census, the names are removed during tokenization if they are in the look-up dictionary
Adding Meaning to Text
price_mention: We have found in the data corpus that one of the meaningful features is the mention of price in a review text. We noticed that the use of $ indicated a word was a price, so we replaced words containing $ with the token ‘price_mention’.
web_url_mention: We have found in the data corpus that another meaningful features is web addresses. We use a Regular Expression to find web urls and we replace them with the token ‘web_url_mention’
review_len token: Also we have added a feature in each review based on the length of the review. We created 10 equal sized buckets between 100 and 1,000 words and one bucket above 1,000 words and add a token to the review text such as ‘review_len_100’
User Data
average_useful_votes: Votes received by the previous reviews written by the user is another important feature we decided to add. For that we made seven tokens based on average votes per reivew and added these tokens to the review text that a user authored. We increased the poor and high average token frequencies based on experimentation to emphasize users with poor reviews and highly valued reviews. 
Training Data Selection
Per Business Category
Number of Reviews in a Category: Each review is written for a business and the business belongs to a number of separate categories. e.g., a restaurant can belong to categories such as ‘foods’, ‘restaurants’, ‘lunch and breakfast’, and ‘bar’ for example. We separated the training data by category and trained a classifier with every category review for each category.
Multinomial Naive Bayes
For Bayes model training we are training the model with reviews if the number of Useful vote=0 or >=3. The reason for doing so is we think training the models with real ‘Useful’ (reviews with more than 3 votes) and with real ‘Not Useful’ (reviews with only 0 votes) reviews will help the model to classify the reviews better. We excluded the reviews with 1 or 2 useful votes because they created too much noise in the vocabulary for the classifier.
Linear Regression
For Linear Regression we only took the Reviews with useful vote >= 1. Because we are using Linear Regression to predict number of votes after Bayes has given a class label, we are training Linear Regression with only positive data, or reviews with 1 or more useful votes.
Model Training
Multinomial Naive Bayes
During the training of the Bayes classifier, we remove words from the vocabulary with total frequency less than 60. 
We also generate a list of stop words based on class probability, that are excluded during the labeling process, this was previous mentioned in preprocessing steps
Linear Regression
Because Linear Regression is sensitive to using a large number of attributes, we use a max entropy function we developed to extract a vocabulary for Linear Regression from our Bayes classifier. This generates 1,000 words maximizing the class probability difference, we also remove words from this list based on Part of Speech tagging.
After that we do Part of Speech removal leaving Noun, Verb, Adjective and Adverbs intact.
Labeling
Per Review Business Category
Labeling is done for  each Business Category of the business each review is written about, for example a business with categories “Restaurants”, “Food”, and “Mexican”, would be labeled by a classifier trained on each category seperatly.
The Bayes Classifier labels a review with the binary class labels, either ‘Useful’ or ‘Not Useful’
If classified as ‘Useful’ then we use the Linear Regression model to calculate the predicted number of Useful votes.
If classified as ‘Not Useful’ then number of votes are 0.
Ensemble
After calculating votes per category; if a business received ‘Useful’ votes in 25% or more categories then use it for final prediction.
The final prediction is calculated by averaging number of ‘Useful’ votes a review has received for all the categories.
 
Model Performance
We present our model performance by showing our base performance and then layering on features to the classifier which we believe increased performance.

Error Measurement = RMSE (Root Mean Square Error)

Base Parameters = Categories with more than 50 reviews, Per category Bayes (unigram token) and LR classifiers, 1000 word vocab extracted for LR vocabulary
Base Performance: 2.553
Feature - Stop Words from stop-words-english3-google.txt 
Performance: 2.559
Feature - Stop Names from 1990 U.S. Census
Performance: 3.149
Feature - Stop Words during labeling from 60 most common
Performance: 3.217
Feature - price_mention during tokenization
Performance: 3.146
Feature - review_len token
Performance: 2.732
Feature - web_url token
Performance: 2.964
Feature - outlier removal of useful_votes > 14
Performance: 2.369
Feature - user_avg token
Performance: 2.251
Feature - remove words of min_frequency 60 
Performance: 1.805
Feature - max_vocab for Bayes based on 2000 words maximizing entropy
Performance: 1.778

Experiments 
These were steps we experimented with, but opted not to use because they did not improve model performance or because they significantly decreasing the training and labeling time. 

Preprocessing
Word Correction - We used a word correct algorithm by Peter Norvig.  The algorithm uses a data set which is a concatenation of public domain books from Project Gutenberg and the British National corpus.  It uses edit distance function which modifies the misspelt word to give a list of probable words and selects words that occur most in the word corpus. We opted to exclude this from our training and labeling because it significantly slowed the application.
Collapsed word spellings - using a Regular Expression to collapse words with repeated letters to a single letter. For example, “mmmm” would be collapsed to “m” or “mississippi” would be collapse to “misisipi”. We found that this did not improve the performance of our 
Stemming and Part of Speech tagging with NLTK - using these NLP techniques to clean the words in a document. We found that both of these features of the NLTK library were extremely slow and affected the performance of our training and labeling
Training Data Selection
TFIDF(Term Frequency Inverse Document Frequency)  - We tried using TFIDF for both stopword removal and feature extraction. To determine features , we took percentile range of tdidf weights for each review and took a union of features corresponding to those weights.
Entropy - Our goal was to identify features that help identify what made a review useful. We labelled our data as useful and not useful based on useful vote count and calculated entropy for all words in the review text.  We used top N words for extracting features and determine vocabulary for Linear Regression Model. We varied the value of N while inspecting rmse to find the ideal value of N.
Model Training
Naive Bayes - Resampling to create balanced classes - we hypothesized that one solution to having highly unbalanced classes in a Bayesian classifier would be to find a method to create balanced classes. We tried randomly resampling smaller classes so that all classes would have the same amount of training tuples. This did not improve classification performance, so it was not used as in the algorithm.
Naive Bayes - Bigram tokenization - we tried bigram tokenization of documents, but it did not improve the classification performance.
Labeling
Ensemble Model Voting - We tried a number of minimum votes to signal classification agreement when positively labeling a document to be useful with the Bayes classifier. We settle on 25% as the best performing agreement percent allowing for a very loose agreement criteria.

Appendix

Section a. - Output of Naive Bayes Stop Words (50) for Category ‘Restaurants’
0 and
1 ive
2 because
3 ordered
4 dont
5 some
6 one
7 have
8 our
9 chicken
10 your
11 out
12 would
13 also
14 menu
15 there
16 had
17 its
18 their
19 only
20 which
21 time
22 got
23 really
24 we
25 get
26 food
27 after
28 here
29 if
30 they
31 not
32 eat
33 great
34 salad
35 restaurant
36 could
37 up
38 order
39 she
40 were
41 very
42 didnt
43 think
44 came

Section b. - From the max_entroy function the 20 words that maximize class probability difference for Category “Restaurants”
0 biz_photos
1 uye
2 gabi
3 lobbys
4 elite
5 http
6 panera
7 fez
8 review_len_100
9 palatte
10 review_len_long
11 review_len_1000
12 christophers
13 cous
14 pics
15 lumpia
16 review_len_800
17 fucking
18 review_len_900
19 effing

Section c. - Code

Our complete classification process presented in IPython Notebook - http://nbviewer.ipython.org/urls/raw.github.com/AJRenold/yelp_project/master/Ensemble_test_multicategory.ipynb
Our Multinomial Naive Bayes implementation - https://github.com/AJRenold/python_naive_bayes
Our Project Github Repo - https://github.com/AJRenold/yelp_project

Github User Names
www.github.com/AJRenold
https://github.com/tantanm
https://github.com/salantryrohan



