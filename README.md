# multinomial-naivebayes-IMDBreview
 
### Download Dataset 
This algorithm is built with the help of a [dataset](https://www.kaggle.com/purvitsharma/imdb-sentiment-analysis-90-accuracy)

## What is this?
Multinomial-Naive Bayes is commonly used in NLP prediction and classification of text, but sometimes sentiment analysis. According to the article [1], the algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output. 

#### Originally, the dataset is created for Linear Support Vector Classifier, but I implement Multi Nomial Naive Bayes to see a different result. Currently the algorithm have 94% accuracy rate.

## The Project
The ML algorithm itself is to classifiy whether the opinion or sentiment of a specific comment is **POSITIVE or NEGATIVE**, with the use of the given dataset. The algorithm purely used python, multi-nomial naive bayes and other libraries such as numpy, sklearn, pandas, etc.

## Usage:
pip install jupyter notebook <br />
-clone the code <br />
-input the movie review sentences in driver code <br />

You can use any dataset by using pandas before using the driver code below <br />

#read dataset <br />
df=pd.read_csv("datasets/IMDB Dataset.csv" ,  names=['review','sentiment']) <br />

#output <br />
movie_reviews_array=np.array(["So bad, it is so bad movie"]) <br />
movie_review_vector = vectorizer.transform(movie_reviews_array) <br />
print (clf.predict(movie_review_vector))<br />

### Reference
[1](https://www.upgrad.com/blog/multinomial-naive-bayes-explained) <br />

If you want to upgrade, revise or you have questioons, just drop a message and I am happy to assist you.


