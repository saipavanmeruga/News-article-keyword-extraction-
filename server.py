# Import neccessary packages, modules and files
from nc_extraction import KeywordsExtract
from keywords_post_process import KeywordsPostProcessor
from utils import NewsClean
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

#Initializing the flask app
app = Flask(__name__)

CORS(app) # Enabling Cross Origin Resource Sharing

news_cleaner = NewsClean() #Initializing NewsClean() class this will help to clean the news articles

keywords_extractor = KeywordsExtract() # Initializing KeywordsExtractor() class this will help to extract the keywords from articles

# app.route decarator to declare the route and the functionality of the api in that route # Here  only Post method is used.
@app.route('/extract_keywords',methods = ['POST']) 
def extract_keywords():
        """This function includes functionality of the "/extract_keywords" route
        Given input is json request object which consists of the list of news articles it will return a json 
        response object of dictionary conisting the keywords.
        Once news articles are given in this function uses methods from NewsClean() class and KeywordsExtract() class
        to clean and extract the keywords respectively.
        Input:
            request {object} -- consists of list of strings of news articles
        Returns:
            response {object} -- consists of nested dictionary of keywords extracted `(matched_word, similarity score)`
        """
        
        query_list = request.json.get("data") 
        processed_news = []
        t0 = time.time()
        for article in query_list:
            clean_article = news_cleaner.clean_news_article(article)
            keywords = keywords_extractor.extract_keywords(clean_article)
            processed_news.append(keywords)
        t1 = time.time()
        print("number of articles {}, keywords extraction time {}s".format(len(query_list), t1-t0))
        if processed_news is None:
            return jsonify('No keywords extracted')

        kpp = KeywordsPostProcessor()
        res_dict = kpp.get_top_keywords_from_articles(processed_news)
        print(res_dict)
        return jsonify({"noun_chunks": res_dict})


if __name__=='__main__':
    app.run() # Running the flask on local host