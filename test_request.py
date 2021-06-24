# Import required packages, modules and files
from utils import DataLoader
import pickle
import requests
if __name__ == "__main__":
    #path to root data folder consisting of news article files in json format
    data_folder_path = 'E:\\aidetic\\data\\tech_news' 
    
    # Initializing DataLoader class to load the json format files
    dl = DataLoader(data_folder_path)

    # The function get_filecontent() will read the json files from the folder and convert them into list of strings
    news_articles = dl.get_filecontent()

    # sending a POST request to server using requests package
    response_ = requests.post('http://127.0.0.1:5000/extract_keywords', json = {"data":news_articles})
    
    # saving the response object to a pickle file
    with open('reponse.pkl','wb') as file_:
        pickle.dump(response_, file_)