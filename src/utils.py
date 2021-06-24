import os
import json
import re
import unicodedata
import nltk

class DataLoader:
    """ The main use of this class to load the data from a given data folder during it's initialization. 
    The json format files in the folder are converted to list of strings containting the text of the news articles.
    Arguments:
            data_folder_path {str} -- base folder path
    
    """
    def __init__(self, root_folder_name):
        """This is the constructor method and takes in the argument of base data folder that is passed during the class initialization
        and returns list of all the json filenames.
        Arguments:
            root_folder_name {str} -- base folder path
        Returns:
            article_data     {list} -- list of strings of json filenames
        
        Raises: `FileNotFoundError`: If the path is correct.
        """

        self.data_folder_path = root_folder_name
        
        self.train_file_paths = [os.path.join(self.data_folder_path, filename) for filename in os.listdir(self.data_folder_path) 
                        if os.path.isfile(os.path.join(self.data_folder_path, filename)) if filename.endswith('.json')]
    
    def number_of_files(self):
        """This function returns the number files present in the base data folder.
            Returns:
            {int} -- number of json files in the folder.
        """
        return len(self.train_file_paths)
    
    def get_filecontent(self):
        """This function extract the news article in a string format from json file.
            Returns:
                {list} -- list of strings of news atricles
        """
        article_data = []
        for file in self.train_file_paths:
            with open(file, 'rb') as f:
                news = json.load(f)
                article_data.append(news['text'])
        return article_data

class NewsClean:
    """
    This class allows to apply some text preprocessing techniques like removing emails, urls, punctuations and text containing 
    certain patterns to make the text more cleaner for the further steps.
    """
    def __init__(self):
        """Assigning some attributes to the class during the initialization. """
        self.non_en_chars = {
        "’": "'",
        "‘": "'"}
        self.remove_from_title = ["BREAKING:", "[\-:] report", "[\-:] sources",  "[\-:] source", "source says", "Exclusive\:",
                         "Factbox\:", "Timeline[\-:]",  "Instant View\:", "Explainer\:", ": Bloomberg",
                         ": WSJ","Thomson Reuters journalist","/PRNewswire/","Calif.","Published on"]
        self.remove_if_start_with = ['close breaking news']
        self.replace_if_contain = ['click here to see']
    
    def clean_news_article(self, content):
        """
        Cleans the news article by applying available preprocessing functions

        Arguments:
            content {str}: string of content of news article
        Return:
            _clean_content_ {str} : string of clean content of news article after preprocessing
        """
        _clean_content_ = self.clean_content(content)
        return _clean_content_
    def remove_urls(self, text):
        """
        Remove urls from the string using the regular expressions `re` package.

        Arguments:
            text {str}: string of the news containing urls
        
        Returns:
            text {str}: string of news with urls removed
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text  = url_pattern.sub(r'', text)
        return text
    def remove_emails(self, text):
        """
        Remove urls from the string using the regular expressions `re` package.

        Arguments:
            text {str}: string of the news containing email ids
        
        Returns:
            text {str}: string of news with email ids removed
        """
        email_add = re.compile('[.\w]{3,}@[.\w]{5,}')
        text = email_add.sub(r'', text)
        return text

    def remove_punctuation(self, words):
        """
        Remove punctuation from list of tokenized words
        
        Arguments:
            words {list}: list of the words with punctuation
        
        Returns:
            words {list}: list of the words without punctuation
        """
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def remove_non_ascii(self, words):
        """
        Remove non-ASCII characters from list of tokenized words using unicodedata package

        Argumnents:
            words {list}: list of words containing accented words and non-ASCII characters
        
        Returns:
            words {list}: list of words without accented words and non-ASCII characters
        """
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    def clean_content(self, raw_content):
        """
        This is a wrapper functions which wraps all the functions to convert the raw_content into clean text
        Argumnents:
            raw_content {str}: raw content string
        
        Returns:
            clean_text {str}: clean string
        """
        if not raw_content:
            return ""
        remove_if_start_with = self.remove_if_start_with
        replace_if_contain = self.replace_if_contain
        txt = self.remove_urls(raw_content)
        txt = self.remove_emails(txt)
        txt = re.sub("'s",'', txt)
        p = "|".join(self.remove_from_title)
        txt = re.sub(p,'', txt)
        txt = re.sub('Image \d of \d FILE','', txt)
        txt = re.sub('AP Image \d of / \d Caption','',txt)
        txt = re.sub('Follow us on Twitter','',txt)

        # using nltk package to word tokenize the string and apply the remaining functions word-wise
        txt = nltk.word_tokenize(txt) 

        txt = self.remove_non_ascii(txt)

        txt = self.remove_punctuation(txt)

        txt = ' '.join(txt)
        
        lines = txt.split("\n")
        # remove short sentence
        clean1 = [l for l in lines if len(l) > 30]
        clean2 = [l for l in clean1 if len(l.split(" ")) > 3]
        # remove the words if the sentence starts with certain pattern
        p_start = r"^ *" + r"|^ *".join(remove_if_start_with)
        clean3 = [l for l in clean2 if re.match(p_start, l) is None]
        # remove the words if the sentence contains certain pattern
        p_contain = r"|".join(replace_if_contain)
        clean4 = [re.sub(p_contain, '', l) for l in clean3]
        clean5 = " ".join(clean4)

        clean_text = re.sub(r'\s+', ' ', clean5).strip()
        return clean_text

