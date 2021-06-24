import nltk
import pandas as pd


class KeywordsPostProcessor:
    """
        This class used to post process the keywords obtained from KeywordExtract() class
        by checking if the keywords do not contain any duplicates and also they belong NNP,NN tags
    """
    def __init__(self):
        pass

    def pos_taggers(self, df):
        """
            This method to tag the words in the keywords dataframe.
            Arguements:
                df {DataFrame} -- dataframe obtained from keywords dictionary
            Returns:
                p1,p2 {list} -- Two lists consisting of keywords and their POS tag information
        """
        p2 = []
        post_process = df['Keyword'].tolist()    
        p1 = nltk.pos_tag(post_process)
        for i in post_process:
            p2.append(nltk.pos_tag([i]))
        return p1,p2
    def combine_pos_score(self, word):
        """
            This method is used to assign POS score based on POS tag of the word
            Arguments:
                word {tuple} -- Keyword tuple with POS tag inofrmation
            Returns:
                {int} -- POS score
        """
        if word[1] == 'NNP':
            return 5
        elif word[1] == 'NN':
            return 2
        else:
            return 1
    def specific_pos_score(self, word):
        """
            This method is used to assign POS score based on POS tag of the word
            Arguments:
                word {list} -- Keyword list with POS tag inofrmation
            Returns:
                {int} -- POS score
        """
        if word[0][1] == 'NNP':
            return 5
        if word[0][1] == 'NN':
            return 2
        else:
            return 1
    def remove_repeat_words(self, string):
        """
            This method is used to remove duplicate values within the string
            Example: "B2BGateway B2BGateway" --> "B2BGateway"
            Arguments:
                string {str} -- Keyword
            Returns:
                result {str} -- Keyword 
        """
        seen = set()
        result = []
        for item in string.split(' '):
            if item not in seen:
                if nltk.pos_tag([item])[0][1] in ['NNS', 'NNPS']:
                    item = item.lower().rstrip('s')
                    item = item.title()
                if nltk.pos_tag([item])[0][1] in ['NN','NNP', 'CD']:
                    if item.islower():
                        seen.add(item)
                        continue
                    seen.add(item)
                    item = item.title()
                    result.append(item)

        if len(result) <= 1:
            return None
        result = ' '.join(map(str, result))
        return result


    def get_top_keywords_from_articles(self, kwords_list):
        """
            This method is a wrapper of post processing functions and sort the keywords for Top 10 keywords.
            Arguments:
                kwords_list {list} -- List of dictionaries containing keywords, weight, label returned from KeywordExtract Process.
            Returns:
                response_dict {dict} -- Dictionary consisting of top 10 keywords
        """
        _all_keywords = []
        for a in kwords_list:
            if a != []:
                for w in a:
                    _all_keywords.append([w['keyword'],w['weight'],w['label']])
        _df_g  = pd.DataFrame(_all_keywords, columns=["Keyword", "Count","Label"])
        _df_g.sort_values(by="Count", inplace=True, ascending=False)
        _df_g.reset_index(drop=True, inplace=True)
        _df_g.to_csv('test.csv')
        print(len(_df_g))

        _df_g['Keyword'] = _df_g['Keyword'].apply(self.remove_repeat_words)
        _df_g.dropna(axis=0, inplace=True)
        p1,p2 = self.pos_taggers(_df_g)
        _df_g['c_POS'] = p1
        _df_g['s_POS'] = p2
        _df_g['c_POS_score'] = _df_g['c_POS'].apply(self.combine_pos_score)
        _df_g['s_POS_score'] = _df_g['s_POS'].apply(self.specific_pos_score)
        _df_g['Count'] = _df_g['Count'] + _df_g['c_POS_score'] + _df_g['s_POS_score'] 
        print(len(_df_g))
        _df_g.sort_values(by='Count',inplace=True, ascending=False)
        print(len(_df_g))
        _df_g = _df_g.reset_index(drop=True)
        _df_g = _df_g[:10]
        response_dict = dict()
        response_dict['nc'] = ", ".join(_df_g['Keyword'].to_list())
        return response_dict