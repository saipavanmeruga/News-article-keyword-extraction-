
from __future__ import division
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import calendar as cd
import pandas as pd
import numpy as np
from cosine_similarity import calculate_cosine_similarity
from collections import defaultdict
import re

class KeywordsExtract:
    """
        This class helps to extract the keywords from the articles.
        It is initialized with certain attributes like MODEL which consists of the spacy model object.
    """
    def __init__(self):
        """
            Initializing attributes along with the class initialization.
        """
        self.MODEL = spacy.load('en_core_web_lg')
        self.allow_types = ['PERSON', 'GPE', 'ORG', 'NORP', 'LOC', 'FAC', 'WORK_OF_ART', 'EVENT', 'LAW', 'PRODUCT']
        self.remove_words = ['new', 'time', 'matter', 'source', 'people', 'story', 'reuters story']
        self.remove_entities = ['REUTERS', 'Reuters', 'Thomson Reuters', 'CNBC']
        self.months = [cd.month_name[i] for i in range(1, 13)] + [cd.month_abbr[i] for i in range(1, 13)]
        self.lookups = Lookups()
        self.lemma_keep = ['data']
        self.lemma_exc = self.MODEL.vocab.lookups.get_table("lemma_exc")
        for w in self.lemma_keep:
            del self.lemma_exc[self.MODEL.vocab.strings["noun"]][w]
        self.lookups.add_table("lemma_exc", self.lemma_exc)
        self.lookups.add_table("lemma_rules", self.MODEL.vocab.lookups.get_table("lemma_rules"))
        self.lookups.add_table("lemma_index", self.MODEL.vocab.lookups.get_table("lemma_index"))
        self.lemmatizer = Lemmatizer(self.lookups)
    def extract_keywords(self,content):
        """
            This function wraps all the functions that are available to extract keywords from the article.
            Arguments:
                content {str} -- cleaned content of the news article.
            Returns:
                keywords {list} -- list of keyword dictionaries containing keyword, score, weight, label
        """
        if len(content) <= 1:
            return []
        ner_doc = self.MODEL(content) # applying the spacy nlp model on the article  
        df_ent_grp, _ = self.extract_entities(ner_doc) # extracting entities 
        df_chunk_grp,_ = self.extract_noun_chunks(ner_doc) # extracting noun chunks
        df_keywords = self.rake_score(df_ent_grp,df_chunk_grp) # calculating RAKE score
        keywords = []
        if not df_keywords.empty:
            keywords = [{'keyword': str(r['entity']), 'score': int(r['r_score']), 'weight':int(r['weight']), 'label':str(r['label'])} for _, r in df_keywords.iterrows()]
        return keywords
    def extract_entities(self, ner_doc):
        """
            This method is used to extract entities using .ents_ method from the spacy module 
            The extracted entity is tested whether it belongs to the allowed entity types from 
            the allow_types attribute of the KeywordExtractor() class.
            Arguments:
                ner_doc {object} -- spacy object.
            Returns:
                df_grp {DataFrame} -- dataframe grouped by entity and similar entities merged with the help of cosine similarity.
                df {DataFrame} -- dataframe consisting of entities filtered by cleaning missclassified dates, special characters
                                    and also removing the entities from remove_entities list attribute of the class.
        """
        #extract the allowed entities from the given article and store them in a list
        ents = [ent for ent in ner_doc.ents if ent.label_ in self.allow_types]

        # checking and removing the words in the entity using pos tagger that belong to PDT,DT,CC,IN
        ents = [self.trim_tags(ent) for ent in ents]
        
        # removing entities which might me like '' empty strings
        ents = [ent for ent in ents if len(ent) > 0]

        # lemmatize any noun in plural form to singular for and get the labels of the entity again and store them in a list.
        ent_list = [[self.lemma_last_word(ent),ent.start_char,ent.end_char, ent.label_,ent.lemma_] for ent in ents]
        
        # converting the list of lists to dataframe for simple calculations
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(ent_list, columns=cols)
        
        # checking the entities for special characters and removing entitites which have length greater than 35 and also empty entities
        df = self.filter_keywords(df)

        df['ent_type'] = 'entity'
        df['weight'] = 1
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
        # the dataframe is then checked for missclassified dates and unwanted publisher names then the weight is 
        #assigned to each entity on the basis of 
        # the occurence in the article
        df_grp, df = self.filter_keywords_and_calculate_weight(df)
        
        if not df_grp.empty:
            # similarity between the keywords and merging them.
            df_grp = self.merge_keywords_by_similarity(df_grp)
        
        return df_grp,df
    def extract_noun_chunks(self, ner_document):
        """
            This method is used to extract noun chunks using .noun_chunks method from the spacy module. 
            The extracted nounchunks are then cleaned and filtered.
            Arguments:
                ner_doc {object} -- spacy object.
            Returns:
                df_grp {DataFrame} -- dataframe grouped by noun_chunks and similar noun_chunks are merged with the help of cosine similarity.
                df {DataFrame} -- dataframe consisting of noun chunks filtered by cleaning missclassified dates, special characters
                                    and also removing the noun_chunks from remove_entities list attribute of the class .
        """
        #extracting the noun chunks from the article string and storing in a list
        chunks = [ch for ch in list(ner_document.noun_chunks) if (ch.root.ent_type_ == '')]
        
        #checking for any empty noun chunks
        chunks = [ch for ch in chunks if len(ch) > 0]

        # checking and removing the words in the chunk using pos tagger that belong to PDT,DT,CC,IN
        chunks = [self.trim_tags(ch) for ch in chunks]

        # checking and removing the stopwords from the chunks
        chunks = [self.trim_stop_words(ch) for ch in chunks]

        #checking and removing the noun chunks using POS tags
        chunks = [self.trim_entities(ch) for ch in chunks]
        
        #noun chunks which are not empty
        chunks = [ch for ch in chunks if len(ch) > 0]
        
        #chunk information like start_character end_character chunk_label and chunk_lemma.
        chunks_list = [[self.lemma_last_word(ch), ch.start_char, ch.end_char, ch.label_, ch.lemma_] for ch in chunks]
        
        # converting the list of lists to dataframe for simple calculations
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(chunks_list, columns=cols)
        
        # checking the entities for special characters and removing entitites which have length greater than 35 and also empty entities
        df = self.filter_keywords(df)
        df['ent_type'] = 'noun_chunk'
        df['weight'] = 1
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
            df = df[(df['entity'].str.len() > 3) | df['entity'].str.isupper()]
        
        # the dataframe is then checked for missclassified dates and unwanted publisher names then the weight is 
        #assigned to each entity on the basis of 
        # the occurence in the article
        df_grp, df = self.filter_keywords_and_calculate_weight(df)
        
        if not df_grp.empty:
            # similarity between the keywords and merging them.
            df_grp = self.merge_keywords_by_similarity(df_grp)
        
        return df_grp, df

    def filter_keywords_and_calculate_weight(self, df):
        """
            In this method the dataframe is then checked for missclassified dates and unwanted publisher names then the weight is 
            assigned to each entity on the basis of the occurence in the article
            Arguments:
                df {DataFrame} -- dataframe consisting of entity text or noun chunks text with some special characters
            Returns:
                df_merge, df {DataFrame} -- a merged dataframe of with a weight column, a normal dataframe consisting of clean text
        """
        _months = self.months
        if not df.empty:
            # remove unwanted entities
            df = df[~df['entity'].isin(self.remove_entities)]
            # remove misclassified dates
            df = df[df['entity'].apply(lambda x: re.search("\s+\d+|".join(_months) + "\s+\d+", str(x)) is None)]
            if df.empty:
                return pd.DataFrame(), df
            df1 = df.groupby(['entity'])[['weight']].sum()
            df0 = df.drop_duplicates(subset=['entity']).copy()
            if "weight" in df0.columns:
                df0.drop(['weight'], axis=1, inplace=True)
            df_merge = pd.merge(df0, df1, left_on='entity', right_index=True)
        else:
            return pd.DataFrame(), df
        df_merge.sort_values(by=['weight', 'start'], inplace=True, ascending=[False, True])
        df_merge = df_merge.reset_index(drop=True)
        return df_merge, df
    
    def filter_keywords(self, df):
        """
            This method is used for checking the entities for special characters and 
            removing entitites which have length greater than 35 and also empty entities
            Arguments:
                df {DataFrame} -- dataframe consisting of entity text or noun chunks text with some special characters
            Returns:
                df {DataFrame} -- dataframe consisting of clean entity text or noun chunks text

        """
        if not df.empty:
            # remove special characters
            df['entity'] = df['entity'].apply(lambda x: re.sub('\.|[\-\'\$\/\\\*\+\|\^\#\@\~\`]{2,}', '', str(x)))
            df = df[df['entity'].apply(lambda x: re.search('\(|\)|\[|\]|\"|\:|\{|\}|\^|\*|\;|\~|\|', str(x)) is None)]
            # remove unwanted words
            if not df.empty:
                df = df[df['entity'].apply(lambda x: x.lower() not in self.remove_words)]
            # remove too long entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) < 35)]
            # remove too short entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) > 1)]
        return df
    
    def lemma_last_word(self, s):
        """
           This method is used to Lemmatize any noun in plural form to singular form by removing the trailing 's' in chunk text
           Arguments:
                s {object} -- entity or noun_chunk object
            Returns:
                txt {str} -- a string with clean chunk text with noun in singular form
        """
        if s[-1].tag_ in ['NNS', 'NNPS']:
            lemma = self.lemmatizer(s[-1].text, 'NOUN')[0]
#             print(lemma.title())
            txt = lemma.title() if s[-1].text.istitle() else lemma
            if len(s) > 1:
                txt = s[:-1].text + " " + txt
        else:
            txt = s.text
        return txt
    
    def trim_tags(self,s, for_type='chunk', trim_tags=['PDT', 'DT', 'IN', 'CC'], punctuation=[',', '\'']):
        """
            A method to check if there are any punctuation with help of re.search() in them or
            any words that fall under PDT,DT,IN,CC categories.
            Arguments:
                s {object} -- entity or noun_chunk object
            Default Arguments:
                for_type {string} -- this arguements checks for the entity_type -- deafault = 'chunk'
                trim_tags {list} -- consists of pos tags to be removed from chunk text
                punctuation {list} -- consists of punctuation to be removed from chunk text
            Returns:
                s1 {str} -- a string with clean chunk text
         
         """
        if len(s) < 1:
            return s
        s1 = s
        if (for_type == 'chunk') and (re.search('|'.join(punctuation), s1.text) is not None):
            for i in range(len(s1) - 1, -1, -1):
                if s1[i].text in punctuation:
                    s1 = s1[i + 1:]
#                     print(s1)
                    break
        if len(s1) > 0:
            s1 = s1[1:] if (s1[0].tag_ in trim_tags) else s1
        if len(s1) > 1:
            s1 = s1[:-1] if (s1[-1].tag_ in trim_tags) else s1
        return s1
    def trim_stop_words(self, s):
        """
            This method is used to checking and removing the stopwords from the chunks
            Arguements:
                s {object} -- noun chunk object
            Returns:
                s {str}  -- noun chunk text without stopwords
        """
        if len(s) < 1:
            return s
        if len(s) == 1:
            if str(s) == 'IT':
                return s
            s1 = [] if (s[0].text.lower() in STOP_WORDS) else s
        else:
            s1 = s[1:] if (s[0].text.lower() in STOP_WORDS) else s
            if len(s1) == 1:
                s1 = [] if (s[-1].text.lower() in STOP_WORDS) else s
            else:
                s1 = s1[:-1] if (s1[-1].text.lower() in STOP_WORDS) else s1
        return s1
    def trim_entities(self, s):
        """
            This method is used to checking and removing the chunks with POS tags other than NOUN PROPERNOUN 
            ADJECTIVE from the noun chunks text.
            Arguements:
                s {object} -- noun chunk object
            Returns:
                s {str}  -- noun chunk text with specific POS tags
        """
        if len(s) < 2:
            return s
        if s[-1].text == s.root.text:
            n = len(s) - 1
        else:
            n = len(s) - 2
        for i in range(n - 1, -1, -1):
            if not s[i].pos_ in ['NOUN', 'PROPN', 'ADJ', 'PUNCT']:
                s1 = s[i + 1:n + 1]
                break
        else:
            s1 = s[:n + 1]
        return s1
    def merge_keywords_by_similarity(self, df, th=0.4):
        """
            This method is used to calculate cosine similarity between two keywords
            Arguments:
                df {DataFrame} -- consisting of clean filtered keywords
            Default Arguments:
                th {int} -- Threshold value beyond which two keywords are considered similar
            Returns:
                df {DataFrame} -- a dataframe consisting of keywords sorted on basis of total score
        """
        if len(df) > 1:
            m = calculate_cosine_similarity(df['entity'].tolist())
            most_sim = [[j for j in range(len(m)) if m[i, j] >= th] for i in range(len(m)) if i < 20]
            m[np.tril_indices(len(m), -1)] = 0
            m[m < th] = 0
            m[m >= th] = 1
            df['total_score'] = np.matmul(m, df['weight'].values.reshape(-1, 1))
            for idx, x in enumerate(most_sim):
                if len(x) <= 1:
                    continue
                a = df.loc[x, 'entity']
                df.loc[idx, 'entity'] = a.values[0]
            df.loc[df['total_score'] == 0.0, 'total_score'] = df.loc[df['total_score'] == 0.0, 'weight'] * 1.0
            df = df.drop_duplicates(subset=['entity']).reset_index(drop=True)
            df.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
        else:
            df['total_score'] = df['weight'] if 'weight' in df.columns else 1
        return df


    def rake_score(self, df_ent_grp,df_chunk_grp):
        """
            This method is used to RAKE score cosine for the keywords in a corpus.
            Arguments:
                df_ent_grp {DataFrame} -- Dataframe consisting of entity keywords merged and sorted
                df_chunk_grp {DataFrame} -- Dataframe consisting of noun chunk keywords merged and sorted
            Returns:
                df {DataFrame} -- a dataframe consisting of bigram and trigram keywords sorted on basis of rake score and merged by similarity
        """
        combined_df = pd.concat([df_ent_grp,df_chunk_grp], ignore_index = True)    
        x_list = combined_df['entity'].to_list()
        phrases = [i.split() for i in x_list]
        frequency = defaultdict(int)
        degree = defaultdict(int)
        word_score = defaultdict(float)

        vocabulary = []

        for phrase in phrases:
            for word in phrase:
                frequency[word]+=1
                degree[word]+=len(phrase)
                if word not in vocabulary:
                    vocabulary.append(word)

        for word in vocabulary:
            word_score[word] = degree[word]/frequency[word]

        rake_dic = {}
        phrase_scores = []
        keywords = []
        phrase_vocabulary=[]

        for phrase in phrases:
            if phrase not in phrase_vocabulary:
                phrase_score=0
                for word in phrase:
                    phrase_score+=degree[word]
                phrase_scores.append(phrase_score)
                phrase_vocabulary.append(phrase)

        phrase_vocabulary = []
        j=0
        for phrase in phrases:

            if phrase not in phrase_vocabulary:
                keyword=''
                for word in phrase:
                    keyword += str(word)+" "
                phrase_vocabulary.append(phrase)
                keyword = keyword.strip()
                keywords.append(keyword)
                rake_dic[keywords[j]] = phrase_scores[j]


                j+=1
        rake_dic = dict(sorted(rake_dic.items(), key=lambda item: item[1], reverse = True))
        bitrigram  = {}
        for key,value in rake_dic.items():
            if len(key.split()) == 2:
                bitrigram[key] = value
            if len(key.split()) == 3:
                bitrigram[key] = value
        bi_trigram_df = pd.DataFrame(bitrigram.items(), columns = ['entity','r_score'])
        merged = pd.merge(combined_df,bi_trigram_df)
        if not merged.empty:
            merged_grp = self.merge_keywords_by_similarity(merged)
        return merged_grp
        
