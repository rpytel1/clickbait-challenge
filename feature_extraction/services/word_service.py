import nltk as nltk
import numpy as np

from feature_extraction.services.image_service import get_text_from_image


class WordService:
    def __init__(self):
        self.image_text = ""
        self.matrix_list = []
        self.feat_list = []

    def calculate_all_linguistic_features(self, entry):
        ## Initialization
        self.image_text = get_text_from_image(entry)
        self.create_matrices(entry)

        ##Features
        all_ling = self.calculate_linguistic_features()
        ratios = self.calculate_ratios()
        diffs = self.calculate_diff_features()

        return self.flatten_list_of_lists([all_ling, ratios, diffs]) + self.feat_list

    def calculate_linguistic_features(self):
        return self.unwrap_from_np_array(self.matrix_list)

    def calculate_diff_features(self):
        # Triangular difference for all parts
        diffs = []
        for i in range(len(self.matrix_list)):
            for j in range(i+1, len(self.matrix_list)):
                self.feat_list += ['diff_'+self.feat_list[2*i]+'_'+self.feat_list[2*j]]
                self.feat_list += ['diff_' + self.feat_list[2*i+1] + '_' + self.feat_list[2*j+1]]
                diffs.append(list(map(abs, list(self.matrix_list[i] - self.matrix_list[j]))))
        return self.flatten_list_of_lists(diffs)

    def calculate_ratios(self):
        # Triangular ratios between all parts

        diffs = []
        for i in range(len(self.matrix_list)):
            for j in range(i+1, len(self.matrix_list)):
                self.feat_list += ['ratio_' + self.feat_list[2*i] + '_' + self.feat_list[2*j]]
                self.feat_list += ['ratio_' + self.feat_list[2*i+1] + '_' + self.feat_list[2*j+1]]
                ratio = np.divide(self.matrix_list[i], self.matrix_list[j], out=np.ones_like(self.matrix_list[j])*(-1),
                          where=np.multiply(self.matrix_list[i], self.matrix_list[j]) != 0)
                diffs.append(ratio)
        return self.flatten_list_of_lists(diffs)

    def create_matrices(self, entry):
        # create list of matrixes containing number of chars and num of words per each part

        # Post title
        post_title_feats = np.array(self.calculate_basic_linguistic_features(entry["postText"][0]), dtype=np.float)
        self.feat_list += ['post_title_len_words', 'post_title_len_chars']

        # Post image
        post_image_feats = np.array([0., 0.], dtype=np.float)
        if self.image_text:
            post_image_feats = np.array(self.calculate_basic_linguistic_features(self.image_text), dtype=np.float)
        self.feat_list += ['post_image_len_words', 'post_image_len_chars']

        # articles title
        article_title_feats = np.array(self.calculate_basic_linguistic_features(entry["targetTitle"]), dtype=np.float)
        self.feat_list += ['article_title_len_words', 'article_title_len_chars']

        # article description
        article_desc_feats = np.array(self.calculate_basic_linguistic_features(entry["targetDescription"]), dtype=np.float)
        self.feat_list += ['article_desc_len_words', 'article_desc_len_chars']

        # articles keyword
        article_keyword_feats = np.array(self.calculate_basic_linguistic_features(entry["targetKeywords"]), dtype=np.float)
        self.feat_list += ['article_keyword_len_words', 'article_keyword_len_chars']

        # articles captions
        article_captions_feats = np.array(self.calculate_basic_linguistic_features(" ".join(entry["targetCaptions"])), dtype=np.float)
        self.feat_list += ['article_caption_len_words', 'article_caption_len_chars']

        # articles paragraphs
        article_paragraph_feats = np.array(
            self.calculate_basic_linguistic_features(" ".join(entry["targetParagraphs"])), dtype=np.float)
        self.feat_list += ['article_paragraph_len_words', 'article_paragraph_len_chars']

        # triangular diff
        self.matrix_list = [post_title_feats, article_title_feats, article_desc_feats, article_keyword_feats,
                            article_captions_feats, article_paragraph_feats, post_image_feats]

    @staticmethod
    def unwrap_from_np_array(lis_of_np_array):
        list_of_lists = [x.tolist() for x in lis_of_np_array]
        return [item for sublist in list_of_lists for item in sublist]

    @staticmethod
    def flatten_list_of_lists(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    @staticmethod
    def calculate_basic_linguistic_features(text):
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        chars = "".join(words)
        if not len(words):
            len_words = 0
        else:
            len_words = len(words)

        if not len(chars):
            len_chars = 0
        else:
            len_chars = len(chars)

        return len_words, len_chars
