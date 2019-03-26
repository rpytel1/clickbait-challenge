import nltk as nltk
import numpy as np

from feature_extraction.services.image_service import get_text_from_image


class WordService:
    image_text = ""
    matrices = []

    def calculate_all_linguistic_features(self, entry):
        ## Initialization
        self.image_text = get_text_from_image(entry)
        self.create_matrices(entry)

        ##Features
        ratios = self.calculate_ratios()
        diffs = self.calculate_diff_features()
        all_ling = self.calculate_linguistic_features()

        return self.flatten_list_of_lists([ratios, diffs, all_ling])

    def calculate_linguistic_features(self):
        return self.unwrap_from_np_array(self.matrix_list)

    def calculate_diff_features(self):
        # Triangular difference for all parts
        diffs = []
        for i in range(len(self.matrix_list)):

            for j in range(i+1, len(self.matrix_list)):
                diffs.append(list(self.matrix_list[i] - self.matrix_list[j]))
        return self.flatten_list_of_lists(diffs)

    def calculate_ratios(self):
        # Triangular ratios between all parts

        diffs = []
        for i in range(len(self.matrix_list)):

            for j in range(i+1, len(self.matrix_list)):
                ratio = np.divide(self.matrix_list[i], self.matrix_list[j], out=np.ones_like(self.matrix_list[j])*(-1),
                          where=self.matrix_list[j] != 0 )
                ratio[self.matrix_list[i] == 0] = 0

                diffs.append(ratio)

        return self.flatten_list_of_lists(diffs)

    def create_matrices(self, entry):
        # create list of matrixes containing number of chars and num of words per each part

        # Post title
        post_title_feats = np.array(self.calculate_basic_linguistic_features(entry["postText"][0]), dtype=np.float)

        # Post image
        post_image_feats = np.array([0., 0.], dtype=np.float)
        if self.image_text:
            post_image_feats = np.array(self.calculate_basic_linguistic_features(self.image_text), dtype=np.float)

        # articles title
        article_title_feats = np.array(self.calculate_basic_linguistic_features(entry["targetTitle"]), dtype=np.float)

        # article description
        article_desc_feats = np.array(self.calculate_basic_linguistic_features(entry["targetDescription"]), dtype=np.float)

        # articles keyword
        article_keyword_feats = np.array(self.calculate_basic_linguistic_features(entry["targetKeywords"]), dtype=np.float)

        # articles captions
        article_captions_feats = np.array(self.calculate_basic_linguistic_features(" ".join(entry["targetCaptions"])), dtype=np.float)

        # articles paragraphs
        article_paragraph_feats = np.array(
            self.calculate_basic_linguistic_features(" ".join(entry["targetParagraphs"])), dtype=np.float)

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
