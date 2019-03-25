import nltk as nltk
import numpy as np

from feature_extraction.services.image_service import get_text_from_image


class WordService:
    image_text = ""
    matrices = []

    def calculate_all_linguistic_features(self, entry):
        ## Initialization
        # TODO: check what is retrieved from non text image
        self.image_text = get_text_from_image(entry)
        self.create_matrices(entry)

        ##Features
        ratios = self.calculate_ratios()
        diffs = self.calculate_diff_features()
        all_ling = self.calculate_linguistic_features()

        return ratios, diffs, all_ling

    @staticmethod
    def calculate_basic_linguistic_features(text):
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        chars = "".join(words)
        return len(words), len(chars)

    def calculate_linguistic_features(self):
        return self.matrix_list

    def calculate_diff_features(self):
        #Triangular difference for all parts
        diffs = []
        for i in range(len(self.matrix_list)):

            for j in range(i, len(self.matrix_list)):
                diffs.append(self.matrix_list[i] - self.matrix_list[j])
        return diffs

    def calculate_ratios(self):
        #Triangular ratios between all parts

        diffs = []
        for i in range(len(self.matrix_list)):

            for j in range(i, len(self.matrix_list)):
                diffs.append(self.matrix_list[i] / self.matrix_list[j])
        return diffs

    def create_matrices(self, entry):
        #create list of matrixes containing number of chars and num of words per each part

        # Post title
        post_title_feats = np.array(self.calculate_basic_linguistic_features(entry["postText"]))

        # Post image
        post_image_feats = np.array([-1, -1])
        if self.image_text is not None:
            post_image_feats = np.array(self.calculate_basic_linguistic_features(self.image_text))

        # articles title
        article_title_feats = np.array(self.calculate_basic_linguistic_features(entry["targetTitle"]))

        # article description
        article_desc_feats = np.array(self.calculate_basic_linguistic_features(entry["targetDescription"]))

        # articles keyword
        article_keyword_feats = np.array(self.calculate_basic_linguistic_features(entry["targetKeywords"]))

        # articles captions
        article_captions_feats = np.array(self.calculate_basic_linguistic_features(" ".join(entry["targetCaptions"])))

        # articles paragraphs
        article_paragraph_feats = np.array(self.calculate_basic_linguistic_features(" ".join(entry["targetParagraphs"])))

        # triangular diff
        self.matrix_list = [post_title_feats, article_title_feats, article_desc_feats, article_keyword_feats, article_captions_feats,
                            article_paragraph_feats, post_image_feats]
