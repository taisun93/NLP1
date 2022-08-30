# models.py

from sentiment_data import *
from utils import *
import string
import random
import numpy as np

from collections import Counter


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self) -> Indexer:
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.blackList = ['in', 'is', 'the', 'and']

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """

        answer = []
        # start = True
        for word in sentence:
            # if word in string.punctuation:
            #     continue
            # if not start and not word.islower():
            #     continue
            # start = False
            # word = word.lower()
            # if word in self.blackList:
            #     continue
            self.indexer.add_and_get_index(word)
            answer.append(word)
        


        return Counter(set(answer))


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, featureExtractor: FeatureExtractor):
        self.weights = [0]*100000
        self.extractor = featureExtractor
        self.indexer = featureExtractor.get_indexer()

    def train(self, train_exs: List[SentimentExample]):
        for example in train_exs:
            features = self.extractor.extract_features(example.words, True)
            if self.predict(features) != example.label:
                self.adjust(features, example.label)

    def adjust(self, features: Counter, label):
        if label == 0:
            label = -1
        for feat in features:
            self.weights[self.indexer.index_of(feat)] += label

    def predict(self, features: Counter) -> int:
        answer = 0
        for feat in features:
            answer += self.weights[self.indexer.index_of(feat)]

        if answer > 0:
            return 1
        return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, featureExtractor: FeatureExtractor):
        self.weights = [0.5]*100000
        self.extractor = featureExtractor
        self.indexer = featureExtractor.get_indexer()
        

    def train(self, train_exs: List[SentimentExample]):
        for example in train_exs:
            features = self.extractor.extract_features(example.words, True)
            prediction = self.get_prob(features)
            self.adjust(features, prediction, example.label)

    def adjust(self, features: Counter, prediction, label):
        diff = label-prediction
        # print([label,prediction,diff])
        for feat in features:
            self.weights[self.indexer.index_of(feat)] += diff
            

    def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

    def get_prob(self, features: Counter) -> int:
        answer = 0
        for feat in features:
            answer += self.weights[self.indexer.index_of(feat)]

        return self.sigmoid(answer)

    def predict(self, features: Counter) -> int:
        return round(self.get_prob(features))

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    
    # random.seed(10)
    
    classifier = PerceptronClassifier(feat_extractor)
    for _ in range(20):
        random.shuffle(train_exs)
        classifier.train(train_exs)

    return classifier


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    
    random.seed(10)

    classifier = LogisticRegressionClassifier(feat_extractor)
    for _ in range(15):
        random.shuffle(train_exs)
        classifier.train(train_exs)

    # print(classifier.extractor.get_indexer().__len__())

    return classifier


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception(
            "Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception(
            "Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
