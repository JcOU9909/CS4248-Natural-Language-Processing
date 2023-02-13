'''
    NUS CS4248 Assignment 1 - Objective 3 (n-gram Language Model)

    Class NgramLM for handling Objective 3

    Important: please strictly comply with the input/output formats for
               the methods of generate_word & generate_text & get_perplexity, 
               as we will call them during testing

    Sentences for Task 3:
    1) "They just entered a beautiful walk by"
    2) "The rabbit hopped onto a beautiful walk by the garden."
    3) "They had just spotted a snake entering"
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import re
import random, math
import collections
from nltk.tokenize import word_tokenize, sent_tokenize


class NgramLM(object):

    def __init__(self, path: str, n: int, k: float):
        '''This method is mandatory to implement with the method signature as-is.

            Initialize your n-gram LM class

            Parameters:
                n (int) : order of the n-gram model
                k (float) : smoothing hyperparameter

            Suggested function dependencies:
                read_file -> init_corpus |-> get_ngrams_from_seqs -> add_padding_to_seq
                                         |-> get_vocab_from_tokens

                generate_text -> generate_word -> get_next_word_probability

                get_perplexity |-> get_ngrams_from_seqs
                               |-> get_next_word_probability

        '''
        # Initialise other variables as necessary
        # TODO Write your code here
        self.n = n
        self.k = k
        self.text = None

        # Fields below are optional but recommended; you may replace as you like
        self.ngram_dict = collections.defaultdict(int)
        self.vocabs = collections.defaultdict(int)
        self.special_tokens = {'bos': '~', 'eos': '<EOS>'}
        self.read_file(path)

    def read_file(self, path: str):
        ''' Reads text from file path and initiate n-gram corpus.

        PS: Change the function signature as you like.
            This method is a suggested method to implement,
            which you may call in the method of __init__
        '''
        # TODO Write your code here
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        self.init_corpus(text)

    def init_corpus(self, text: str):
        ''' Initiates n-gram corpus based on loaded text

        PS: Change the function signature as you like.
            This method is only a suggested method,
            which you may call in the method of read_file
        '''
        # TODO Write your code here
        text = text.replace(")", ' ')
        text = text.replace("(", ' ')
        text = text.replace("“", ' ')
        text = text.replace("”", ' ')
        self.text = text

        # sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=[.?,!;])\s", text)
        # sentences = [sentence.replace('\n', ' ') for sentence in sentences]

        sentences = sent_tokenize(self.text)
        sentences = [sentence.lower() for sentence in sentences]

        self.ngram_dict = self.get_ngrams_from_seqs(sentences)
        self.get_vocab_from_tokens(sentences)

        return

    def get_vocab_from_tokens(self, sentences):
        ''' Returns the vocabulary (e.g. {word: count}) from a list of tokens

        Hint: to get the vocabulary, you need to first tokenize the corpus.

        PS: Change the function signature as you like.
            This method is a suggested method to implement,
            which you may call in the method of init_corpus.
        '''
        # TODO Write your code here
        sentences = [self.add_padding_to_seq(sentence) for sentence in sentences]

        for sentence in sentences:
            if not sentence:
                continue
            words = word_tokenize(sentence)
            for word in words:
                self.vocabs[word] += 1

        return

    def get_ngrams_from_seqs(self, sentences):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)]
            where sequence context is the ngram and word is its last word

        Hint: to get the ngrams, you may need to first get split sentences from corpus,
            and add paddings to them.

        PS: Change the function signature as you like.
            This method is a suggested method to implement,
            which you may call in the method of init_corpus
        '''
        # TODO Write your code here
        sentences = [self.add_padding_to_seq(sentence) for sentence in sentences]
        ngram_dict = collections.defaultdict(int)
        for sentence in sentences:
            if not sentence:
                continue
            words = word_tokenize(sentence)
            for i in range(len(words) - self.n + 1):
                ngram_dict[tuple(words[i:i + self.n])] += 1

        return ngram_dict

    def add_padding_to_seq(self, sentence: str):
        '''  Adds paddings to a sentence.
        The goal of the method is to pad start token(s) to input sentence,
        so that we can get token '~ I' from a sentence 'I like NUS.' as in the bigram case.

        PS: Change the function signature as you like.
            This method is a suggested method to implement,
            which you may call in the method of get_ngrams_from_seqs
        '''
        # TODO Write your code here
        # Use '~' as your padding symbol
        if len(sentence) >= 1:
            return '~ ' + sentence
        else:
            return ''

    def get_next_word_probability(self, text: str, word: str, dom: float):
        ''' Returns probability of a word occurring after specified text,
        based on learned ngrams.

        PS: Change the function signature as you like.
            This method is a suggested method to implement,
            which you may call in the method of generate_word
        '''
        # TODO Write your code here
        count_vocab = len(self.vocabs.keys())
        if self.n == 1:
            count_word = sum(self.vocabs.values())
            return (self.vocabs[word] + self.k) / (count_word + self.k * count_vocab)
        else:
            if len(text) == 0 or bool(re.search(r".*[,.!;?~]$", text)):
                return (self.ngram_dict[('~', word)] + self.k) / (self.vocabs['~'] + self.k * count_vocab)

            words = word_tokenize(text)

            subtext = words[len(words) - self.n + 1:]
            prob = self.ngram_dict[tuple(subtext + [word])] + self.k

            return prob / dom

        return

    def generate_word(self, text: str):
        '''
        Generates a random word based on the specified text and the ngrams learned
        by the model

        PS: This method is mandatory to implement with the method signature as-is.
            We only test one sentence at a time, so you may not need to split
            the text into sentences here.

        [In] string (a full sentence or half of a sentence)
        [Out] string (a word)
        '''
        # TODO Write your code here
        weight = []
        dom = 0
        if self.n >= 2 and not bool(re.search(r".*[,.!;?~]$", text)):
            words = word_tokenize(text)
            # print(words)
            subtext = words[len(words) - self.n + 1:]
            for j in self.vocabs.keys():
                dom += self.ngram_dict[tuple(subtext + [j])] + self.k

        for i in self.vocabs.keys():
            weight.append(self.get_next_word_probability(text, i, dom))

        # print(weight)

        return random.choices(list(self.vocabs.keys()), weights=weight)[0]

    def generate_text(self, length: int):
        ''' Generate text of a specified length based on the learned ngram model

        [In] int (length: number of tokens)
        [Out] string (text)

        PS: This method is mandatory to implement with the method signature as-is.
            The length here is a reasonable int number, (e.g., 3~20)
        '''
        # TODO Write your code here
        list_ans = ['~']

        while len(list_ans) < length + 1:
            next_word = self.generate_word(' '.join(list_ans))
            list_ans.append(next_word)
        return ' '.join(list_ans)

    def get_perplexity(self, text: str):
        '''
        Returns the perplexity of texts based on learned ngram model.
        Note that text may be a concatenation of multiple sequences.

        [In] string (a short text)
        [Out] float (perplexity)

        PS: This method is mandatory to implement with the method signature as-is.
            The output is the perplexity, not the log form you use to avoid
            numerical underflow in calculation.

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
              Also handle the case when the LM assigns zero probabilities.
        '''
        # TODO Write your code here
        ppl = 0
        text = text.lower()
        sentences = sent_tokenize(text)
        length_text = len(word_tokenize(text))

        tmp_dict = self.get_ngrams_from_seqs(sentences)

        if self.n == 2:
            for token in tmp_dict.keys():
                front = token[0]
                dom = 0
                for back in self.vocabs.keys():
                    dom += self.ngram_dict[tuple([front] + [back])] + self.k
                prob = self.get_next_word_probability(token[0], token[1], dom)
                # print(self.vocabs[front], len(self.vocabs.keys()), dom, token, prob, self.ngram_dict[token])
                ppl -= math.log(prob)
        else:
            for token in tmp_dict.keys():
                ppl -= math.log(self.get_next_word_probability('', token[0], 0))

        return ppl / length_text


if __name__ == '__main__':
    print('''[Alert] Time your code and make sure it finishes within 2 minutes!''')

    LM = NgramLM('../data/Pride_and_Prejudice.txt', n=2, k=1.0)

    test_cases = ["The rabbit hopped onto a beautiful walk by the garden.",
                  "They just entered a beautiful walk by",
                  "They had just spotted a snake entering"]

    for case in test_cases:
        word = LM.generate_word(case)
        ppl = LM.get_perplexity(case)
        print(f'input text: {case}\nnext word: {word}\nppl: {ppl}')

    _len = 10
    text = LM.generate_text(length=_len)
    print(f'\npredicted text of length {_len}: {text}')
