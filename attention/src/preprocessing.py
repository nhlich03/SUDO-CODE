import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Preprocessor:
    def __init__(self, max_len_content=300, max_len_summary=30, num_words_content=None, num_words_summary=None):
        self.max_len_content = max_len_content
        self.max_len_summary = max_len_summary
        self.num_words_content = num_words_content
        self.num_words_summary = num_words_summary
        self.content_tokenizer = None
        self.summary_tokenizer = None

    def clean_text(self, text, keep_punct=True):
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit_tokenizers(self, contents, summaries):
        self.content_tokenizer = Tokenizer(num_words=self.num_words_content, oov_token="<unk>")
        self.content_tokenizer.fit_on_texts(contents)

        self.summary_tokenizer = Tokenizer(num_words=self.num_words_summary, oov_token="<unk>")
        self.summary_tokenizer.fit_on_texts(summaries)

    def transform(self, contents, summaries):
        contents = [self.clean_text(t) for t in contents]
        summaries = [self.clean_text(t) for t in summaries]
        summaries = ["<sos> " + s + " <eos>" for s in summaries]

        content_seq = self.content_tokenizer.texts_to_sequences(contents)
        summary_seq = self.summary_tokenizer.texts_to_sequences(summaries)

        content_pad = pad_sequences(content_seq, maxlen=self.max_len_content, padding='post')
        summary_pad = pad_sequences(summary_seq, maxlen=self.max_len_summary, padding='post')
        return content_pad, summary_pad

    def get_vocab_size(self):
        content_vocab_size = len(self.content_tokenizer.word_index) + 1
        summary_vocab_size = len(self.summary_tokenizer.word_index) + 1
        return content_vocab_size, summary_vocab_size
