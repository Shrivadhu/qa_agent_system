"""NLP utility functions for QA Agent system."""

import re
import string
from typing import List, Dict, Tuple
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import yake


class NLPUtils:
    """NLP utility functions."""
    
    def __init__(self):
        """Initialize NLP utilities."""
        self._ensure_nltk_data()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Offline and no corpus available: use sklearn's built-in list
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            self.stop_words = set(ENGLISH_STOP_WORDS)
        self._use_nltk_tokenizer = self._nltk_tokenizer_available()

    @staticmethod
    def _nltk_tokenizer_available() -> bool:
        try:
            sent_tokenize("Test one. Test two.")
            word_tokenize("test")
            return True
        except LookupError:
            return False

    _nltk_checked = False

    @staticmethod
    def _ensure_nltk_data():
        """Download required NLTK data once (punkt_tab is needed by NLTK >= 3.9)."""
        if NLPUtils._nltk_checked:
            return
        NLPUtils._nltk_checked = True
        import nltk
        for resource, path in [
            ('punkt', 'tokenizers/punkt'),
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('stopwords', 'corpora/stopwords'),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_text(self, text: str, lowercase: bool = True) -> str:
        """Preprocess text for analysis."""
        text = self.clean_text(text)
        if lowercase:
            text = text.lower()
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self._use_nltk_tokenizer:
            return sent_tokenize(text)
        return [s for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words."""
        if self._use_nltk_tokenizer:
            return word_tokenize(text)
        return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def extract_keywords_tfidf(
        self, 
        text: str, 
        max_features: int = 100,
        ngram_range: Tuple[int, int] = (1, 3)
    ) -> List[Dict[str, any]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Input text
            max_features: Maximum number of keywords
            ngram_range: Range of n-grams to consider
            
        Returns:
            List of dicts with 'keyword' and 'score'
        """
        # Preprocess
        clean_text = self.preprocess_text(text)
        
        # A single document with min_df=2 always yields an empty vocabulary,
        # so treat each sentence as a document and score terms over the corpus.
        documents = [s for s in self.tokenize_sentences(clean_text) if s.strip()]
        if not documents:
            return []

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=1
        )
        
        try:
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Aggregate: sum of TF-IDF weights per term across sentences
            scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            
            # Create keyword list
            keywords = [
                {"keyword": feature_names[i], "score": float(scores[i])}
                for i in range(len(feature_names))
            ]
            
            # Sort by score
            keywords.sort(key=lambda x: x['score'], reverse=True)
            
            return keywords
            
        except ValueError:
            # Not enough content for TF-IDF
            return []
    
    def extract_keywords_yake(
        self,
        text: str,
        max_keywords: int = 100,
        ngram_size: int = 3
    ) -> List[Dict[str, any]]:
        """
        Extract keywords using YAKE algorithm.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            ngram_size: Maximum n-gram size
            
        Returns:
            List of dicts with 'keyword' and 'score'
        """
        # Create YAKE extractor
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=ngram_size,
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=1,
            top=max_keywords
        )
        
        # Extract keywords
        keywords = kw_extractor.extract_keywords(text)
        
        # Convert to standard format (lower score is better in YAKE)
        return [
            {"keyword": kw.lower(), "score": 1.0 - min(score, 1.0)}
            for kw, score in keywords
        ]
    
    def extract_keywords_combined(
        self,
        text: str,
        max_keywords: int = 100
    ) -> List[Dict[str, any]]:
        """
        Extract keywords using both TF-IDF and YAKE, then combine.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            
        Returns:
            List of dicts with 'keyword' and 'score'
        """
        # Get keywords from both methods
        tfidf_keywords = self.extract_keywords_tfidf(text, max_features=max_keywords)
        yake_keywords = self.extract_keywords_yake(text, max_keywords=max_keywords)
        
        # Combine and normalize
        keyword_dict = {}
        
        # Add TF-IDF keywords
        for kw in tfidf_keywords:
            keyword_dict[kw['keyword']] = kw['score']
        
        # Add YAKE keywords (average if duplicate)
        for kw in yake_keywords:
            if kw['keyword'] in keyword_dict:
                keyword_dict[kw['keyword']] = (
                    keyword_dict[kw['keyword']] + kw['score']
                ) / 2
            else:
                keyword_dict[kw['keyword']] = kw['score']
        
        # Convert back to list
        combined = [
            {"keyword": k, "score": v}
            for k, v in keyword_dict.items()
        ]
        
        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        return combined[:max_keywords]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using TF-IDF.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        vectorizer = TfidfVectorizer()
        
        try:
            tfidf = vectorizer.fit_transform([text1, text2])
            similarity = (tfidf * tfidf.T).toarray()[0, 1]
            return float(similarity)
        except ValueError:
            return 0.0
    
    def detect_questions(self, text: str) -> List[str]:
        """
        Detect questions in text.
        
        Handles both punctuated questions ending with '?' and speech-to-text (ASR)
        dialogue sentences starting with interrogative or auxiliary words.
        
        Args:
            text: Input text
            
        Returns:
            List of detected questions
        """
        sentences = self.tokenize_sentences(text)
        questions = []
        
        wh_starters = (
            "what", "why", "how", "when", "where", "who", "which", "whose", "whom"
        )
        aux_starters = (
            "can", "could", "would", "should", "do", "does", "did",
            "is", "are", "was", "were", "have", "has", "had",
            "will", "shall", "may", "might"
        )
        subject_pronouns = (
            "you", "i", "we", "they", "he", "she", "it",
            "this", "that", "there", "the", "a", "an"
        )
        
        for s in sentences:
            stripped = s.strip()
            if not stripped:
                continue
            if stripped.endswith('?'):
                questions.append(stripped)
                continue
            
            # For ASR transcripts without punctuation:
            words = [w.lower() for w in re.findall(r"\w+", stripped)]
            if len(words) >= 3:
                first_word = words[0]
                if first_word in wh_starters:
                    questions.append(stripped)
                elif first_word in aux_starters and words[1] in subject_pronouns:
                    questions.append(stripped)
        
        return questions
    
    def detect_statements_of_understanding(self, text: str) -> List[str]:
        """
        Detect statements indicating understanding.
        
        Args:
            text: Input text
            
        Returns:
            List of understanding statements
        """
        understanding_phrases = [
            "i understand",
            "i get it",
            "makes sense",
            "makes total sense",
            "makes complete sense",
            "got it",
            "i see",
            "i see what you mean",
            "that's clear",
            "thats clear",
            "that is clear",
            "now i know",
            "learned that",
            "understand now",
            "now i understand",
            "now i get it",
            "i feel like i understand",
            "i follow you",
            "i follow",
            "clear now",
            "got the hang of it",
            "cleared that up",
            "clears it up",
            "oh i see",
            "aha"
        ]
        
        sentences = self.tokenize_sentences(text.lower())
        understanding_statements = []
        
        for sentence in sentences:
            for phrase in understanding_phrases:
                if phrase in sentence:
                    understanding_statements.append(sentence)
                    break
        
        return understanding_statements
    
    def calculate_sentence_length_variance(self, text: str) -> float:
        """
        Calculate variance in sentence lengths.
        
        Args:
            text: Input text
            
        Returns:
            Coefficient of variation (std / mean)
        """
        sentences = self.tokenize_sentences(text)
        lengths = [len(self.tokenize_words(s)) for s in sentences]
        
        if len(lengths) < 2:
            return 0.0
        
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        if mean_length == 0:
            return 0.0
        
        return std_length / mean_length
    
    def count_specific_details(self, text: str) -> int:
        """
        Count specific details (numbers, proper nouns, etc.).
        
        Args:
            text: Input text
            
        Returns:
            Count of specific details
        """
        # Count numbers
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = len(re.findall(number_pattern, text))
        
        # Count capitalized words (potential proper nouns)
        words = self.tokenize_words(text)
        capitalized = sum(1 for w in words if len(w) > 1 and w[0].isupper())
        
        return numbers + capitalized
