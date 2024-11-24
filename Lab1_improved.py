from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import spacy
from datasets import load_dataset
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import pandas as pd
from collections import Counter
import optuna
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import FastText
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import logging
import json
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification.log'),
        logging.StreamHandler()
    ]
)

class TextPreprocessor:
    """Класс для предварительной обработки текста"""
    
    def __init__(self, spacy_model: str = 'ru_core_news_sm'):
        self.nlp = spacy.load(spacy_model)
        self.initialize_nltk()
        
    @staticmethod
    def initialize_nltk():
        """Инициализация необходимых компонентов NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def clean_text(self, text: str) -> str:
        """Базовая очистка текста"""
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        # Удаление специальных символов
        text = re.sub(r'[^\w\s]', '', text)
        # Удаление цифр
        text = re.sub(r'\d+', '<NUM>', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def normalize_text(self, text: str) -> str:
        """Полная нормализация текста"""
        # Базовая очистка
        text = self.clean_text(text)
        # SpaCy обработка
        doc = self.nlp(text)
        # Лемматизация и фильтрация
        tokens = []
        for token in doc:
            if not token.is_punct and not token.is_space and not token.is_stop:
                tokens.append(token.lemma_.lower())
        return ' '.join(tokens) if tokens else ''

class DataAugmentation:
    """Класс для аугментации текстовых данных"""
    
    @staticmethod
    def synonym_replacement(text: str, nlp, n: int = 1) -> str:
        """Замена слов на синонимы"""
        doc = nlp(text)
        words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        if not words:
            return text
            
        new_words = words.copy()
        for _ in range(n):
            if not words:
                break
            word_idx = np.random.randint(0, len(words))
            word = words[word_idx]
            # Здесь можно добавить логику замены на синонимы
            # Например, использовать WordNet или предварительно подготовленный словарь синонимов
        
        return ' '.join(new_words)
    
    @staticmethod
    def back_translation(text: str, src_lang: str = 'ru', tgt_lang: str = 'en') -> str:
        """Аугментация через обратный перевод"""
        # Здесь можно добавить логику перевода
        # Например, использовать API переводчика или предобученные модели
        return text

class TextVectorizer:
    """Класс для векторизации текста различными методами"""
    
    def __init__(self, method: str = 'tfidf'):
        self.method = method
        self.vectorizer = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.fasttext_model = None
        
    def initialize_bert(self):
        """Инициализация BERT модели"""
        model_name = 'DeepPavlov/rubert-base-cased'
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
    def initialize_fasttext(self, texts: List[str]):
        """Инициализация FastText модели"""
        # Обучение FastText на корпусе текстов
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        self.fasttext_model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1)
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Обучение и трансформация текстов в векторы"""
        if self.method == 'tfidf':
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )
            return self.vectorizer.fit_transform(texts)
        elif self.method == 'bert':
            if self.bert_model is None:
                self.initialize_bert()
            return self._get_bert_embeddings(texts)
        elif self.method == 'fasttext':
            if self.fasttext_model is None:
                self.initialize_fasttext(texts)
            return self._get_fasttext_embeddings(texts)
        else:
            raise ValueError(f"Unknown vectorization method: {self.method}")
            
    def transform(self, texts: List[str]) -> np.ndarray:
        """Трансформация текстов в векторы"""
        if self.method == 'tfidf':
            return self.vectorizer.transform(texts)
        elif self.method == 'bert':
            return self._get_bert_embeddings(texts)
        elif self.method == 'fasttext':
            return self._get_fasttext_embeddings(texts)
            
    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Получение эмбеддингов BERT"""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                outputs = self.bert_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy().flatten())
        return np.vstack(embeddings)
        
    def _get_fasttext_embeddings(self, texts: List[str]) -> np.ndarray:
        """Получение эмбеддингов FastText"""
        embeddings = []
        for text in texts:
            words = word_tokenize(text.lower())
            word_vectors = [self.fasttext_model.wv[word] for word in words if word in self.fasttext_model.wv]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.fasttext_model.vector_size))
        return np.vstack(embeddings)

class TextClassifier:
    """Основной класс для классификации текстов"""
    
    def __init__(self, vectorization_method: str = 'tfidf'):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TextVectorizer(method=vectorization_method)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.best_params = None
        
    def prepare_data(self, texts: List[str], labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Подготовка данных для обучения/предсказания"""
        # Предобработка текстов
        processed_texts = [self.preprocessor.normalize_text(text) for text in tqdm(texts, desc="Processing texts")]
        # Векторизация
        X = self.vectorizer.fit_transform(processed_texts) if labels is not None else self.vectorizer.transform(processed_texts)
        # Кодирование меток
        y = self.label_encoder.fit_transform(labels) if labels is not None else None
        return X, y
        
    def create_ensemble(self) -> VotingClassifier:
        """Создание ансамбля классификаторов"""
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
        ]
        return VotingClassifier(estimators=estimators, voting='soft')
        
    def optimize_hyperparameters(self, X, y, n_trials: int = 100):
        """Оптимизация гиперпараметров с помощью Optuna"""
        def objective(trial):
            params = {
                'lr__C': trial.suggest_loguniform('lr__C', 1e-3, 1e3),
                'rf__n_estimators': trial.suggest_int('rf__n_estimators', 50, 300),
                'rf__max_depth': trial.suggest_int('rf__max_depth', 3, 20)
            }
            self.model.set_params(**params)
            return np.mean(cross_val_score(self.model, X, y, cv=5, scoring='f1_macro'))
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return study.best_params
        
    def handle_imbalanced_data(self, X, y):
        """Обработка несбалансированных данных"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
        
    def fit(self, texts: List[str], labels: List[str], optimize: bool = True):
        """Обучение модели"""
        # Подготовка данных
        X, y = self.prepare_data(texts, labels)
        
        # Обработка несбалансированных данных
        X, y = self.handle_imbalanced_data(X, y)
        
        # Создание ансамбля
        self.model = self.create_ensemble()
        
        # Оптимизация гиперпараметров
        if optimize:
            best_params = self.optimize_hyperparameters(X, y)
            self.model.set_params(**best_params)
        
        # Обучение модели
        self.model.fit(X, y)
        
    def predict(self, texts: List[str]) -> List[str]:
        """Предсказание классов"""
        X, _ = self.prepare_data(texts)
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
        
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """Оценка модели"""
        X, y_true = self.prepare_data(texts, labels)
        y_pred = self.model.predict(X)
        
        # Базовые метрики
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        
        # Визуализация результатов
        self._plot_confusion_matrix(cm, self.label_encoder.classes_)
        self._plot_precision_recall_curves(X, y_true)
        
        return report
        
    def save(self, path: str):
        """Сохранение модели и метаданных"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'best_params': self.best_params
        }
        joblib.dump(model_data, path)
        
    @classmethod
    def load(cls, path: str):
        """Загрузка модели"""
        model_data = joblib.load(path)
        classifier = cls()
        classifier.model = model_data['model']
        classifier.vectorizer = model_data['vectorizer']
        classifier.label_encoder = model_data['label_encoder']
        classifier.best_params = model_data['best_params']
        return classifier
        
    def _plot_confusion_matrix(self, cm: np.ndarray, classes: List[str]):
        """Построение матрицы ошибок"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    def _plot_precision_recall_curves(self, X: np.ndarray, y_true: np.ndarray):
        """Построение кривых precision-recall"""
        plt.figure(figsize=(12, 8))
        for i, class_name in enumerate(self.label_encoder.classes_):
            y_true_binary = (y_true == i).astype(int)
            y_score = self.model.predict_proba(X)[:, i]
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            plt.plot(recall, precision, label=class_name)
            
        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png')
        plt.close()

def main():
    """Основная функция"""
    # Настройка логирования
    logging.info("Starting text classification pipeline")
    
    try:
        # Загрузка данных
        logging.info("Loading dataset")
        train_data, val_data, test_data, classes_list = load_sib200_ru()
        
        # Создание и обучение классификатора
        logging.info("Initializing classifier")
        classifier = TextClassifier(vectorization_method='tfidf')
        
        # Обучение
        logging.info("Training model")
        classifier.fit(train_data[0], train_data[1], optimize=True)
        
        # Оценка на валидационном наборе
        logging.info("Evaluating on validation set")
        val_results = classifier.evaluate(val_data[0], val_data[1])
        
        # Оценка на тестовом наборе
        logging.info("Evaluating on test set")
        test_results = classifier.evaluate(test_data[0], test_data[1])
        
        # Сохранение результатов
        results = {
            'validation': val_results,
            'test': test_results,
            'best_params': classifier.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # Сохранение модели
        classifier.save('text_classifier.joblib')
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()