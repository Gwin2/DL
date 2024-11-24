from typing import Dict, List, Union, Optional, Tuple
from datasets import load_dataset
from fuzzywuzzy import fuzz, process
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
from tqdm.notebook import tqdm
from transformers import pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_classification.log'),
        logging.StreamHandler()
    ]
)

class LLMClassifier:
    """Класс для классификации текстов с использованием LLM"""
    
    def __init__(self, model_name: str = 'Qwen/Qwen2-7B-Instruct', 
                 device: str = 'auto', 
                 batch_size: int = 8,
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.categories: List[str] = []
        self.pipeline = None
        self.initialize_pipeline()
        
    def initialize_pipeline(self):
        """Инициализация модели с учетом доступных ресурсов"""
        try:
            # Определение оптимального dtype в зависимости от устройства
            if torch.cuda.is_available() and self.device != 'cpu':
                dtype = torch.float16
            else:
                dtype = torch.float32
                
            self.pipeline = pipeline(
                model=self.model_name,
                device_map=self.device,
                torch_dtype=dtype,
                model_kwargs={'cache_dir': self.cache_dir}
            )
            logging.info(f"Successfully initialized model {self.model_name}")
        except Exception as e:
            logging.error(f"Error initializing pipeline: {str(e)}")
            raise
            
    @staticmethod
    @lru_cache(maxsize=1024)
    def prepare_single_message(text: str, categories_str: str) -> List[Dict[str, str]]:
        """Подготовка одного сообщения для LLM с кэшированием"""
        prompt = (
            f'Прочтите, пожалуйста, следующий текст и определите, какая тема из известного '
            f'списка тем наиболее представлена в следующем тексте. '
            f'В качестве ответа напишите только название темы из списка, больше ничего.\n'
            f'Список тем: {categories_str}.\nТекст: {" ".join(text.split())}\nВаш ответ: '
        )
        return [
            {
                'role': 'system',
                'content': 'Вы - полезный помощник, умеющий читать тексты на русском языке, глубоко понимать их и анализировать.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]

    def prepare_message_for_llm(self, 
                              text: Union[str, List[str]], 
                              categories: List[str]) -> Dict[str, Union[List[Dict[str, str]], List[List[Dict[str, str]]]]]:
        """Подготовка сообщений для LLM с проверками и оптимизацией"""
        if len(categories) < 2:
            raise ValueError(f'The category list is too small! Expected 2 or more categories, got {len(categories)} ones.')
            
        categories_as_string = ', '.join(categories[:-1]) + ' и ' + categories[-1]
        
        if isinstance(text, str):
            messages = self.prepare_single_message(text, categories_as_string)
        else:
            messages = [
                self.prepare_single_message(t, categories_as_string)
                for t in text
            ]
            
        return {'message_for_llm': messages}

    def process_batch(self, batch: List[Dict]) -> List[str]:
        """Обработка пакета текстов"""
        try:
            results = self.pipeline(
                batch,
                max_new_tokens=10,
                batch_size=self.batch_size
            )
            return [r['generated_text'] for r in results]
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            return ['ERROR'] * len(batch)

    def predict(self, texts: List[str], categories: List[str]) -> List[str]:
        """Предсказание категорий для списка текстов"""
        self.categories = categories
        prepared_data = self.prepare_message_for_llm(texts, categories)
        messages = prepared_data['message_for_llm']
        
        predictions = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i + self.batch_size]
            batch_predictions = self.process_batch(batch)
            predictions.extend(batch_predictions)
            
        # Нормализация предсказаний
        normalized_predictions = [
            process.extractOne(
                pred[-1]['content'], 
                categories, 
                scorer=fuzz.token_sort_ratio
            )[0]
            for pred in predictions
        ]
        
        return normalized_predictions

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """Расширенная оценка качества классификации"""
        # Базовый отчет
        report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True
        )
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred, labels=self.categories)
        
        # Дополнительные метрики
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='weighted'
        )
        
        # Визуализация
        self._plot_confusion_matrix(cm)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'overall_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
        
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Визуализация матрицы ошибок"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.categories,
            yticklabels=self.categories
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('llm_confusion_matrix.png')
        plt.close()

def load_and_prepare_data(dataset_name: str, 
                         language: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Загрузка и подготовка данных"""
    try:
        logging.info(f"Loading dataset {dataset_name}")
        train_set = load_dataset(dataset_name, language, split='train')
        validation_set = load_dataset(dataset_name, language, split='validation')
        test_set = load_dataset(dataset_name, language, split='test')
        
        # Получение уникальных категорий
        categories = sorted(list(
            set(train_set['category']) |
            set(validation_set['category']) |
            set(test_set['category'])
        ))
        
        logging.info(f"Found {len(categories)} categories")
        return categories, train_set['text'], validation_set['text'], test_set['text']
        
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def main():
    """Основная функция"""
    try:
        # Параметры
        DATASET_NAME = 'Davlan/sib200'
        DATASET_LANGUAGE = 'rus_Cyrl'
        
        # Загрузка данных
        categories, train_texts, val_texts, test_texts = load_and_prepare_data(
            DATASET_NAME,
            DATASET_LANGUAGE
        )
        
        # Инициализация классификатора
        classifier = LLMClassifier(batch_size=8)
        
        # Классификация валидационного набора
        logging.info("Processing validation set")
        val_predictions = classifier.predict(val_texts, categories)
        
        # Оценка результатов
        validation_results = classifier.evaluate(validation_set['category'], val_predictions)
        
        # Сохранение результатов
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': classifier.model_name,
            'dataset': DATASET_NAME,
            'validation_results': validation_results
        }
        
        with open('llm_classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logging.info("Classification completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()