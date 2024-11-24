import logging
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from fuzzywuzzy import fuzz, process
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import pipeline
import torch
from functools import lru_cache
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the LLM-based classifier"""
    model_name: str = 'Qwen/Qwen2-7B-Instruct'
    dataset_name: str = 'Davlan/sib200'
    dataset_language: str = 'rus_Cyrl'
    max_new_tokens: int = 10
    batch_size: int = 8
    cache_dir: str = '.cache'
    results_dir: str = 'results'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class LLMTextClassifier:
    """LLM-based text classifier for Russian language"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.setup_directories()
        
        logger.info(f"Initializing LLM classifier with model: {config.model_name}")
        self.llm_pipeline = pipeline(
            model=config.model_name,
            device_map='auto',
            torch_dtype='auto'
        )
        
        self.load_datasets()
        self.setup_categories()
        
    def setup_directories(self):
        """Create necessary directories for caching and results"""
        Path(self.config.cache_dir).mkdir(exist_ok=True)
        Path(self.config.results_dir).mkdir(exist_ok=True)
        
    def load_datasets(self):
        """Load and prepare datasets"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        try:
            self.train_set = load_dataset(self.config.dataset_name, self.config.dataset_language, split='train')
            self.validation_set = load_dataset(self.config.dataset_name, self.config.dataset_language, split='validation')
            self.test_set = load_dataset(self.config.dataset_name, self.config.dataset_language, split='test')
        except Exception as e:
            logger.error(f"Failed to load datasets: {str(e)}")
            raise
            
    def setup_categories(self):
        """Setup category examples and list"""
        logger.info("Setting up categories")
        self.list_of_categories = sorted(list(
            set(self.train_set['category']) |
            set(self.validation_set['category']) |
            set(self.test_set['category'])
        ))
        
        self.examples_by_categories = self._get_category_examples()
        logger.info(f"Found {len(self.list_of_categories)} categories")
        
    def _get_category_examples(self) -> Dict[str, str]:
        """Get example texts for each category"""
        examples = {}
        for category in self.list_of_categories:
            examples[category] = random.choice(
                self.train_set.filter(lambda x: x['category'] == category)['text']
            )
        return examples
    
    @lru_cache(maxsize=1024)
    def prepare_message_for_llm(self, text: str, categories_tuple: Tuple[str, ...]) -> Dict:
        """Prepare message for LLM with caching"""
        categories = dict(zip(categories_tuple, [self.examples_by_categories[cat] for cat in categories_tuple]))
        
        if len(categories) < 2:
            raise ValueError(f'Expected 2+ categories, got {len(categories)}')
            
        categories_list = sorted(list(categories.keys()))
        categories_string = ', '.join(categories_list[:-1]) + ' и ' + categories_list[-1]
        
        prompt = (
            f'Прочтите, пожалуйста, следующий текст и определите, какая тема из известного '
            f'списка тем наиболее представлена в следующем тексте. '
            f'В качестве ответа напишите только название темы из списка, больше ничего.\n'
            f'Список тем: {categories_string}.\n'
        )
        
        for category in categories:
            prompt += f'Текст: {" ".join(categories[category].split())}\nВаш ответ: {category}\n'
        prompt += f'Текст: {" ".join(text.split())}\nВаш ответ: '
        
        return {
            'message_for_llm': [
                {
                    'role': 'system',
                    'content': 'Вы - полезный помощник, умеющий читать тексты на русском языке, глубоко понимать их и анализировать.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for LLM classification"""
        categories_tuple = tuple(sorted(self.list_of_categories))
        return dataset.map(
            lambda x: self.prepare_message_for_llm(x['text'], categories_tuple)
        )
    
    def predict_batch(self, messages: List) -> List[str]:
        """Make predictions for a batch of messages"""
        try:
            predictions = self.llm_pipeline(
                messages,
                max_new_tokens=self.config.max_new_tokens,
                batch_size=self.config.batch_size
            )
            return [pred['generated_text'] for pred in predictions]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return [''] * len(messages)
    
    def normalize_predictions(self, predictions: List[str]) -> List[str]:
        """Normalize predictions using fuzzy matching"""
        return [
            process.extractOne(pred, self.list_of_categories, scorer=fuzz.token_sort_ratio)[0]
            for pred in predictions
        ]
    
    def evaluate(self, dataset: Dataset, split_name: str) -> Dict:
        """Evaluate model on dataset"""
        logger.info(f"Evaluating on {split_name} split")
        
        prepared_dataset = self.prepare_dataset(dataset)
        
        y_pred = []
        for i in tqdm(range(0, len(prepared_dataset), self.config.batch_size)):
            batch = prepared_dataset['message_for_llm'][i:i + self.config.batch_size]
            y_pred.extend(self.predict_batch(batch))
            
        y_true = dataset['category']
        y_pred_norm = self.normalize_predictions(y_pred)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred_norm, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred_norm, labels=self.list_of_categories)
        
        # Visualize results
        self.plot_confusion_matrix(conf_matrix, f"{split_name}_confusion_matrix.png")
        
        # Save results
        results = {
            'classification_report': report,
            'predictions': y_pred_norm,
            'true_labels': y_true
        }
        
        with open(f"{self.config.results_dir}/{split_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, filename: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            xticklabels=self.list_of_categories,
            yticklabels=self.list_of_categories,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f"{self.config.results_dir}/{filename}")
        plt.close()

def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = ModelConfig()
        
        # Initialize classifier
        classifier = LLMTextClassifier(config)
        
        # Evaluate on validation set
        logger.info("Starting validation set evaluation")
        val_results = classifier.evaluate(classifier.validation_set, 'validation')
        
        # Evaluate on test set
        logger.info("Starting test set evaluation")
        test_results = classifier.evaluate(classifier.test_set, 'test')
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()