import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import evaluate
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rubert_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the RuBERT classifier"""
    model_name: str = 'DeepPavlov/rubert-base-cased'
    dataset_name: str = 'Davlan/sib200'
    dataset_language: str = 'rus_Cyrl'
    output_dir: str = 'rubert_sib200'
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 1e-3
    max_length: int = 512
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    results_dir: str = 'results'

class RuBERTClassifier:
    """RuBERT-based text classifier for Russian language"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.setup_directories()
        self.setup_device()
        
        logger.info(f"Initializing RuBERT classifier with model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.load_datasets()
        self.setup_categories()
        self.setup_model()
        
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.output_dir, self.config.results_dir]:
            Path(directory).mkdir(exist_ok=True)
            
    def setup_device(self):
        """Setup device and mixed precision training"""
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU")
            self.config.device = 'cpu'
        
        if self.config.mixed_precision and self.config.device == 'cuda':
            logger.info("Enabling mixed precision training")
        
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
        """Setup category mappings"""
        logger.info("Setting up categories")
        self.list_of_categories = sorted(list(
            set(self.train_set['category']) |
            set(self.validation_set['category']) |
            set(self.test_set['category'])
        ))
        self.n_categories = len(self.list_of_categories)
        
        indices = list(range(self.n_categories))
        self.id2label = dict(zip(indices, self.list_of_categories))
        self.label2id = dict(zip(self.list_of_categories, indices))
        
        logger.info(f"Found {self.n_categories} categories: {self.list_of_categories}")
        
    def setup_model(self):
        """Initialize the model"""
        logger.info("Setting up model")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.n_categories,
                id2label=self.id2label,
                label2id=self.label2id
            ).to(self.config.device)
            
            # Ensure contiguous parameters
            for param in self.model.parameters():
                param.data = param.data.contiguous()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
            
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize and prepare dataset"""
        tokenized = dataset.map(
            lambda x: self.tokenizer(
                x['text'],
                truncation=True,
                max_length=self.config.max_length
            ),
            batched=True,
            batch_size=self.config.batch_size
        )
        
        return tokenized.add_column(
            'label',
            [self.label2id[val] for val in tokenized['category']]
        )
        
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        metric = evaluate.load('f1')
        return metric.compute(
            predictions=predictions,
            references=labels,
            average='macro'
        )
        
    def train(self):
        """Train the model"""
        logger.info("Starting model training")
        
        # Prepare datasets
        tokenized_train = self.tokenize_dataset(self.train_set)
        tokenized_val = self.tokenize_dataset(self.validation_set)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            fp16=self.config.mixed_precision and self.config.device == 'cuda'
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        try:
            trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate(self, split: str = 'validation') -> Dict[str, Any]:
        """Evaluate model on specified split"""
        logger.info(f"Evaluating model on {split} split")
        
        # Setup classification pipeline
        classifier = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.config.device == 'cuda' else -1
        )
        
        # Get dataset
        dataset = self.validation_set if split == 'validation' else self.test_set
        
        # Make predictions
        try:
            with autocast(enabled=self.config.mixed_precision and self.config.device == 'cuda'):
                predictions = classifier(dataset['text'], batch_size=self.config.batch_size)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
            
        # Process predictions
        y_pred = [pred['label'] for pred in predictions]
        y_true = dataset['category']
        
        # Calculate metrics
        report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.list_of_categories)
        
        # Visualize results
        self.plot_confusion_matrix(conf_matrix, f"{split}_confusion_matrix.png")
        
        # Save results
        results = {
            'classification_report': report,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
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
        plt.savefig(os.path.join(self.config.results_dir, filename))
        plt.close()

def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = ModelConfig()
        
        # Initialize classifier
        classifier = RuBERTClassifier(config)
        
        # Train model
        classifier.train()
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set")
        val_results = classifier.evaluate('validation')
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_results = classifier.evaluate('test')
        
        logger.info("All operations completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()