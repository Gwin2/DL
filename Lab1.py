import logging
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import spacy
from datasets import load_dataset
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the text classifier"""
    dataset_name: str = 'Davlan/sib200'
    dataset_language: str = 'rus_Cyrl'
    spacy_model: str = 'ru_core_news_sm'
    results_dir: str = 'results'
    model_dir: str = 'models'
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    max_iter: int = 100
    
    # Grid search parameters
    ngram_ranges: List[Tuple[int, int]] = None
    C_values: List[float] = None
    penalties: List[str] = None
    
    def __post_init__(self):
        if self.ngram_ranges is None:
            self.ngram_ranges = [(1, 1), (1, 2), (1, 3)]
        if self.C_values is None:
            self.C_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        if self.penalties is None:
            self.penalties = ['l1', 'l2']

class TextClassifier:
    """Text classifier for Russian language"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.setup_directories()
        
        logger.info("Initializing text classifier")
        self.load_spacy()
        self.load_datasets()
        self.setup_classifier()
        
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.results_dir, self.config.model_dir]:
            Path(directory).mkdir(exist_ok=True)
            
    def load_spacy(self):
        """Load Spacy model"""
        logger.info(f"Loading Spacy model: {self.config.spacy_model}")
        try:
            self.nlp = spacy.load(self.config.spacy_model)
            logger.info("Spacy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Spacy model: {str(e)}")
            raise
            
    def normalize_text(self, text: str) -> str:
        """Normalize text using Spacy"""
        try:
            doc = self.nlp(text)
            lemmas = [
                ('<NUM>' if token.like_num else token.lemma_.lower())
                for token in filter(lambda t: not t.is_punct, doc)
            ]
            return ' '.join(lemmas) if lemmas else ''
        except Exception as e:
            logger.warning(f"Error normalizing text: {str(e)}")
            return text
            
    def load_datasets(self):
        """Load and prepare datasets"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        try:
            # Load datasets
            self.train_set = load_dataset(self.config.dataset_name, self.config.dataset_language, split='train')
            self.val_set = load_dataset(self.config.dataset_name, self.config.dataset_language, split='validation')
            self.test_set = load_dataset(self.config.dataset_name, self.config.dataset_language, split='test')
            
            # Extract features and labels
            self.X_train = self.train_set['text']
            self.y_train = self.train_set['category']
            self.X_val = self.val_set['text']
            self.y_val = self.val_set['category']
            self.X_test = self.test_set['text']
            self.y_test = self.test_set['category']
            
            # Setup categories
            self.setup_categories()
            
            logger.info("Datasets loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load datasets: {str(e)}")
            raise
            
    def setup_categories(self):
        """Setup category mappings"""
        # Get unique categories
        categories = set(self.y_train)
        
        # Check for unknown categories in validation set
        unknown_categories = set(self.y_val) - categories
        if unknown_categories:
            raise ValueError(f'Unknown categories in validation set: {unknown_categories}')
            
        # Check for unknown categories in test set
        unknown_categories = set(self.y_test) - categories
        if unknown_categories:
            raise ValueError(f'Unknown categories in test set: {unknown_categories}')
            
        # Sort categories and create mappings
        self.categories = sorted(list(categories))
        self.n_categories = len(self.categories)
        
        # Convert labels to indices
        self.y_train = [self.categories.index(label) for label in self.y_train]
        self.y_val = [self.categories.index(label) for label in self.y_val]
        self.y_test = [self.categories.index(label) for label in self.y_test]
        
        logger.info(f"Found {self.n_categories} categories")
        
    def setup_classifier(self):
        """Setup the classification pipeline"""
        # Calculate class probability for max_df
        class_probability = 1.0 / self.n_categories
        max_df = 1.0 - 0.2 * class_probability
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                token_pattern=r'\w+',
                max_df=max_df,
                min_df=1
            )),
            ('classifier', LogisticRegression(
                solver='saga',
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            ))
        ])
        
        # Setup grid search
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid={
                'vectorizer__ngram_range': self.config.ngram_ranges,
                'classifier__C': self.config.C_values,
                'classifier__penalty': self.config.penalties
            },
            scoring='f1_macro',
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
    def train(self):
        """Train the model"""
        logger.info("Starting model training")
        
        try:
            # Normalize texts
            logger.info("Normalizing training texts")
            X_train_norm = [
                self.normalize_text(text)
                for text in tqdm(self.X_train, desc="Normalizing training texts")
            ]
            
            # Train model
            logger.info("Training model with grid search")
            self.grid_search.fit(X_train_norm, self.y_train)
            
            # Save best parameters
            best_params = self.grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
            
            with open(os.path.join(self.config.results_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
                
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate(self, split: str = 'validation') -> Dict[str, Any]:
        """Evaluate model on specified split"""
        logger.info(f"Evaluating model on {split} split")
        
        try:
            # Get appropriate dataset
            X = self.X_val if split == 'validation' else self.X_test
            y = self.y_val if split == 'validation' else self.y_test
            
            # Normalize texts
            X_norm = [
                self.normalize_text(text)
                for text in tqdm(X, desc=f"Normalizing {split} texts")
            ]
            
            # Make predictions
            y_pred = self.grid_search.predict(X_norm)
            
            # Calculate metrics
            report = classification_report(
                y_true=y,
                y_pred=y_pred,
                target_names=self.categories,
                output_dict=True
            )
            
            conf_matrix = confusion_matrix(
                y_true=y,
                y_pred=y_pred,
                labels=range(self.n_categories)
            )
            
            # Visualize results
            self.plot_confusion_matrix(conf_matrix, f"{split}_confusion_matrix.png")
            
            # Save results
            results = {
                'classification_report': report,
                'predictions': y_pred.tolist(),
                'true_labels': y
            }
            
            with open(os.path.join(self.config.results_dir, f'{split}_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
                
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, filename: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            xticklabels=self.categories,
            yticklabels=self.categories,
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
        classifier = TextClassifier(config)
        
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