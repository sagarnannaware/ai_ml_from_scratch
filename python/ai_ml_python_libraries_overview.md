# Python AI/ML Libraries — Usage & Functionality Overview

---

## NumPy
- **Usage:** Foundational library for numerical computing in Python. Supports fast operations on large arrays and matrices, linear algebra, random number generation.
- **Key Functionalities:**
  - Multidimensional arrays (`ndarray`)
  - Broadcasting for arithmetic operations
  - Linear algebra (dot product, matrix multiplication)
  - Random number generation
  - Integration with other libraries (Pandas, Scikit-learn, etc.)

---

## Pandas
- **Usage:** Data manipulation, cleaning, and analysis. Ideal for tabular data (think CSV, Excel, SQL).
- **Key Functionalities:**
  - DataFrame and Series structures for data storage
  - Easy handling of missing data
  - Data filtering, aggregation, grouping
  - Merging, joining, reshaping datasets
  - Reading/writing data from/to multiple formats

---

## Matplotlib & Seaborn
- **Usage:** Data visualization. Matplotlib is the base plotting library; Seaborn builds on it for statistical and attractive plots.
- **Key Functionalities:**
  - Line, bar, scatter, histogram, pie charts (Matplotlib)
  - Heatmaps, pairplots, violin plots (Seaborn)
  - Customizable figures and axes
  - Support for publication-level graphics

---

## Scikit-learn
- **Usage:** Classical machine learning library. Provides tools for modeling, evaluation, and preprocessing.
- **Key Functionalities:**
  - Algorithms for classification, regression, clustering, dimensionality reduction
  - Model selection, hyperparameter tuning (GridSearchCV)
  - Data preprocessing: scaling, encoding, imputation
  - Pipelines for reproducible workflows

---

## TensorFlow & Keras
- **Usage:** Deep learning and neural networks. TensorFlow is a high-performance framework; Keras is its user-friendly API.
- **Key Functionalities:**
  - Build, train, and deploy neural networks (CNNs, RNNs, etc.)
  - GPU acceleration
  - Model serialization and export
  - Handling of large-scale datasets
  - Deployment to cloud, edge, or mobile

---

## PyTorch
- **Usage:** Deep learning with a focus on flexibility and research prototyping.
- **Key Functionalities:**
  - Dynamic computation graphs (eager execution)
  - Neural network modules (torch.nn)
  - GPU acceleration
  - Native support for custom layers and operations
  - Widely used in academia and cutting-edge research

---

## OpenCV
- **Usage:** Computer vision and image processing.
- **Key Functionalities:**
  - Image and video reading/writing
  - Image transformation (resize, crop, rotate, etc.)
  - Feature detection (edges, faces, objects)
  - Real-time computer vision (webcam, video processing)
  - Integration with deep learning models

---

## NLTK & spaCy
- **Usage:** Natural Language Processing (NLP)
- **NLTK:** Great for education, research, and prototyping. Extensive text processing, tokenization, stemming, tagging.
- **spaCy:** Fast, production-ready NLP. Industrial-strength tokenization, POS, NER, dependency parsing.
- **Key Functionalities:**
  - Text tokenization, lemmatization, stemming
  - Part-of-speech tagging, named entity recognition
  - Syntactic parsing (spaCy)
  - Text classification, sentiment analysis

---

## HuggingFace Transformers
- **Usage:** State-of-the-art models for NLP (BERT, GPT, etc.), vision, audio, and multi-modal tasks.
- **Key Functionalities:**
  - Easy access to pre-trained models
  - Simple APIs for text generation, classification, translation, etc.
  - Model fine-tuning
  - Support for PyTorch and TensorFlow

---

## XGBoost & LightGBM
- **Usage:** Gradient boosting libraries for high-performance tabular modeling.
- **Key Functionalities:**
  - Fast, scalable training for classification/regression
  - Handles missing values, categorical features
  - Model export for deployment
  - Feature importance and visualization tools

---

## Joblib & Pickle
- **Usage:** Model and object serialization (saving/loading models to disk)
- **Key Functionalities:**
  - `joblib`: Efficient for large numpy arrays, scikit-learn models
  - `pickle`: General-purpose Python object serialization

---

## FastAPI & Flask
- **Usage:** Exposing Python code (including ML models) as web APIs.
- **Key Functionalities:**
  - FastAPI: Modern, async, automatic OpenAPI docs, robust input validation
  - Flask: Micro web framework, simple and flexible, good for quick prototypes

---