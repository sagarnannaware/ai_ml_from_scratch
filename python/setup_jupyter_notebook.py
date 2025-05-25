"""
Setting Up Jupyter Notebook for Python AI/ML Projects

Description:
Jupyter Notebook is an open-source web application that allows you to create and share documents containing live code, equations, visualizations, and narrative text. It's widely used in data science, AI, and ML for interactive development, visualization, and reporting.

Key Functionalities:
- Interactive coding and immediate feedback
- Supports code, markdown, visualizations, and rich outputs
- Easy integration with libraries like NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow, PyTorch, etc.
- Suitable for experimentation, prototyping, and presentations

Setup Instructions:
1. Create and activate a Python virtual environment (recommended for isolation)
2. Install Jupyter using pip inside the environment
3. Launch the Jupyter Notebook server
4. Create and work with .ipynb notebook files in your browser

Commands to Setup Virtual Environment and Start Jupyter:

# On Linux/macOS:
python3 -m venv venv
source venv/bin/activate
pip install notebook

# On Windows:
python -m venv venv
venv\Scripts\activate
pip install notebook

# Start Jupyter Notebook (after activation and installation):
jupyter notebook

Sample Code to Try in a Notebook Cell:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.randn(100)
df = pd.DataFrame({'values': data})

plt.hist(df['values'])
plt.title("Histogram in Jupyter Notebook")
plt.show()