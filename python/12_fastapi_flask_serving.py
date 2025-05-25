"""
FastAPI & Flask — Model Serving

Description:
FastAPI and Flask are web frameworks to turn your Python code (including AI/ML models) into RESTful APIs for production use.

Key Functionalities:
- FastAPI: Modern, async, automatic OpenAPI docs, robust input validation
- Flask: Lightweight, simple, flexible, good for quick prototypes
- Expose models as web services

Sample Code:
"""

# FastAPI Example
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Flask Example
from flask import Flask

flask_app = Flask(__name__)

@flask_app.route("/")
def hello():
    return "Hello World!"

print("API endpoints defined for FastAPI and Flask.")