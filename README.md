# SMS Spam Classifier 📱

This project aims to classify SMS messages into spam or not spam categories using machine learning techniques. It predicts whether a given SMS message is spam or not.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Cleaning](#data-cleaning)
3. [Exploratory Data Analysis (EDA)](#eda)
4. [Text Preprocessing](#text-preprocessing)
5. [Model Building](#model-building)
6. [Evaluation](#evaluation)
7. [Improvement](#improvement)
8. [Streamlit Website](#streamlit-website)
9. [Deployment on Streamlit](#deployment-on-streamlit)
10. [Acknowledgements](#acknowledgements)
11. [Repository Files](#repository-files)

## Introduction 🚀

This project classifies SMS messages into spam or not spam categories using various machine learning algorithms. It's built under the guidance of Campus X, especially thanks to Nitish Sir.

## Data Cleaning 🧹

The dataset undergoes cleaning to handle missing values, duplicates, or any inconsistencies.

## EDA 🔍

Exploratory Data Analysis involves analyzing the dataset to gain insights and understand its characteristics.

## Text Preprocessing ✨

Text preprocessing is crucial in NLP tasks. It includes lowercasing, tokenization, removing special characters, stop words, punctuation, and stemming.

## Model Building 🏗️

Several ML algorithms are trained and evaluated to build the SMS spam classifier.

## Evaluation 📊

The performance of each model is evaluated using metrics such as accuracy and precision.

## Improvement 🛠️

Potential improvements and optimizations for the classifier are discussed.

## Streamlit Website 🌐

The SMS spam classifier is deployed as a web application using Streamlit.

## Deployment on Streamlit 🚀

Instructions for deploying the classifier on Streamlit are provided.

## Acknowledgements 🙏

Special thanks to Campus X - Nitish Sir for their guidance and support in building this project.

## Repository Files

- `.idea`: Contains project-related settings and configurations for JetBrains IDEs.
- `Procfile.txt`: Specifies the commands that are executed by the app on Heroku.
- `app.py`: Python script containing the Streamlit application code for the SMS spam classifier.
- `gitignore.txt`: Specifies intentionally untracked files to ignore.
- `model.pkl`: Trained machine learning model for classifying SMS messages.
- `nltk.txt`: Contains NLTK data dependencies.
- `requirements.txt`: Lists the Python packages required for the project.
- `setup.sh`: Shell script for setting up the project environment.
- `sms-spam-detection1.ipynb`: Jupyter Notebook containing the code used for training the model and analyzing the dataset.
- `spam.csv`: Dataset containing SMS messages labeled as spam or not spam.
- `spam_image.png`: Image file used in the Streamlit app to display when a message is classified as spam.
- `vectorizer.pkl`: Vectorizer used for preprocessing text data.

These files are essential components of the SMS spam classifier project and are used for various purposes such as code implementation, model training, data analysis, and deployment.
