# 703-HYD

Analysing a decade of multilingual Twitter data to uncover public discourse patterns around hydrogen energy (2013–2022).

---

## 📋 Overview

This project processes approximately 30 million tweets collected via the Twitter Academic API using hydrogen-related keywords across English, Japanese, Korean, and Hindi. The goal is to understand how public conversation around hydrogen energy has evolved over time — and what drives engagement with that content.

The pipeline runs from raw data ingestion through translation, NLP-based feature extraction, and time-series engagement modelling using three deep learning architectures.

---

## 📁 Project Structure

```
703-HYD/
├── 1_Data_Sample.py                               # Temporal sampling from raw dataset
├── 2_Data_Translate.ipynb                         # Multilingual → English translation
├── 3_Data_Processing.ipynb                        # NLP cleaning, sentiment, embeddings
├── 4_Identifying_Key_Drivers_of_Engagement.ipynb  # Correlation analysis
├── Data_Process.py                                # Engagement scoring & daily aggregation
├── Data_Display.ipynb                             # Trend visualisation
├── Bert_Model_Development.ipynb                   # BERT-based engagement predictor
├── LSTM_Model_Development.ipynb                   # LSTM engagement predictor
├── RoBertA_Model_Development.ipynb                # RoBERTa-based engagement predictor
├── tokenization_small100.py                       # SMaLL-100 tokenisation utility
├── best_bert_model.pth                            # Saved BERT weights
├── best_roberta_model.pth                         # Saved RoBERTa weights
└── data_processed_filtered*.csv                   # Intermediate pipeline outputs
```

---

## 🔄 Pipeline

**Sampling** — The raw dataset is sorted chronologically and evenly sampled to a manageable subset for NLP processing.

**Translation** — Non-English tweets are translated to English using Facebook's M2M100 multilingual model, enabling unified downstream analysis regardless of source language.

**Processing** — Tweets are cleaned, tokenised, and enriched with sentiment scores (VADER) and dense semantic embeddings (SBERT) for both text and metadata such as hashtags and mentions.

**Engagement Scoring** — A weighted engagement metric is computed per tweet (combining likes, retweets, and replies) and aggregated to daily granularity for time-series modelling.

**Driver Analysis** — Pearson correlation and linear regression identify which features — sentiment, hashtag usage, mentions, user influence, time of day, day of week — most strongly predict engagement.

**Modelling** — Three sequence models are trained to predict daily engagement trends: a stacked LSTM, and two custom transformer architectures inspired by BERT and RoBERTa. All models are evaluated on engagement prediction accuracy as well as peak detection — identifying spikes in public attention.

---

## ⚙️ Requirements

```bash
pip install torch tensorflow transformers sentence-transformers pandas numpy scikit-learn matplotlib scipy nltk tqdm
```

> The raw dataset is not included due to size. Processed CSV checkpoints and pretrained model weights (`best_bert_model.pth`, `best_roberta_model.pth`) are provided for reproducibility without retraining.
