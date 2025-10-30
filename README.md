# 💊 Drug Review Rating Prediction

Predicting patient satisfaction scores from drug reviews using **TF-IDF** and **Transformer embeddings**  
*(based on the Drugs.com Reviews Dataset)*

---

## 📌 Overview
This project compares two NLP approaches for predicting drug review ratings (1–10):

- 🧩 **TF-IDF + Logistic Regression** — interpretable, keyword-based baseline  
- 🧠 **MiniLM Sentence Embeddings + Logistic Regression** — semantic, context-aware model  

Both were tested under:
- **3-class sentiment setup** (negative / neutral / positive)  
- **10-class exact rating setup**

> 📄 The full methodology, detailed analysis, and visualizations are available in the **accompanying report**.

---

## ⚙️ Methodology
- **Text Preprocessing:** HTML tag removal, lemmatization, and lowercasing  
- **Feature Composition:** `review + [DRUG] drugName + [COND] condition`  
- **Modeling:** Logistic Regression with GridSearchCV tuning  
- **Evaluation Metrics:** Accuracy, Macro-F1, Weighted-F1, Confusion Matrix  

---

## 🧳 Results

| Model | Classes | Accuracy | Macro-F1 | Weighted-F1 |
|:------|:---------|:----------|:----------|:--------------|
| TF-IDF + LR | 3 | **0.8739** | **0.7869** | **0.8745** |
| TF-IDF + LR | 10 | 0.6530 | 0.6019 | 0.6526 |
| MiniLM + LR | 3 | 0.7420 | 0.4769 | 0.7001 |
| MiniLM + LR | 10 | 0.3889 | 0.1619 | 0.3080 |

✅ The **simpler TF-IDF approach outperformed** the Transformer-based model - delivering higher accuracy, stability, and interpretability.

---

## 💡 Key Insights
- The **3-class setup** (negative / neutral / positive) aligned better with how users express satisfaction.  
- **Neutral ratings (5–6)** were hardest to classify - the MiniLM model often confused them with adjacent positive or negative classes.  
- **Transformer embeddings** captured general sentence meaning but not emotional nuance or intensity.  
- **TF-IDF**, despite being a more *naive* approach, effectively captured strong lexical sentiment cues (e.g. “love it”, “not recommend”).  
- The findings emphasize that **simpler models can outperform complex ones** when dealing with clear lexical sentiment patterns.

---

## 🚀 Future Work
- Experiment with **more advanced architectures**, such as fine-tuned transformers or neural models (e.g., LSTM, CNN).  
- Explore **domain-specific pretraining** on medical or healthcare review data.  
- Combine **lexical (TF-IDF)** and **semantic (embeddings)** features into a hybrid model.  
- Develop improved visualization methods for **feature interpretation** and sentiment trends.  
- Investigate **class imbalance handling** and methods to better capture neutral sentiments.  

---

## 🧮 Technologies
Python · scikit-learn · NLTK · SentenceTransformers · pandas · matplotlib  

---

## 📂 Project Structure
Drug-Review-Rating-Prediction/
│
├── 📘 Drug_Reviews_TFIDF_vs_Embeddings.ipynb # Main Jupyter notebook
├── 📄 Drug_Reviews_Report.pdf # Full report with analysis
│
├── 📁 data/
│ ├── drugsComTrain_raw.csv # Training dataset
│ └── drugsComTest_raw.csv # Test dataset
│
└── 📜 README.md
---

## 👤 Author
**Asaf Baroch**  
B.Sc. Software & Information Systems Engineering - Ben-Gurion University  
📧 asafbaruch81@gmail.com · [GitHub Profile](https://github.com/asafbaroch)