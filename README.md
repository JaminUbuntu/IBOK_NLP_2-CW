7120CEM - Natural Language Processing (Part 2)

# 🤖 Robustness of Transformer-Based Models in Noisy Sentiment Analysis  
_Deep Learning Evaluation with BERT vs BiLSTM on Twitter Data_

---

## 🔍 Overview

Transformer models like **BERT** are leading the NLP space — but how do they hold up under noisy, real-world inputs like Twitter typos or slang? This study evaluates the **robustness** of **BERT** and a **BiLSTM** model on **Twitter sentiment analysis**, using datasets from **SemEval 2015 & 2017**. We apply **linguistic noise**, test both models under clean and perturbed conditions, and provide a comparative performance report.

---

## 🧠 Project Objectives

- Compare BiLSTM (GloVe) and BERT (HuggingFace) on sentiment classification.
- Simulate **keyboard typos** and **semantic noise** using `nlpaug`.
- Quantify performance degradation across metrics: Accuracy, Precision, Recall, F1, AUC.
- Visualize and interpret model behavior under adversarial input.

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Frameworks:** TensorFlow, PyTorch
- **Libraries:** HuggingFace Transformers, sklearn, wandb, nlpaug
- **Visualization:** seaborn, matplotlib
- **Embeddings:** GloVe (100d)

---

## 📁 Dataset

The dataset includes two real-world Twitter corpora:

- **SemEval-2017** (70% Train / 20% Test)
- **SemEval-2015** (Validation)
- **Noise:** Applied using `nlpaug.keyboardAug` and synonym replacement.

[Dataset Link](https://github.com/leelaylay/TweetSemEval/tree/master/dataset)

---

## 🔬 Methodology

1. **Preprocessing:** Clean text, tokenize, label encode.
2. **Models:** BiLSTM with GloVe and BERT (fine-tuned).
3. **Noise Injection:** Simulated typos and synonyms.
4. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC.
5. **Class Handling:** Class Weighting instead of SMOTE/RUS.

---

## 📈 Results

| Model     | Clean F1 | Noisy F1 | AUC (Clean) | AUC (Noisy) |
|-----------|----------|----------|-------------|-------------|
| BiLSTM    | 0.6021   | 0.3299   | 0.7752      | 0.7689      |
| BERT      | 0.6962   | 0.6228   | 0.8347      | 0.7689      |

---

## 🧩 Real-World Applications

- **Chatbots & Moderation Tools**
- **Adversarial Input Detection**
- **Social Media Sentiment Analysis**
- **Robust NLP APIs for Industry Use**

---

## 🔗 How to Leverage This Repository

```bash
git clone https://github.com/JaminUbuntu/IBOK_NLP_2-CW.git
cd IBOK_NLP_2-CW
pip install -r requirements.txt
```

- Main Notebook: `IBOK_NLP_DL_CW.ipynb`
- Analyze clean vs noisy performance
- Confusion Matrices and Visualizations in `/assets` and `/outputs`

---

## 🧭 Future Directions

- Add multilingual/multitask variants like RoBERTa or T5
- Explore back-translation or paraphrasing augmentation
- Deploy adversarial training pipelines
- Use LIME/SHAP for explainability

---

## 📚 Citation

```text
Ibok, B. (2025). Robustness of Transformer-Based Models Against Linguistic Noise and Adversarial Inputs in Social Media Sentiment Tasks. Coventry University.
```

---

## 🎓 Academic Context

This project was developed for the **7120CEM – Natural Language Processing** module at **Coventry University**.

---

## 📬 Contact

**Author:** Benjamin Ibok  
**Institution:** Coventry University  
**Email:** ibokb@coventry.ac.uk  
**Personal Email:** benjaminsibok@gmail.com  

---

## ⚙️ Environment Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Dependencies:
- Python ≥ 3.8
- `transformers`, `torch`, `tensorflow`
- `sklearn`, `nlpaug`, `wandb`, `matplotlib`, `seaborn`, `pandas`

---

## 📊 Visualizations & Evaluation

Visual insights:
- WordClouds for noisy tokens
- Clean vs Noisy Confusion Matrices
- KDE plot of tweet length distributions
- ROC & AUC curves

---

## 🤝 Contribution Guidelines

1. Fork the repo
2. `git checkout -b your-feature`
3. `git commit -m "Add feature"`
4. `git push origin your-feature`
5. Open a Pull Request

Follow PEP8 and document your code.

---

## 💾 Model Saving

```python
torch.save(model.state_dict(), 'bert_model.pt')
model.load_state_dict(torch.load('bert_model.pt'))
model.eval()
```

---

## 🏷️ Project Badges

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Colab](https://img.shields.io/badge/platform-Colab-green)
![Model: BERT](https://img.shields.io/badge/model-BERT-orange)
![Status: Completed](https://img.shields.io/badge/status-completed-brightgreen)

---

## ❓ FAQ

- **Q: Can this run on CPU only?**  
  A: BiLSTM, yes. BERT requires GPU (Google Colab preferred).

- **Q: Why does BERT outperform BiLSTM under noise?**  
  A: BERT uses contextual embeddings, unlike static GloVe.

---
