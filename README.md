# 🤖 Robustness of Transformer-Based Models in Noisy Sentiment Analysis  
## _Deep Learning Evaluation with BERT vs BiLSTM on Twitter Data_

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

| Component        | Tool/Library            |
|------------------|-------------------------|
| Language         | Python                  |
| Frameworks       | TensorFlow, PyTorch     |
| DL Tools         | HuggingFace Transformers |
| Preprocessing    | nlpaug, sklearn         |
| Visualizations   | seaborn, matplotlib     |
| Tracking         | wandb                   |
| Embeddings       | GloVe (100d)            |
| Model Types      | BiLSTM, BERT            |

---

## 📁 Dataset

| Source      | Type          | Details                                       |
|-------------|---------------|-----------------------------------------------|
| SemEval-2017| Train/Test    | 70/20 split for BERT/BiLSTM training/testing  |
| SemEval-2015| Validation    | For early-stopping and robustness tuning      |
| Noise       | Injected      | Simulated with `nlpaug` (typos & synonyms)    |
| 📎 [Dataset Link](https://github.com/leelaylay/TweetSemEval/tree/master/dataset)

---

## 🔬 Methodology

1. **Preprocessing:** Text cleaning, label encoding, tokenization.
2. **Modeling:**
   - **BiLSTM:** Static GloVe Embeddings, TensorFlow/Keras, Dropout.
   - **BERT:** Pretrained `bert-base-uncased`, fine-tuned using HuggingFace.
3. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.
4. **Noise Testing:** Augmentations using `nlpaug.keyboardAug` and synonym injection.
5. **Class Handling:** Label imbalance solved using **class weights**.

---

## 📊 Results Summary

| Model     | Clean F1 | Noisy F1 | AUC (Clean) | AUC (Noisy) |
|-----------|----------|----------|-------------|-------------|
| BiLSTM    | 0.6021   | 0.3299   | 0.7752      | 0.7689      |
| BERT      | 0.6962   | 0.6228   | 0.8347      | 0.7689      |

---

## ⚙️ Environment & Setup Instructions

```bash
# Clone repository
git clone https://github.com/JaminUbuntu/IBOK_NLP_2-CW.git
cd IBOK_NLP_2-CW

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` includes:
- `transformers`
- `torch`
- `tensorflow`
- `nlpaug`
- `scikit-learn`
- `wandb`
- `seaborn`, `pandas`, `matplotlib`

---

## 🗂️ Folder Structure

```
IBOK_NLP_2-CW/
├── IBOK_NLP_DL_CW.ipynb
├── README.md
├── requirements.txt
├── /outputs
├── /models
└── /assets
```

---

## 🧠 Model Interpretability

- Confusion Matrices
- Word Clouds
- F1/AUC Comparisons
- KDE Tweet Length Analysis
- Integration-ready for SHAP/LIME

---

## 🖼️ Sample Outputs & Screenshots

Add your figures in `/assets` and link here (example below):

- ![Confusion Matrix](assets/confusion_clean.png)
- ![AUC Comparison](assets/auc_comparison.png)

---

## 🧬 Model Persistence

```python
# Save model
torch.save(model.state_dict(), 'bert_model.pt')

# Load model
model.load_state_dict(torch.load('bert_model.pt'))
model.eval()
```

---

## 💬 Contribution Guidelines

1. Fork the repo.
2. Create a feature branch.
3. Commit changes.
4. Open a pull request.

Follow PEP8 and include docstrings.

---

## 🎓 Academic Context

Part of the coursework for **7120CEM – Natural Language Processing** at **Coventry University**.

---

## 🧭 Future Directions

- Add multilingual BERT/RoBERTa.
- Use adversarial training.
- Implement T5, XLNet, DistilBERT.
- Add explainability via SHAP/LIME.
- Introduce cross-domain noise tests.

---

## 📛 Badge Flair

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![BERT](https://img.shields.io/badge/model-BERT-orange)
![Colab](https://img.shields.io/badge/platform-Colab-green)
![Status: Completed](https://img.shields.io/badge/status-completed-brightgreen)

---

## ❓ FAQ / Known Issues

- **Q:** Why does BiLSTM perform worse under noise?
  **A:** GloVe is static; it lacks contextual awareness.

- **Q:** Can I run this without GPU?
  **A:** BERT training requires a GPU. Use Colab.

---

## 📚 Citation

```text
Ibok, B. (2025). Robustness of Transformer-Based Models Against Linguistic Noise and Adversarial Inputs in Social Media Sentiment Tasks. Coventry University.
```