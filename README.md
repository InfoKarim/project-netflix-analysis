# 🎬 Netflix Content Analysis & Prediction

This project explores and analyzes Netflix’s library of titles using Python, and builds a machine learning model to classify titles as **Movies** or **TV Shows** based on their features.

---

## 📌 Project Goals
- Perform Exploratory Data Analysis (EDA) on Netflix content.
- Understand patterns across **content type**, **release years**, **countries**, and **genres**.
- Build a predictive ML model to classify content type.
- Visualize key insights using matplotlib and seaborn.

---

## 🧰 Tools Used
- Python, pandas, matplotlib, seaborn
- scikit-learn, TfidfVectorizer, Logistic Regression
- Tableau (optional dashboard)
- Jupyter Notebook

---

## 📊 Key EDA Insights
- Netflix has more **Movies** than **TV Shows**.
- Most content was released between **2010–2020**.
- The **United States**, **India**, and **UK** are top producers.
- Dominant genres include **Dramas**, **Documentaries**, and **Comedies**.

---

## 🤖 ML Model Summary
- Model: Logistic Regression
- Features: `listed_in`, `release_year`, `duration`
- **Accuracy:** 99.94%
- Saved model file: `output_netflix_model.pkl`

---

## 📁 Project Structure

```
project-netflix-analysis/
├── data/
│   └── netflix_titles.csv
├── notebooks/
│   └── netflix_eda_ml.ipynb
│   └── netflix_type_classifier.py
├── output/
│   ├── output_content_type_distribution.png
│   ├── output_release_year_distribution.png
│   ├── output_top_countries.png
│   ├── output_model_report.txt
│   └── output_netflix_model.pkl
└── README.md
```

---

## 📬 Contact
Karim Elsayed 
- 📧[info.karimelsayed@gmail.com](mailto:info.karimelsayed@gmail.com)  
- LinkedIn:** [linkedin.com/in/karim-elsayed](https://www.linkedin.com/in/karim-elsayed-b6791011a/)
- 🔗 [GitHub](https://github.com/InfoKarim) 
- **Portfolio Website:** *Coming Soon*
