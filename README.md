# 🧹 DataPrepKit: A Python Library for Data Preprocessing

**DataPrepKit** is a lightweight, modular library designed to simplify common data preprocessing tasks for machine learning workflows. It provides intuitive methods to read files, summarize data, handle missing values, and encode categorical features — all in a reusable and object-oriented manner.

---

## 📦 Features

- 📖 **Flexible File Reading**  
  Supports `.csv`, `.json`, and Excel (`.xls`, `.xlsx`) formats.

- 📊 **Descriptive Summaries & Statistics**  
  Generate quick statistical summaries of numerical data.

- 🧩 **Missing Value Handling**  
  Drop missing rows or fill them using mean, median, or mode imputation.

- 🏷️ **Categorical Encoding**  
  Encode categorical variables using One-Hot Encoding or Label Encoding.

---

## 🧰 Components

- `DataPrepKit`: Main controller class
- `DataReading`: File loader for multiple formats
- `Summary`: Descriptive statistics and custom stats
- `MissingValuesHandler`: Tools for handling NaNs
- `CategoricalEncoding`: One-hot and label encoding methods

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DataPrepKit.git
cd DataPrepKit
