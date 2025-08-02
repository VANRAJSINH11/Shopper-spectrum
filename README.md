# 🛍️ Shopper Spectrum - Customer Segmentation & Recommendation System

This project uses **RFM analysis** and **unsupervised learning** to segment customers based on their behavior and provide personalized product recommendations using **cosine similarity**.

## 📌 Features

- 📊 **Customer Segmentation** using RFM (Recency, Frequency, Monetary) model
- 🤖 **KMeans Clustering** for behavioral grouping
- 🧠 **Product Recommendation System** using similarity matrix
- 💻 Built with **Flask**, **Pandas**, **Scikit-learn**
- 🌐 Easy-to-use Web Interface

## 📁 Project Structure
│
├── app.py # Main Flask application
├── data.csv # Input customer transaction data
├── requirements.txt # Python dependencies
├── rfm_model.pkl # Saved KMeans model
├── similarity_matrix.pkl # Saved product similarity matrix

🧪 How It Works
🎯 RFM Segmentation
Recency: Days since last purchase
Frequency: Number of transactions
Monetary: Total amount spent

🧩 KMeans Clustering
Segments customers into categories like:
🟢 High-Value
🟡 Regular
🔵 Occasional
🔴 At-Risk

🛒 Recommendations
Based on item co-occurrence and cosine similarity
🧠 Technologies Used
Python
Flask
Pandas
Scikit-learn
NumPy
Pickle
