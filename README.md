## 🎬 **Movie Recommendation System**  
### **Dynamix Networks – Machine Learning Internship (Task 2)**  

This project was developed as part of the **Machine Learning Internship at Dynamix Networks**.  
A complete **Hybrid Movie Recommendation System** is implemented using the **MovieLens Latest-Small dataset**.  

The system includes:  
- **Collaborative Filtering**  
- **Content-Based Filtering**  
- **Personalized User Recommendations**  
- **RMSE Evaluation**  
- **Local Streamlit User Interface for Demo**  


---

## 📁 **Dataset Information**  

**Dataset:** MovieLens Latest-Small  
**Users:** 610  
**Movies:** 9,742  
**Ratings:** 100,836  
**Source:** https://grouplens.org/datasets/movielens/latest/  

Files used:  
- `ratings.csv`  
- `movies.csv`  
- `tags.csv`  
- `links.csv`  

Folder structure:                                                                           
data/                                                                                           
  ratings.csv     
  movies.csv                                                                           
  tags.csv                  
  links.csv  



---

## 📌 **Core System Features**  

### **1️⃣ Collaborative Filtering**  
Recommends movies based on user–item rating similarities.  
Cosine similarity is used to find movies rated similarly to a selected movie.  

### **2️⃣ Content-Based Filtering**  
Uses:  
- TF-IDF on genres + tags  
- Cosine similarity  
Returns movies with similar attributes to the selected title.  

### **3️⃣ Personalized Recommendations**  
Predicts movie preferences for individual users using rating patterns.  

### **4️⃣ RMSE Evaluation**  
A baseline model is used to calculate RMSE and measure rating prediction error.  


---

## 🧪 **Model-Level Execution (Notebook)**  

1. Open the `Movie_Recommendation.ipynb` notebook in Google Colab or Jupyter.  
2. Run all cells step-by-step.  
3. View outputs including:  
   - EDA charts  
   - Collaborative filtering results  
   - Content-based similarity results  
   - Personalized top–N recommendations  
   - RMSE score  

_No additional configuration required for notebook execution._  


---

## 🖥 **Local Frontend Execution (Streamlit UI)**  

A **local interface** is provided for demonstration and video recording ease.  

### 🔧 Installation Requirements  
Run these commands once:  
pip install streamlit
pip install scikit-learn
pip install pandas
pip install numpy


### ▶️ Running the Interface  
Inside the project folder, run:  


Access UI at: 
http://localhost:8501

## 🌐 Live Application (Streamlit Cloud)

The Movie Recommendation System is deployed using **Streamlit Cloud** and can be accessed here:

🔗 **Live App:** https://ai-movie-recommendation-system.streamlit.app/

> Note: The app may take a few seconds to load due to cold start behavior on free hosting.


### UI Options  
- **Similar Movies (Collaborative Filtering)**  
- **Similar Movies (Content-Based Filtering)**  
- **Personalized User Recommendations**  

## 🚀 Deployment Details

- Platform: Streamlit Cloud
- Backend: Python
- Model Type: Hybrid Recommendation System
- Hosting Type: Free Tier (Educational / Internship Use)



---

## 🎥 **Recommended Demo Flow for Video**  

1. Launch Streamlit interface  
2. Select **Toy Story (1995)** → show CF recommendations  
3. Select **Jumanji (1995)** → show Content-Based results  
4. Select **User ID 1** → show personalized predictions  
5. Close with short summary referencing the GitHub repo  


---

## 📂 Project Structure

DynamixNetworks_Movie_Recommendation_System/       
│                                           
├── app.py                                   
├── requirements.txt                            
├── README.md                                 
├── .gitignore                                   
│                                               
├── .streamlit/                          
│   └── config.toml                                
├── data/                          
│   ├── ratings.csv                              
│   ├── movies.csv                               
│   ├── tags.csv                                 
│   └── links.csv                              
│                                        
└──          DynamixNetworks_ML_Internship_Movie_Recommendation.ipynb
                                  




---

## 📌 **Internship Compliance Checklist**  

| Requirement | Status |
|-------------|--------|
| Task 2 Completed | ✔ |
| Notebook Execution Verified | ✔ |
| Frontend UI Working Locally | ✔ |
| README Fully Updated | ✔ |
| GitHub Repository Uploaded | ✔ |
| LinkedIn Post + Video Explanation Done | ✔ |
| Dataset Usage Correct | ✔ |  


---

## 🙏 **Acknowledgements**  

Dataset sourced from **MovieLens / GroupLens Research**.  
Project completed under **Dynamix Networks Machine Learning Internship**.  





