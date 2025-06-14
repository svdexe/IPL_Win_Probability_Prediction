# 🏏 IPL Win Probability Predictor

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
</div>

<br>

<div align="center">
  <img src="https://github.com/svdexe/IPL_Win_Probability_Prediction/blob/main/Predictio_Website_Screen.png" alt="IPL Win Predictor Interface" width="700"/>
  <p><em>Real-time cricket match outcome prediction with interactive analytics dashboard</em></p>
</div>

<br>

## ✨ Features

🎯 **Real-time Predictions** · Get instant win probability predictions during live IPL matches  
📊 **Interactive Dashboard** · Clean and intuitive Streamlit web interface with visual analytics  
🏆 **Team Analysis** · Support for all 8 major IPL teams with historical data insights  
📈 **Match Progression** · Track probability changes throughout the match with visual indicators  
🎨 **Dynamic Visualization** · Progress bars and metrics showing current match situation  
🔍 **Detailed Analytics** · Required run rate, wickets remaining, and match state analysis

## 🛠️ Tech Stack

**Python** · Core programming language  
**Streamlit** · Web application framework  
**Pandas & NumPy** · Data manipulation and numerical analysis  
**Scikit-learn** · Machine learning algorithms and preprocessing  
**Logistic Regression** · Primary prediction model  
**Pickle** · Model serialization and deployment  
**CSS/HTML** · Custom styling for enhanced UI  

## 🚀 Quick Start

**Prerequisites**
```
Python 3.7+ · pip package manager
```

**Installation**
```bash
# Clone the repository
git clone https://github.com/svdexe/IPL_Win_Probability_Prediction.git
cd IPL_Win_Probability_Prediction

# Install dependencies
pip install streamlit pandas scikit-learn numpy pickle-mixin

# Run the application
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
IPL_Win_Probability_Prediction/
│
├── app.py                                    # Main Streamlit application
├── IPL_Win_Probability_Predictor.ipynb      # Jupyter notebook with model development
├── pipe.pkl                                 # Trained ML pipeline
├── matches.csv                              # IPL matches dataset
├── deliveries.csv                           # Ball-by-ball delivery data
├── requirements.txt                         # Python dependencies
├── README.md                               # Project documentation
└── Predictio_Website_Screen.png            # Interface screenshot
```

## 🔧 How It Works

**Data Processing** · Merges IPL matches and deliveries datasets · Filters second innings data for chase scenarios · Handles team name standardization and data cleaning

**Feature Engineering** · Calculates current run rate (CRR) and required run rate (RRR) · Tracks wickets remaining and balls left · Creates situational features like runs needed and match pressure

**Model Training** · Uses Logistic Regression with OneHotEncoder for categorical features · Trains on historical IPL data with 80-20 train-test split · Achieves high accuracy in predicting match outcomes

**Real-time Prediction** · Takes current match state as input · Processes through trained ML pipeline · Returns win probability for both teams with confidence intervals

## 🎯 Algorithm Details

The prediction system uses **Supervised Machine Learning**

**Model** · Logistic Regression with liblinear solver  
**Features** · Batting team, bowling team, city, runs left, balls left, wickets remaining, target score, CRR, RRR  
**Preprocessing** · OneHotEncoding for categorical variables, feature scaling  
**Evaluation** · Accuracy score, probability calibration  

## 📊 Dataset

**Source** · IPL Historical Data (2008-2020+)  
**Matches** · 800+ IPL matches  
**Deliveries** · 150,000+ ball-by-ball records  
**Teams** · 8 major IPL franchises  
**Features** · Match details, player statistics, venue information  

## 🎮 Usage

1. **Select Teams** · Choose batting and bowling teams from dropdown
2. **Set Match Context** · Select host city and target score
3. **Enter Current State** · Input current score, overs completed, and wickets lost
4. **Get Prediction** · Click predict to see win probabilities and match insights
5. **Analyze Results** · View detailed breakdown with required rate and match pressure

## 🔮 Future Enhancements

- [ ] **Live API Integration** · Real-time data from cricket APIs
- [ ] **Player Impact** · Individual player performance metrics
- [ ] **Weather Conditions** · Include weather impact on match outcomes
- [ ] **Venue Analysis** · Stadium-specific historical performance
- [ ] **Mobile App** · Native mobile application for on-the-go predictions
- [ ] **Advanced Models** · Neural networks and ensemble methods
- [ ] **Historical Comparison** · Compare current match with similar past scenarios

## 📈 Model Performance

**Accuracy** · 85%+ prediction accuracy on test data  
**Precision** · High precision for close match scenarios  
**Recall** · Effective identification of winning probabilities  
**F1-Score** · Balanced performance across different match situations  

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

**CampusX** · for the comprehensive [original tutorial](https://www.youtube.com/watch?v=1YoD0fg3_EM&t=5947s) that served as the foundation for this project 
**IPL Data** · Historical match and delivery data from official sources  
**Streamlit** · Amazing framework for rapid web app development  
**Cricket Analytics Community** · Inspiration from cricket data science projects  


## 📬 Contact

**Shivam Dali** · shivamdali@gmail.com

**GitHub** · [https://github.com/svdexe](https://github.com/svdexe)  
**LinkedIn** · [https://www.linkedin.com/in/shivam-dali-86b0a1201/](https://www.linkedin.com/in/shivam-dali-86b0a1201/)

Project Link: [https://github.com/svdexe/IPL_Win_Probability_Prediction](https://github.com/svdexe/IPL_Win_Probability_Prediction)

---

<div align="center">
  <p>⭐ Star this repo if you found it helpful!</p>
  <p>Made with ❤️ and lots of ☕</p>
</div>
