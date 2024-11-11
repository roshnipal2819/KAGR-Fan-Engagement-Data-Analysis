# Overview
This project analyzes fan engagement, demographics, and spending patterns to determine the optimal venue for a collegiate women’s basketball tournament. Using machine learning models and interactive visualizations, the project provides insights into fan engagement trends and venue logistics, with a chatbot feature to make data exploration accessible and user-friendly.

**Application**: [Fan Engagement Data Analysis App](https://kagr-fan-engagement-data-analysis-tv8a3aqztwsfjntmasqw34.streamlit.app/)

## Features
- **Data Collection & Cleaning**: Processes raw data to ensure consistency and accuracy.
- **Machine Learning Models**: Applies classification, clustering, and predictive modeling to understand fan engagement and spending behavior.
- **Interactive Visualizations**: Built with Streamlit and Plotly, providing an accessible web-based interface.
- **LLM Chatbot**: Uses OpenAI API for a Q&A chatbot, allowing users to ask questions about the data and insights.
- **Deployment**: The project is containerized using Docker for smooth deployment.

## Repository Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for data cleaning, EDA, and model training.
- `app.py`: Main Streamlit application script.
- `Dockerfile`: Docker setup to containerize the application.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of dependencies.

## Technologies Used
- **Jupyter Notebook**: Data analysis and model building.
- **Streamlit**: Web application framework for interactive visualizations.
- **OpenAI API**: Integrates a chatbot for data Q&A.
- **Docker**: Containers for consistent environment setup and deployment.

## Install Dependencies
```bash
pip install -r requirements.txt

Run with Docker
docker build -t fan-data-analysis .
docker run -p 8501:8501 fan-data-analysis

Command to Run the App- streamlit run streamlit_app/app.py

Usage
Data Analysis: Explore the notebooks in notebooks/ for detailed analysis steps.
Streamlit Application: Launch the Streamlit app to visualize data and interact with the chatbot.
LLM Chatbot: Use the chatbot to query insights on fan engagement, demographics, and venue recommendations.
