# 🏛️ Property AI Analyzer
An AI-powered dashboard for real estate analysis, developed during my MSCS at Northeastern and as an AI/ML Intern at Blackbird Investments.

## 🚀 Overview
This application uses Computer Vision to analyze property images and generate reports on valuation strategies, climate health, and neighborhood design.

## 📊 Key Features
- **AI Visual Analysis:** Descriptive property summaries using the BLIP model.
- **Valuation Strategy:** Estimated property values and renovation roadmaps.
- **Climate & Health:** Air quality scoring and greenery (canopy) indexing.
- **Urban Planning:** Walkability and community design metrics.

## 🛠️ Tech Stack
- **Python 3.13**
- **PyTorch** (Optimized for macOS MPS)
- **Hugging Face Transformers** (BLIP-base)
- **Gradio** (Web UI)

## 🏗️ Getting Started
1. `python3 -m venv my_env`
2. `source my_env/bin/activate`
3. `pip install -r requirements.txt` (Note: Create this by running 'pip freeze > requirements.txt')
4. `python3 app.py`