# Streamlit Chat with Data

Streamlit Chat with Data is an interactive chat application built with Streamlit, enabling users to engage in conversations powered by data from various datasets. This application demonstrates how to implement chat functionalities with a focus on querying single and multiple datasets, including jobs, social media, and movies, for insightful data-driven interactions.

## Features

- **Data-Driven Chat Interface**: Engage in conversations that fetch and display data from datasets in real-time.
- **Support for Multiple Datasets**: Seamlessly switch between datasets on jobs, social media, and movies for diverse inquiries.
- **Dynamic SQL Query Generation**: Automatically generates SQL queries based on user questions to retrieve relevant data.
- **Secure and Private**: Implements checks for Personally Identifiable Information (PII) and SQL injection vulnerabilities to ensure secure interactions.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or later
- Streamlit
- Pandas
- DuckDB
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/predictionguard/datachat-streamlit.git
cd streamlit-chat-with-data
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Running the Application
To run the application, navigate to the project directory and execute:
```bash
python -m streamlit run Home.py
```
