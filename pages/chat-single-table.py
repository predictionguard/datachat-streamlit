import time
import os

import pandas as pd
import streamlit as st
import lancedb
from lancedb.embeddings import with_embeddings
from langchain import PromptTemplate
import predictionguard as pg
import streamlit as st
import duckdb
import re
import numpy as np

#---------------------#
# Streamlit config    #
#---------------------#

if "login" not in st.session_state:
    st.session_state["login"] = False

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#--------------------------#
# Get Shopify Data        #
#--------------------------#

@st.cache_resource
def load_data():

    df=pd.read_csv('datasets/social.csv')
    return df

#--------------------------#
# Create Schema           #
#--------------------------#

@st.cache_resource
def create_schema(df):
    # Here we will create an example SQL schema based on the data in this dataset.
    # In a real use case, you likely already have this sort of CREATE TABLE statement.
    # Performance can be improved by manually curating the descriptions.

    columns_info = []

    # Iterate through each column in the DataFrame
    for col in df.columns:
        # Determine the SQL data type based on the first non-null value in the column
        first_non_null = df[col].dropna().iloc[0]
        if isinstance(first_non_null, np.int64):
            kind = "INTEGER"
        elif isinstance(first_non_null, np.float64):
            kind = "DECIMAL(10,2)"
        elif isinstance(first_non_null, str):
            kind = "VARCHAR(255)"  # Assuming a default max length of 255
        else:
            kind = "VARCHAR(255)"  # Default to VARCHAR for other types or customize as needed

        # Sample a few example values
        example_values = ', '.join([str(x) for x in df[col].dropna().unique()[0:4]])

        # Append column info to the list
        columns_info.append(f"{col} {kind}, -- Example values are {example_values}")

    # Construct the CREATE TABLE statement
    create_table_statement = "CREATE TABLE df (\n  " + ",\n  ".join(columns_info) + "\n);"

    # Adjust the statement to handle the final comma, primary keys, or other specifics
    create_table_statement = create_table_statement.replace(",\n);", "\n);")

    return create_table_statement

df=load_data()
schema=create_schema(df)
st.sidebar.header("Schema for the Social Media Data")
st.sidebar.code(schema)
conn = duckdb.connect(database=':memory:')
conn.register('df', df)

#--------------------------#
#  Prompt Templates          #
#--------------------------#

### PROMPT TEMPLATES
qa_template = """### System:
You are a data chatbot who answers the user question. To answer these questions we need to run SQL queries on our data and its output is given below in context. You just have to frame your answer using that context. Give a short and crisp response.Don't add any notes or any extra information after your response.

### User:
Question: {question}

context: {context}

### Assistant:
"""

qa_prompt = PromptTemplate(template=qa_template,input_variables=["question", "context"])

sql_template = """<|begin_of_sentence|>You are a SQL expert and you only generate SQL queries which are executable. You provide no extra explanations.
You respond with a SQL query that answers the user question in the below instruction by querying a database with the schema provided in the below instruction.
Always start your query with SELECT statement and end with a semicolon.

### Instruction:
User question: \"{question}\"

Database schema:
{schema}

### Response:
"""
sql_prompt=PromptTemplate(template=sql_template, input_variables=["question","schema"])

#--------------------------#
# Generate SQL Query      #
#--------------------------#
def generate_sql_query(question):

  prompt_filled = sql_prompt.format(question=question,schema=schema)

  try:
      result = pg.Completion.create(
          model="deepseek-coder-6.7b-instruct",
          prompt=prompt_filled,
          max_tokens=300,
          temperature=0.1
      )
      sql_query = result["choices"][0]["text"]
      return sql_query

  except Exception as e:
      return None


def extract_and_refine_sql_query(sql_query):

  # Extract SQL query using a regular expression
  match = re.search(r"(SELECT.*?);", sql_query, re.DOTALL)
  if match:

      refined_query = match.group(1)

      # Check for and remove any text after a colon
      colon_index = refined_query.find(':')
      if colon_index != -1:
          refined_query = refined_query[:colon_index]

      # Ensure the query ends with a semicolon
      if not refined_query.endswith(';'):
          refined_query += ';'
      return refined_query

  else:
      return ""


def get_answer_from_sql(question):

  sql_query = generate_sql_query(question)
  sql_query = extract_and_refine_sql_query(sql_query)

  try:

      # print("Executing SQL Query:", sql_query)
      result = conn.execute(sql_query).fetchall()

      # print("Result:", result)
      return result, sql_query

  except Exception as e:

      print(f"Error executing SQL query: {e}")
      return "There was an error executing the SQL query."

#--------------------------#
# Get Answer               #
#--------------------------#
    
def get_answer(question,context):
  try:

      prompt_filled = qa_prompt.format(question=question, context=context)

      # Respond to the user
      output = pg.Completion.create(
          model="Neural-Chat-7B",
          prompt=prompt_filled,
          max_tokens=200,
          temperature=0.1
      )
      completion = output['choices'][0]['text']

      return completion

  except Exception as e:
      completion = "There was an error executing the SQL query."
      return completion
        
#--------------------------#
# Streamlit app            #
#--------------------------#
        
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # contruct prompt thread
        examples = []
        turn = "user"
        example = {}
        for m in st.session_state.messages:
            latest_message = m["content"]
            example[turn] = m["content"]
            if turn == "user":
                turn = "assistant"
            else:
                turn = "user"
                examples.append(example)
                example = {}
        if len(example) > 2:
            examples = examples[-2:]
        else:
            thread = ""

        # # Check for PII
        # with st.spinner("Checking for PII..."):
        #     pii_result = pg.PII.check(
        #         prompt=latest_message,
        #         replace=False,
        #         replace_method="fake"
        #     )

        # # Check for injection
        # with st.spinner("Checking for security vulnerabilities..."):
        #     injection_result = pg.Injection.check(
        #         prompt=latest_message,
        #         detect=True
        #     )

        # Check the cache

        # # Handle insecure states
        # elif "[" in pii_result['checks'][0]['pii_types_and_positions']:
        #     st.warning('Warning! PII detected. Please avoid using personal information.')
        #     full_response = "Warning! PII detected. Please avoid using personal information."
        # elif injection_result['checks'][0]['probability'] > 0.5:
        #     st.warning('Warning! Injection detected. Your input might result in a security breach.')
        #     full_response = "Warning! Injection detected. Your input might result in a security breach."

        # generate response
        with st.spinner("Generating an answer..."):
            context=get_answer_from_sql(latest_message)
            print("context",context)
            completion = get_answer(latest_message,context)

        # display response
            for token in completion.split(" "):
                full_response += " " + token
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.075)
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})