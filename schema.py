import os
import time
import shutil

import pandas as pd
import lancedb
from lancedb.embeddings import with_embeddings
from langchain import PromptTemplate
import predictionguard as pg
import numpy as np
from sentence_transformers import SentenceTransformer

#---------------------#
# Lance DB Setup     #
#---------------------#

#Import datasets

#JOBS
df1=pd.read_csv('datasets/jobs.csv')
df1_table_name = "jobs"

#SOCIAL
df2=pd.read_csv('datasets/social.csv')
df2_table_name = "social"

#movies
df3=pd.read_csv('datasets/movies.csv')
df3_table_name = "movies"

# local path of the vector db
uri = "schema.lancedb"
db = lancedb.connect(uri)

# Embeddings setup
name="all-MiniLM-L12-v2"

# Load model

def load_model():
   return SentenceTransformer(name)

def embed(query, embModel):
    return embModel.encode(query)

#---------------------#
# SQL Schema Creation #
#---------------------#

def create_schema(df,table_name):
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
    create_table_statement = "CREATE TABLE" + " " + table_name + " (\n  " + ",\n  ".join(columns_info) + "\n);"

    # Adjust the statement to handle the final comma, primary keys, or other specifics
    create_table_statement = create_table_statement.replace(",\n);", "\n);")

    return create_table_statement

# SQL Schema for Table Jobs
df1_schema=create_schema(df1,df1_table_name)

# SQL Schema for Table Social
df2_schema=create_schema(df2,df2_table_name)

# SQL Schema for Table Movies
df3_schema=create_schema(df3,df3_table_name)

#---------------------#
#  Prompt Templates          #
#---------------------#

template=""" 
###System:
Generate a brief description of the below data. Be as detailed as possible.

###User:
{schema}

###Assistant:
"""

prompt=PromptTemplate(template=template,input_variables=["schema"])

#---------------------#
#  Generate Description #
#---------------------#

def generate_description(schema):
    prompt_filled=prompt.format(schema=schema)
    result=pg.Completion.create(
        model="Neural-Chat-7B",
        prompt=prompt_filled,
        temperature=0.1,
        max_tokens=300
    )
    return result['choices'][0]['text']

df1_desc=generate_description(df1_schema)
df2_desc=generate_description(df2_schema)
df3_desc=generate_description(df3_schema)

# Create Pandas DataFrame
df = pd.DataFrame({
    'text': [df1_desc, df2_desc, df3_desc],
    'table_name': [df1_table_name, df2_table_name, df3_table_name],
    'schema': [df1_schema, df2_schema, df3_schema],
})

print(df)

def load_data():

    if os.path.exists("schema.lancedb"):
        shutil.rmtree("schema.lancedb")
    os.mkdir("schema.lancedb")
    db = lancedb.connect(uri)
    
    batchModel = SentenceTransformer(name)

    def batch_embed_func(batch):
       return [batchModel.encode(sentence) for sentence in batch]
    
    vecData = with_embeddings(batch_embed_func, df)

    if "schema" not in db.table_names():
        db.create_table("schema", data=vecData)
    else:
        table = db.open_table("schema")
        table.add(data=vecData)
    
    return

load_data()
print("Done")