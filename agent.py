import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

os.environ["OPENAI_API_KEY"] =  "YOUR_KEY_HERE"

def clean_data(df):
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    return df

def run_agent(df, question):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True
    )
    result = agent.invoke(question)
    return result["output"]