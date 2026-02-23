from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


def clean_data(df):
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    return df


def generate_report(df):
    report = []
    report.append(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    report.append(f"**Columns:** {', '.join(df.columns.tolist())}")
    missing = df.isnull().sum()[df.isnull().sum() > 0]
    report.append(f"**Missing Values:**\n{missing.to_string() if not missing.empty else 'None'}")
    report.append(f"**Numeric Summary:**\n{df.describe().to_string()}")
    return "\n\n".join(report)


def run_agent(df, question, chat_history=None):
    if chat_history is None:
        chat_history = []

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=20,
        max_execution_time=120,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

    if chat_history:
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history[-3:]])
        full_question = (
            f"Previous conversation:\n{history_text}\n\n"
            f"New question: {question}\n"
            f"Always compute the exact answer using Python code. Include actual numbers in your response."
        )
    else:
        full_question = (
            f"{question}\n"
            f"Always compute the exact answer using Python code. Include actual numbers in your response."
        )

    result = agent.invoke(full_question)
    return result["output"]