import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from agent import run_agent, clean_data
from charts import auto_chart

st.set_page_config(page_title="Data Analyst AI Agent", layout="wide")
st.title("🤖 Data Analyst AI Agent")
st.markdown("Upload any CSV and ask business questions in plain English")

def format_currency(value):
    try:
        return f"${value:,.0f}"
    except:
        return value

with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.header("💡 Sample Questions")
    st.markdown("""
    - What are top 3 categories by revenue?
    - Which region has lowest profit?
    - What is the average discount by segment?
    - How many orders were placed each month?
    - Which state has the highest sales?
    """)
    if st.button("🗑 Clear Conversation"):
        st.session_state.clear()
        st.rerun()

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    df = clean_data(df)

    col1, col2, col3 = st.columns(3)
    total_sales = df["Sales"].sum() if "Sales" in df.columns else 0
    total_profit = df["Profit"].sum() if "Profit" in df.columns else 0
    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    col1.metric("💰 Total Sales", format_currency(total_sales))
    col2.metric("📈 Total Profit", format_currency(total_profit))
    col3.metric("🛒 Total Orders", f"{total_orders:,}")

    st.divider()
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    with st.expander("📊 Data Summary"):
        st.write(df.describe())

    st.divider()
    st.subheader("🔍 Ask the AI Agent")
    question = st.text_input("Type your question here...")

    if st.button("🚀 Analyze", type="primary"):
        if question:
            with st.spinner("🔍 AI is analyzing your data..."):
                try:
                    answer = run_agent(df, question)
                    st.session_state.last_answer = answer
                    st.session_state.last_fig = auto_chart(df, question)
                    st.success("✅ Analysis Complete")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question")

    if "last_fig" in st.session_state and isinstance(st.session_state.last_fig, go.Figure):
        st.divider()
        st.subheader("📈 Generated Visualization")
        st.plotly_chart(st.session_state.last_fig, use_container_width=True)
        try:
            img_bytes = st.session_state.last_fig.to_image(format="png")
            st.download_button(
                label="📥 Download Chart",
                data=img_bytes,
                file_name="chart.png",
                mime="image/png"
            )
        except:
            pass

    if "last_answer" in st.session_state:
        st.divider()
        st.markdown("### 💡 AI Business Insight")
        try:
            explanation = run_agent(df.head(20), f"Give a short executive business insight based on: {st.session_state.last_answer}")
            st.success("Insight Generated")
            st.write(explanation)
        except:
            st.info("Insight generation unavailable.")

else:
    st.info("👆 Upload a CSV file to get started")