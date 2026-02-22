import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

def auto_chart(df, question):
    try:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        if not numeric_cols or not cat_cols:
            return None

        question_lower = question.lower()

        # Remove non-useful columns
        exclude = ['row id', 'postal code', 'row_id']
        cat_cols = [c for c in cat_cols if c.lower() not in exclude]
        
        # Pick best columns based on question
        if any(word in question_lower for word in ['category', 'categories']):
            x_col = 'Category' if 'Category' in df.columns else cat_cols[0]
            y_col = 'Sales' if 'Sales' in df.columns else numeric_cols[0]
        elif any(word in question_lower for word in ['region']):
            x_col = 'Region' if 'Region' in df.columns else cat_cols[0]
            y_col = 'Profit' if 'Profit' in df.columns else numeric_cols[0]
        elif any(word in question_lower for word in ['segment']):
            x_col = 'Segment' if 'Segment' in df.columns else cat_cols[0]
            y_col = 'Sales' if 'Sales' in df.columns else numeric_cols[0]
        elif any(word in question_lower for word in ['state']):
            x_col = 'State' if 'State' in df.columns else cat_cols[0]
            y_col = 'Sales' if 'Sales' in df.columns else numeric_cols[0]
        else:
            x_col = cat_cols[0]
            y_col = numeric_cols[0]

        grouped = df.groupby(x_col)[y_col].sum().reset_index()
        grouped = grouped.sort_values(y_col, ascending=False).head(10)

        fig = px.bar(grouped, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        return fig

    except Exception as e:
        print(f"Chart error: {e}")
        return None