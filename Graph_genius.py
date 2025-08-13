import pandas as pd 
import streamlit as s
from langgraph.graph import StateGraph,END
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt

#node 1 : file upload
def upload_file(state):
    uploaded_file = s.file_uploader("UPLOAD CSV FILE",type=["csv"])
    if uploaded_file:
      df=pd.read_csv(uploaded_file)
      state["df"]=df 
      s.session_state["df"]=df
    return state


#node 2: dashboard / chart generation 


def dashboard(state):
    if "df" in state:
        df = state["df"]

        # Data preview
        s.write("### LET'S HAVE A DATA PREVIEW!")
        s.dataframe(df.head())

        # Histogram / Bar chart for numeric columns
        num_columns = df.select_dtypes(include='number').columns
        if len(num_columns) > 0:
            s.write("Bar Chart (Numeric Columns)")
            s.bar_chart(df[num_columns])

        # Scatter Plot
        s.write("Scatter Plot")
        if len(num_columns) >= 2:
            x_axis = s.selectbox("Select X-axis", num_columns, key="scatter_x")
            y_axis = s.selectbox("Select Y-axis", num_columns, key="scatter_y")

            fig, ax = plt.subplots()
            ax.scatter(df[x_axis], df[y_axis])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
            s.pyplot(fig)

        # Line Chart
        s.write("### Line Chart")
        if len(num_columns) >= 2:
            x_axis_line = s.selectbox("Select X-axis for Line Chart", num_columns, key="line_x")
            y_axis_line = s.selectbox("Select Y-axis for Line Chart", num_columns, key="line_y")

            fig, ax = plt.subplots()
            ax.plot(df[x_axis_line], df[y_axis_line])
            ax.set_xlabel(x_axis_line)
            ax.set_ylabel(y_axis_line)
            ax.set_title(f"Line Chart: {x_axis_line} vs {y_axis_line}")
            s.pyplot(fig)

        # Pie Chart
        s.write("### Pie Chart")
        cat_columns = df.select_dtypes(exclude='number').columns
        if len(cat_columns) > 0:
            pie_column = s.selectbox("Select column for Pie Chart", cat_columns)
            pie_data = df[pie_column].value_counts()

            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
            ax.set_title(f"Pie Chart of {pie_column}")
            s.pyplot(fig)

    return state


#node 3: ai_query chatbot 
def ai_query(state):
  if "df" in state:
    df=state["df"]
    qna=s.text_input("FEEL FREE TO ASK DATA RELATED QUERIES HERE!")
    if qna:
      llm=ChatOpenAI(model="gpt4",temperature=0)
      agent=create_pandas_dataframe_agent(ChatOpenAI(temperature=0),df,verbose=True)
      ans=agent.run(qna)
      s.write("answer: ",ans)
  return state

#node 4: build langGraph
builder =StateGraph(dict)

builder.add_node("UPLOAD",upload_file)
builder.add_node("DASHBOARD",dashboard)
builder.add_node("QUERY_BOT",ai_query)

builder.add_edge("UPLOAD","DASHBOARD")
builder.add_edge("DASHBOARD","QUERY_BOT")
builder.add_edge("QUERY_BOT",END)

builder.set_entry_point("UPLOAD")

#run streamlit app

s.title("GraphGenius_AI: AI that visualizes data brilliantly!")
app=builder.compile()
app.invoke({})

