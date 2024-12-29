import streamlit as st
import openai
import gc
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import google.generativeai as genai  # Import gemini for generative AI model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ast
import matplotlib.font_manager as fm
import statsmodels.api as sm


@st.cache_data
def load_data(year):
    return pd.read_excel(f'/xlsx-new/{year}.xlsx')

def load_filtered_data(year, synonyms):
    df = pd.read_excel(f'/{year}.xlsx')
    # Filter rows containing any synonym in the 'title' or 'doc_content'

    filtered_df = df[
        df['title'].str.contains('|'.join(synonyms), case=False, na=False) |
        df['doc_content'].str.contains('|'.join(synonyms), case=False, na=False)
    ]
    return filtered_df if not filtered_df.empty else None

# Load environment variables
load_dotenv()

# Get OpenAI API key and Assistant ID from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
#assistant_playground_id = os.getenv("OPENAI_SYNONYMS_ASSISTANT")
client = OpenAI(api_key=openai.api_key)
#assistant = client.beta.assistants.retrieve(assistant_playground_id)
assistant = client.beta.assistants.create(
name="Chinese Synonyms Assistant",
instructions="""You are a helpful assistant that finds Simplified translations and synonyms of a given word, name or abbreviation in Traditional Chinese or another language. 
Please consider all possible synonyms that consist of at least two characters, including terms used in the 1940s and 50s and those used during the Cultural Revolution. 
The output should be a Python list containing all possible translations/synonyms. 

Examples: 

User: London
Answer: ['伦敦']

User: Bundeswehr
Answer: ['德国联邦国防军', '德国联邦国防军', '德国军队', '德国武装力量',  '德军']

User: 蔣介石
Answer: ['蒋介石', '蒋周泰' '蒋瑞元', '蒋志清', '蒋中正', '中正', '蒋公']

The output should be a JSON in the following format: 
{"synonyms": List // List of the translations/synonyms as specified above} """,
tools=[],
model="gpt-4o-mini",
)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def wait_on_run(run, thread, client):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run




def run_single_gemini_fulldf(df, entity):
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="你是一个设计用于输出情感分类标签的助手。专门分析中文文本中的情感。你将针对提供的中文文本，判断其情感态度（sentiment analysis）。回复应是分配的标签之一，分别是 `['完全正面', '稍微正面', '中立', '稍微负面', '完全负面']`。当你无法识别对相关个别词汇（entity）的任何情感时，你应该分配“中立”标签。 ‘稍微正面’ 和 ‘稍微负面’的标签，则用于当个别词汇的情感轻微、模糊或不确定时。在其他情况下，如果对个别词汇的情感很明确，就使用‘完全正面’或‘完全负面’的标签。你不应依赖常识对词汇进行判断，而应依据给定文本中传达的情感进行分析。如果文本中同时存在正面和负面情感，你必须确定哪个对相关实体传达的情感最主要或是最强烈。 \n输出应为按照以下模式格式化的JSON：\n{\\n\\t\"label\": string // 分配标签给相关个别词汇：['完全正面', '稍微正面', '中立', '稍微负面', '完全负面']。如果在文本中找不到相关词汇，请写‘none’。 \\n}",
    )

    results = defaultdict(dict)

    for index, row in df.iterrows():
        text = str(row['title']) + "\n" + str(row['doc_content'])
        filename = row['filename']
        date = row['timeseq_not_before']
        date = datetime.strptime(str(date), '%Y%m%d').date()

        results[filename]['date'] = date

        chat_session = model.start_chat()

        prompt = f"我们将分析以下文本： \"{text}\" \n你的任务是为文本中与 \"{entity}\" 相关的情感指定一个标签。"

        response = chat_session.send_message(prompt)

        results[filename]['response'] = response.text
        results[filename]['messages'] = response
        # Add further processing if needed (e.g., extract labels)

    return results

# Make sure matplotlib uses a font that supports Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # To avoid issues with minus signs


def plot(results, entity_name, df_for_plot):
    # Process each entry
    for filename, data in results.items():
        response_dict = ast.literal_eval(data['response'])  # Parse the response string as a dictionary
        label = response_dict['label']
        data['label'] = label

    df_for_plot['sentiment_label'] = df_for_plot['filename'].apply(lambda x: results[x]['label'])

    # Map Chinese sentiment labels to numerical values
    sentiment_mapping = {
        '完全正面': 2,
        '稍微正面': 1,
        '中立': 0,
        '稍微负面': -1,
        '完全负面': -2
    }

    df_for_plot['sentiment_value'] = df_for_plot['sentiment_label'].map(sentiment_mapping)

    # Drop rows with missing sentiment values
    df_for_plot = df_for_plot.dropna(subset=['sentiment_value'])

    # Ensure the 'date' column is in datetime format and extract year for the x-axis
    df_for_plot['date'] = pd.to_datetime(df_for_plot['timeseq_not_before'], format='%Y%m%d')
    df_for_plot['year'] = df_for_plot['date'].dt.year

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df_for_plot['date'], df_for_plot['sentiment_value'], marker='o')

    # Format the x-axis to show years clearly
    plt.xlabel('Publication Year')
    plt.ylabel('Sentiment')
    plt.title(f'Sentiment toward {entity_name} over time')
    plt.grid(True)

    # Custom tick labels in Chinese
    plt.yticks(range(-2, 3), ['Negative-Standard', 'Negative-Slight', 'Neutral', 'Positive-Slight', 'Positive-Standard'])

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(plt)
    plt.close()




def plot_loess(results, entity_name, df_for_plot, frac=0.3):
    # Process each entry
    for filename, data in results.items():
        response_dict = ast.literal_eval(data['response'])  # Parse the response string as a dictionary
        label = response_dict['label']
        data['label'] = label

    df_for_plot['sentiment_label'] = df_for_plot['filename'].apply(lambda x: results[x]['label'])

    # Map Chinese sentiment labels to numerical values
    sentiment_mapping = {
        '完全正面': 2,
        '稍微正面': 1,
        '中立': 0,
        '稍微负面': -1,
        '完全负面': -2
    }

    df_for_plot['sentiment_value'] = df_for_plot['sentiment_label'].map(sentiment_mapping)

    # Drop rows with missing sentiment values
    df_for_plot = df_for_plot.dropna(subset=['sentiment_value'])

    # Ensure the 'date' column is in datetime format and extract year for the x-axis
    df_for_plot['date'] = pd.to_datetime(df_for_plot['timeseq_not_before'], format='%Y%m%d')
    df_for_plot['year'] = df_for_plot['date'].dt.year

    # Sort the DataFrame by date for proper line plotting
    df_for_plot = df_for_plot.sort_values(by='date')

    # Process each entry
    for filename, data in results.items():
        response_dict = ast.literal_eval(data['response'])  # Parse the response string as a dictionary
        label = response_dict['label']
        data['label'] = label

    df_for_plot['sentiment_label'] = df_for_plot['filename'].apply(lambda x: results[x]['label'])

    # Map Chinese sentiment labels to numerical values
    sentiment_mapping = {
        '完全正面': 2,
        '稍微正面': 1,
        '中立': 0,
        '稍微负面': -1,
        '完全负面': -2
    }

    df_for_plot['sentiment_value'] = df_for_plot['sentiment_label'].map(sentiment_mapping)

    # Ensure the 'date' column is in datetime format
    df_for_plot['date'] = pd.to_datetime(df_for_plot['timeseq_not_before'], format='%Y%m%d')
    df_for_plot['year'] = df_for_plot['date'].dt.year

    # Sort the DataFrame by date for proper line plotting
    df_for_plot = df_for_plot.sort_values(by='date')

    # Convert dates to numerical values (required for LOESS)
    df_for_plot['date_num'] = pd.to_numeric(df_for_plot['date'])

    # Fit the LOESS model
    lowess = sm.nonparametric.lowess(df_for_plot['sentiment_value'], df_for_plot['date_num'], frac=frac)

    # Scatter plot for original sentiment values
    plt.figure(figsize=(10, 6))
    plt.scatter(df_for_plot['date'], df_for_plot['sentiment_value'], marker='o', color='r', label='Original Sentiment')

    # Plot the LOESS-smoothed line
    plt.plot(pd.to_datetime(lowess[:, 0]), lowess[:, 1], color='b', label=f'LOESS Smoothing (frac={frac})')

    # Format the x-axis to show years clearly
    plt.xlabel('Publication Year')
    plt.ylabel('Sentiment')
    plt.title(f'Sentiment toward {entity_name} over time (LOESS Smoothing)')
    plt.grid(True)

    # Custom tick labels in Chinese
    plt.yticks(range(-2, 3), ['Negative-Standard', 'Negative-Slight', 'Neutral', 'Positive-Slight', 'Positive-Standard'])

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.legend()
    st.pyplot(plt)
    plt.close()







#### STREAMLIT APP STARTS HERE
entity = st.text_input("Entity:")

if st.button("Run"):
    # Call the OpenAI assistant API using the beta client
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=entity
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    run = wait_on_run(run, thread, client)

    messages = client.beta.threads.messages.list(thread_id=thread.id)

    response = json.loads(messages.model_dump_json())

    print(response)

    # Extract the synonyms list
    synonyms_str = response['data'][0]['content'][0]['text']['value']

    # First, parse the string as a dictionary using json.loads
    synonyms_dict = json.loads(synonyms_str)

    # Extract the "synonyms" list from the dictionary
    synonyms = synonyms_dict['synonyms']

    # Now you can append the entity to the list
    synonyms.append(entity)

    if "synonyms" not in st.session_state:
        st.session_state.synonyms = synonyms
    else:
        st.session_state.synonyms = synonyms

    st.session_state.run_clicked = True

if st.session_state.get("run_clicked"):
    for synonym in st.session_state.synonyms:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(synonym)
        with col2:
            if st.button(f"Delete {synonym}", key=synonym):
                st.session_state.synonyms.remove(synonym)
                st.experimental_rerun()

    new_synonym = st.text_input("Add new synonym:")

    if st.button("Add Synonym"):
        if new_synonym:
            if "synonyms" not in st.session_state:
                st.session_state.synonyms = []
            st.session_state.synonyms.append(new_synonym)
            st.experimental_rerun()

    if st.button("Next"):
        st.session_state.synonym_list_saved = True

if st.session_state.get("synonym_list_saved"):
    st.write("Synonym list saved!")

    years = list(range(1946, 2025))
    from_year = st.selectbox('From Year', years, index=0)
    to_year = st.selectbox('To Year', years, index=len(years) - 1)

    if st.button("Search for relevant articles"):
        filtered_dfs = []
        for year in range(from_year, to_year + 1):
            filtered_df = load_filtered_data(year, st.session_state.synonyms)
            if filtered_df is not None:
                filtered_dfs.append(filtered_df)

        if filtered_dfs:
            result_df = pd.concat(filtered_dfs)
            result_message = f'Found {len(result_df)} entries containing the synonyms: {", ".join(st.session_state.synonyms)}'
            st.session_state.result_message = result_message
            st.session_state.result_df = result_df
        else:
            st.session_state.result_message = f'No entries found for the synonyms in the selected years.'
            st.session_state.result_df = None

        st.session_state.search_done = True

if st.session_state.get("search_done"):
    if 'result_message' in st.session_state:
        st.write(st.session_state.result_message)

    if 'result_df' in st.session_state and st.session_state.result_df is not None:
        result_df = st.session_state.result_df

        sample_size = st.number_input('Enter sample size (up to 200)', min_value=1, max_value=200, value=20)

        if st.button("Sample Articles"):
            if len(result_df) > sample_size:
                sampled_df = result_df.sample(sample_size)
            else:
                sampled_df = result_df.copy()

            # Store the sampled DataFrame and remove the large result_df from memory
            st.session_state.sampled_df = sampled_df

            # Remove the large result_df to free up memory
            del result_df


            gc.collect()

        if 'sampled_df' in st.session_state:
            st.write(f'Sampled {len(st.session_state.sampled_df)} entries for analysis')
            st.dataframe(st.session_state.sampled_df)

            selected_synonym = st.selectbox('Choose a synonym for sentiment analysis', st.session_state.synonyms)
            st.session_state.keyword = selected_synonym
            st.write(f'Keyword selected: {st.session_state.keyword}')

            # Inside the "Run Sentiment Analysis" section:
            if st.button("Run Sentiment Analysis"):
                with st.spinner('Running sentiment analysis...'):
                    # Run the sentiment analysis function
                    result = run_single_gemini_fulldf(st.session_state.sampled_df, st.session_state.keyword)

                    # Save results in session state
                    st.session_state.results = result

                    # Add the sentiment score to the sampled DataFrame
                    score_column = f"{st.session_state.keyword}_gemini_score"
                    filename_to_label = {filename: data['response'] for filename, data in result.items()}
                    st.session_state.sampled_df[score_column] = st.session_state.sampled_df['filename'].map(
                        filename_to_label)

                    # Display DataFrame with sentiment scores
                    st.write("DataFrame with Sentiment Score")
                    st.dataframe(st.session_state.sampled_df)

                    # Non-forecasting plots
                    plot(result, entity, st.session_state.sampled_df)
                    plot_loess(result, entity, st.session_state.sampled_df)


                if st.button("Clear Cache"):
                    st.cache_data.clear()
                    st.success("Cache cleared!")

