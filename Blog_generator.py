import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI

template = """
"You are a professional blog writer skilled in creating concise, engaging, and informative articles tailored to a specific audience. Your task is to write a 200-word blog article based on the inputs provided by the user. The article should align with the topic and tone specified and must follow these guidelines:

Title: Create a compelling and relevant blog title in under 10 words.
Structure: Start with a catchy opening sentence to draw the reader's attention, followed by 2–3 well-organized paragraphs. Each paragraph should maintain a logical flow and add value to the reader's understanding.
Tone: Match the tone specified by the user, such as professional, playful, conversational, or inspirational.
Conclusion: End with a strong closing sentence that either summarizes the article or provides a call-to-action to engage the audience.
Input Example:
Topic: "Benefits of AI in Small Businesses"
Tone: Professional
Output Example:
Title: "Transforming Small Businesses with AI Innovation"
Body: "Artificial Intelligence (AI) is revolutionizing how small businesses operate, offering tools to streamline operations and boost growth. From automated customer support to data-driven insights, AI empowers small enterprises to compete with larger players effectively.

One significant benefit is cost efficiency. AI-powered tools like chatbots reduce overheads by automating routine tasks. Meanwhile, predictive analytics enable smarter decision-making, helping businesses anticipate market trends and customer needs.

Small businesses can also improve customer experiences through personalized marketing strategies powered by AI algorithms. These tools ensure tailored messages reach the right audience at the right time.

Embracing AI isn't just an option—it's a necessity for staying competitive. Start small, experiment, and unlock your business's potential today."

Now, write a blog using this structure and style. Always adhere to the 200-word limit while ensuring the article is engaging and informative.
Topic : {topic}
Tone : {tone}
"""
prompt = PromptTemplate(
    input_variables=["topic", "tone"],
    template=template
)

def load_llm(openai_api_key):
    llm = OpenAI(openai_api_key = openai_api_key)
    return llm

st.set_page_config(page_title="Blog writer")
st.header("Creates articles based on the topic and tone provided by the user")

st.markdown(
    """
    The LLM will take in the topic and tone from the user
    and generate a 200 word article for a blog this can be useful 
    for writers as it automates the blog writing.
    """
)

st.write("Link to the GitHub Repo : https://github.com/ArjunTewari/Email_response/blob/main/Blog_generator.py")

st.markdown("## Enter your OPENAI API Key")

def get_api_key():
    api_key = st.text_input(label="Text", label_visibility="collapsed", placeholder="Enter the API key", key="openai_api_key")
    return api_key
openai_api_key = get_api_key()

col1, col2 = st.columns(2)

with col1:
    st.markdown("## Enter the topic")
    def get_topic():
        topic = st.text_input(label="Topic of the article", label_visibility="collapsed", placeholder="Article Topic", key="topic")
        return topic
with col2:
    st.markdown("## Enter the topic")
    def get_tone():
        tone = st.text_input(label="Tone of the article", label_visibility="collapsed", placeholder="Article Tone", key="tone")
        return tone


topic = get_topic()
tone = get_tone()

st.markdown("## Article :")

if topic and tone:
    if not openai_api_key:
        st.warning("Please enter the api key", icon="⚠️")
        st.stop()
    llm = load_llm(openai_api_key=openai_api_key)

    final_prompt = prompt.format(
        topic = topic,
        tone = tone
    )

    data_extracted = llm.stream(final_prompt)
    st.write_stream(data_extracted)
