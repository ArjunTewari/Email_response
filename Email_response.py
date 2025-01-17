import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI

template = """
You are an AI assistant that summarizes emails and drafts concise, professional responses based on the content and tone of the email.
Summarize the email in 2-3 sentences and generate an appropriate response.


Here’s a few-shot prompt that you can use with your LLM for generating summaries and responses for emails. This example provides the model with clear instructions, a few sample inputs, and the expected outputs.

Few-Shot Prompt
Instruction for the LLM: You are an AI assistant that summarizes emails and drafts concise, professional responses based on the content and tone of the email. Summarize the email in 2-3 sentences and generate an appropriate response.

Example 1:

Email:

Subject: Meeting Schedule
Hi,
I hope you're doing well. Could we schedule a meeting to discuss the project updates for this quarter? I’m available this Thursday or Friday, anytime between 10 AM and 3 PM. Let me know what works best for you.

Output:

Summary: The sender wants to schedule a meeting to discuss project updates for this quarter and is available Thursday or Friday from 10 AM to 3 PM.
Response: Hi [Sender's Name],
Thank you for reaching out. I am available this Friday at 11 AM. Please let me know if that works for you. Looking forward to the discussion!
Example 2:

Email:

Subject: Invoice Inquiry
Hello,
I noticed an issue with the invoice you sent last week. It seems to have an incorrect total amount. Could you please review and send me the corrected version? Thank you!

Output:

Summary: The sender is reporting an error in last week's invoice and requests a revised version.
Response: Hi [Sender's Name],
Thank you for pointing this out. I will review the invoice and send you the corrected version shortly. Please let me know if there’s anything else I can assist you with.
Example 3:

Email:

Subject: Event Confirmation
Dear Team,
I’m confirming my attendance at the annual team-building retreat on March 15. Please let me know if any additional information is needed.

Output:

Summary: The sender is confirming their attendance at the team-building retreat on March 15.
Response: Hi [Sender's Name],
Thank you for confirming your attendance at the retreat. If we need any further information, I will let you know. We’re looking forward to seeing you there!

Summarize the email in 2-3 sentences and provide a polite, concise response. 
Adapt the tone of the response to match the email's tone (formal or informal) and include any necessary follow-ups.

Email : {text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

def load_llm(openai_api_key):
    llm = OpenAI(openai_api_key = openai_api_key)
    return llm

st.set_page_config(page_title="Summarize and Generate response")
st.header("Summary and an appropriate response")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        The LLM will take in the content of the email and generate
        a summary of the email content and also generate an appropriate
        response.
        """
    )

with col2:
    st.write("Link to the GitHub Repo : ")

st.markdown("## Enter your OPENAI API Key")

def get_api_key():
    api_key = st.text_input(label="Text", label_visibility="collapsed", placeholder="Enter the API key", key="openai_api_key")
    return api_key
openai_api_key = get_api_key()

st.markdown("## Enter the email content")
def get_content():
    text = st.text_input(label="Text", label_visibility="collapsed", placeholder="Email content", key="text")
    return text
text = get_content()

st.markdown("## Summary and Response: ")

if text:
    if not openai_api_key:
        st.warning("Please enter the api key", icon="⚠️")
        st.stop()
    llm = load_llm(openai_api_key=openai_api_key)

    final_prompt = prompt.format(
        text = text
    )

    data_extracted = llm(final_prompt)
    st.write(data_extracted)