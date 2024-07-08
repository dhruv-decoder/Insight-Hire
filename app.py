import streamlit as st
import requests
import os
from pytube import YouTube
import tempfile
from fpdf import FPDF
import time
from together import Together

# API configurations
together_api_key = os.environ.get('TOGETHER_API_KEY')
assembly_api_key = os.environ.get('ASSEMBLYAI_API_KEY')

client = Together(api_key=together_api_key)

assembly_base_url = "https://api.assemblyai.com/v2"
assembly_headers = {
    "authorization": assembly_api_key
}

def generate_sentiment_score(input_text, parameters):
    prompt = f'''
    You are an experienced interview reviewer and consultant for a reputable company. Your role is to evaluate the sentiment displayed by job candidates during their interviews based on the transcripts of their responses.
    The hiring team has provided you with an interview transcript and has asked you to analyze the candidate's sentiment for the following parameters: {', '.join(parameters)}. Your assessment will help the team make more informed hiring decisions and identify candidates who demonstrate genuine positive sentiment towards the role and the company.
    The parameters to evaluate are:
    {', '.join(parameters)}.
    To complete this task, you will:
    1. Carefully review the provided interview transcript.
    2. Consider phrases, word choices, or patterns of speech that convey positive or negative sentiment for each parameter.
    3. Based on your analysis, provide a sentiment score on a scale of 1-5 for each parameter, with 1 being extremely negative and 5 being extremely positive.
    Provide your scores in the format: Parameter: (Score).
    4. The output should be like a professional interview reviewer(HR)
    Here's the transcript:
    {input_text}
    '''

    response = client.chat.completions.create(
        model="microsoft/WizardLM-2-8x22B",
        messages=[
            {"role": "system", "content": "You are an AI assistant that analyzes interview transcripts and provides sentiment scores."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=0.7
    )

    return response.choices[0].message.content

def generate_detailed_feedback(input_text, parameters):
    prompt = f'''
    As an experienced interview reviewer, provide a detailed analysis of the candidate's responses based on the following parameters: {', '.join(parameters)}. 
    
    Include specific examples, quotes, and adjectives from the transcript that support your analysis.(no need to provide scores)Offer actionable insights and recommendations for the hiring team to make informed decisions. Summarize the candidate's overall sentiment and demeanor.
    Here's the transcript:
    {input_text}
    '''

    response = client.chat.completions.create(
        model="microsoft/WizardLM-2-8x22B",
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides detailed feedback on interview transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2048,
        top_p=0.7
    )

    return response.choices[0].message.content

def upload_to_assemblyai(file_path):
    with open(file_path, "rb") as f:
        response = requests.post(f"{assembly_base_url}/upload", headers=assembly_headers, data=f)
    return response.json().get("upload_url")

def transcribe_with_assemblyai(upload_url):
    data = {
        "audio_url": upload_url
    }
    response = requests.post(f"{assembly_base_url}/transcript", json=data, headers=assembly_headers)
    transcript_id = response.json().get("id")

    while True:
        response = requests.get(f"{assembly_base_url}/transcript/{transcript_id}", headers=assembly_headers)
        result = response.json()
        if result['status'] == 'completed':
            return result['text']
        elif result['status'] == 'failed':
            raise Exception("Transcription failed")
        time.sleep(5)

def transcript(video_link):
    try:
        yt = YouTube(video_link)
        stream = yt.streams.filter(only_audio=True).first()
        temp_file_path = tempfile.mktemp(suffix=".mp4")
        stream.download(output_path=os.path.dirname(temp_file_path), filename=os.path.basename(temp_file_path))

        print(f"Video '{yt.title}' downloaded successfully!")

        upload_url = upload_to_assemblyai(temp_file_path)
        transcription_text = transcribe_with_assemblyai(upload_url)

        os.remove(temp_file_path)

        return transcription_text

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    st.set_page_config(page_title="Insight Hire", page_icon=":bar_chart:")
    st.title("Insight Hire")
    st.write("Analyze interview transcripts or videos to gain valuable insights into candidate sentiment.")

    st.sidebar.markdown("## About")
    st.sidebar.markdown("""
    <div style='color: #1f77b4; font-weight: bold;'>Streamline Your Interview Evaluation</div>
    - Get data-driven sentiment scores for key parameters
    - Identify top candidates based on sentiment analysis
    - Make informed hiring decisions with actionable insights
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)  # Horizontal separator

    st.sidebar.markdown("## Tips")
    st.sidebar.markdown("""
    <div style='color: #2ca02c; font-weight: bold;'>📝 Input Preparation</div>
    - Provide clear interview transcripts or valid video links
    - Specify relevant parameters for sentiment analysis
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)  # Horizontal separator

    st.sidebar.markdown("## About Me")
    st.sidebar.markdown("""
    <div style='color: #d62728; font-weight: bold;'>👋 Hi, I'm Dhruv!</div>
    I want to make a real impact in the field of AI/ML . My main interest lies in model building and deployment. I'm passionate about leveraging cutting-edge technologies to solve real-world problems.
    """, unsafe_allow_html=True)

    input_option = st.radio("Select input type", ("Text", "YouTube Video Link"))

    input_text = ""
    if input_option == "Text":
        input_text = st.text_area("Enter the interview transcript:")
    else:
        video_link = st.text_input("Enter the YouTube video link:")
        if video_link:
            with st.spinner("Processing video..."):
                input_text = transcript(video_link)
                if not input_text:
                    st.error("Error processing the video. Please try again with a different link.")
                    return

    parameters = st.text_input("Enter the parameters for sentiment analysis (comma-separated):", "Enthusiasm, Communication Skills, Technical Knowledge")
    parameters = [param.strip() for param in parameters.split(",")]

    if st.button("Analyze"):
        if input_text and parameters:
            with st.spinner("Generating sentiment scores..."):
                sentiment_scores = generate_sentiment_score(input_text, parameters)
                st.subheader("Sentiment Scores")
                st.write(sentiment_scores)

                # Parse and display scores (adjust as needed based on the model's output format)
                scores = [line.split(":") for line in sentiment_scores.split("\n") if ":" in line]
                for param, score in scores:
                    param = param.strip()
                    try:
                        score = float(score.strip())
                        if score >= 4:
                            color = "#2ca02c"  # Green
                        elif score >= 3:
                            color = "#ff7f0e"  # Orange
                        else:
                            color = "#d62728"  # Red
                        st.markdown(f"**{param}**: <span style='color: {color}'>{score}/5</span>", unsafe_allow_html=True)
                    except ValueError:
                        pass

            # Generate detailed feedback
            with st.spinner("Generating detailed feedback..."):
                detailed_feedback = generate_detailed_feedback(input_text, parameters)
                st.subheader("Detailed Feedback")
                st.write(detailed_feedback)

                # Provide an option to download detailed feedback as a .txt or .pdf file
                temp_txt_path = tempfile.mktemp(suffix=".txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(detailed_feedback)

                st.download_button(
                    label="Download Detailed Feedback as .txt",
                    data=open(temp_txt_path, 'r').read(),
                    file_name="detailed_feedback.txt",
                    mime="text/plain"
                )

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, detailed_feedback)

                temp_pdf_path = tempfile.mktemp(suffix=".pdf")
                pdf.output(temp_pdf_path)

                with open(temp_pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Detailed Feedback as .pdf",
                        data=f.read(),
                        file_name="detailed_feedback.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("Please provide input and parameters for sentiment analysis.")

if __name__ == "__main__":
    main()
  
       
  
