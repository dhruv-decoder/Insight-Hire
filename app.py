import streamlit as st
import openai
from pytube import YouTube
import os
import tempfile
from fpdf import FPDF
# Set your OpenAI API key
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]


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

    Provide your scores in the format: Parameter: Score.
    '''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        top_p=0.7,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    return response['choices'][0]['message']['content']

def generate_detailed_feedback(input_text, parameters):
    prompt = f'''
    As an experienced interview reviewer, provide a detailed analysis of the candidate's responses based on the following parameters: {', '.join(parameters)}. 
    
    Include specific examples, quotes, and adjectives from the transcript that support your analysis. Offer actionable insights and recommendations for the hiring team to make informed decisions. Summarize the candidate's overall sentiment and demeanor.
    '''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        top_p=0.7,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    return response['choices'][0]['message']['content']

def transcript(video_link):
    try:
        # Create a YouTube object
        yt = YouTube(video_link)

        # Choose the stream with the desired quality and resolution
        stream = yt.streams.filter(only_audio=True).first()

        # Download the video to a temporary location
        temp_file_path = stream.download()

        print(f"Video '{yt.title}.mp4' downloaded successfully!")

        # Transcribe the video using OpenAI's Whisper
        with open(temp_file_path, 'rb') as audio_data:
            total_transcript = (openai.Audio.transcribe("whisper-1", audio_data))["text"]
            print("Done with the video processing\n")

        # Remove the temporary file
        os.remove(temp_file_path)

        return total_transcript

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
                sentiment_scores = sentiment_scores.strip().split("\n")

                st.subheader("Sentiment Scores")
                valid_scores = []
                for score in sentiment_scores:
                    if ":" in score:
                        param, score_value = score.split(":")
                        param = param.strip()
                        score_value = score_value.strip()
                        if param in parameters:
                            try:
                                score_value = float(score_value.split("/")[0].strip())
                                valid_scores.append((param, score_value))
                                if score_value >= 4:
                                    color = "#2ca02c"  # Green
                                elif score_value >= 3:
                                    color = "#ff7f0e"  # Orange
                                else:
                                    color = "#d62728"  # Red
                                st.markdown(f"**{param}**: <span style='color: {color}'>{score_value}/5</span>", unsafe_allow_html=True)
                            except ValueError:
                                pass

            if valid_scores:
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
