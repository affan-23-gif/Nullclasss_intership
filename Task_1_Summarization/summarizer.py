from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import streamlit as st

def extractive_summary(text, num_sentences=5):
    """
    Generate a summary by selecting the most important sentences using LexRank.
    
    Args:
        text (str): The input document text.
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: Extractive summary.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

if __name__ == "__main__":
    st.title("Extractive Summarizer")
    text_input = st.text_area("Paste the text you want summarized")
    num_sentences = st.slider("Number of sentences in summary", 1, 10, 3)

    if st.button("Summarize"):
        if text_input.strip():
            summary = extractive_summary(text_input, num_sentences)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("Please paste some text to summarize.")
