from transformers import pipeline
import os


# Use a smaller, faster summarization model for improved speed
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

def summarizer_model(texts, max_length=150, min_length=50, do_sample=False):
    """
    Summarizes the provided text(s) using the pretrained summarization model.
    """
    texts = [text[:512] for text in texts[:3]]
    concatenated_text = " ".join(texts)
    summary = summarizer(concatenated_text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']
