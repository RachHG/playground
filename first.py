import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util 
import numpy as np 
import pandas as pd 

# Load the pre-trained Hugging Face model 
@st.cache_resource 
def load_model(): 
   return SentenceTransformer('all-MiniLM-L6-v2') 

model = load_model() 

def extract_questions_from_pdf(uploaded_file): 
   """Extracts text from a PDF and identifies potential questions.""" 
   questions = [] 
   if uploaded_file is not None: 
       try: 
           pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
           for page_num in range(len(pdf_document)): 
               page = pdf_document.load_page(page_num) 
               text = page.get_text() 
               # A simple heuristic to identify questions - lines ending with '?' 
               for line in text.split('\n'): 
                   if line.strip().endswith('?'): 
                       questions.append(line.strip()) 
       except Exception as e: 
           st.error(f"Error processing {uploaded_file.name}: {e}") 
   return questions 

def main(): 
   st.title("Metadata Harmonization Tool") 

   st.write("Upload two PDF files to compare the questions within them.") 

   col1, col2 = st.columns(2) 

   with col1: 
       pdf_file_1 = st.file_uploader("Upload first PDF", type="pdf") 

   with col2: 
       pdf_file_2 = st.file_uploader("Upload second PDF", type="pdf") 

   if pdf_file_1 and pdf_file_2: 
       with st.spinner('Extracting questions from PDFs...'): 
           questions1 = extract_questions_from_pdf(pdf_file_1) 
           questions2 = extract_questions_from_pdf(pdf_file_2) 

       if questions1 and questions2: 
           st.success(f"Found {len(questions1)} questions in the first PDF and {len(questions2)} in the second.") 

           with st.spinner('Vectorizing questions and calculating similarity...'): 
               # Generate embeddings for the questions 
               embeddings1 = model.encode(questions1, convert_to_tensor=True) 
               embeddings2 = model.encode(questions2, convert_to_tensor=True) 

               # Compute cosine similarity 
               cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2) 

               # Find the best match for each question in the first PDF 
               results = [] 
               for i, question1 in enumerate(questions1): 
                   best_match_score = -1 
                   best_match_question = "" 
                   for j, question2 in enumerate(questions2): 
                       if cosine_scores[i][j] > best_match_score: 
                           best_match_score = cosine_scores[i][j] 
                           best_match_question = question2 
                   results.append([question1, best_match_question, f"{best_match_score.item():.4f}"]) 

           st.subheader("Comparison Results") 

           if results: 
               df = pd.DataFrame(results, columns=["Question from PDF 1", "Most Similar Question from PDF 2", "Similarity Score"]) 
               st.dataframe(df) 
           else: 
               st.write("Could not find any matching questions.") 

       else: 
           st.warning("Could not find any questions in one or both of the PDF files. Please ensure the PDFs contain text and questions end with a '?'.") 

if __name__ == "__main__": 
   main() 
