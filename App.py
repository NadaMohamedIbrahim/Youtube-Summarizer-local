import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(page_title="Youtube AI summarizer", page_icon="🎓", layout="wide")

# --- CACHE THE MODELS ---
@st.cache_resource(show_spinner=False)
def load_llm(context_size=16384):
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    llm = Llama(
        model_path=model_path,
        n_ctx=context_size,
        n_threads=4,
        n_gpu_layers=-1, 
        verbose=False
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return llm, tokenizer

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- HELPER FUNCTIONS ---
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."): host = host[4:]
    if host == "youtu.be": return parsed.path.lstrip("/").split("/")[0]
    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        qs = parse_qs(parsed.query)
        if qs.get("v"): return qs["v"][0]
    raise ValueError("No video ID found in URL.")

def get_text_chunks(text, tokenizer, max_tokens=2000):
    # Reduced chunk size to 2000 so the Chat context is highly specific
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def create_vector_store(chunks, embeddings):
    # Convert string chunks into LangChain Documents, then build a searchable database
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- GENERATION FUNCTIONS ---
def generate_content(prompt, system_instruction, llm, max_len, temperature):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]
    response = llm.create_chat_completion(
        messages=messages, max_tokens=max_len, temperature=temperature, repeat_penalty=1.15
    )
    return response["choices"][0]["message"]["content"].strip()

# --- MAIN UI ---
def main():
    st.title("🎓 Youtube AI Tutor & Summarizer")
    
    with st.spinner("Loading AI Models (LLM + Embeddings)..."):
        llm, tokenizer = load_llm()
        embeddings = load_embedding_model()

    with st.sidebar:
        st.header("⚙️ Settings")
        max_length = st.slider("Max Generation Tokens", 200, 2000, 800, 100)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        languages = st.text_input("Languages", "ar,en")

    url = st.text_input("🔗 Enter YouTube Video URL:")
    
    if st.button("🚀 Process Video") and url:
        try:
            with st.status("Processing Video...", expanded=True) as status:
                st.write("🔍 Extracting video ID & Fetching subtitles...")
                video_id = extract_video_id(url)
                fetched = YouTubeTranscriptApi().fetch(video_id, languages=[l.strip() for l in languages.split(',')])
                raw_text = "\n".join(s.text for s in fetched)
                
                st.write("✂️ Chunking text & Building Vector Database...")
                chunks = get_text_chunks(raw_text, tokenizer)
                
                # Save Vectorstore and Transcript to Session State so we can chat later
                st.session_state.vectorstore = create_vector_store(chunks, embeddings)
                st.session_state.raw_text = raw_text
                st.session_state.chunks = chunks
                
                # Clear chat history when a new video is processed
                st.session_state.chat_history = []
                
                status.update(label="Processing Complete!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    # --- UI TABS ---
    if "raw_text" in st.session_state:
        tab1, tab2, tab3 = st.tabs(["📝 Summary", "📚 Study Pack & Quiz", "💬 Chat with Video"])
        
        # TAB 1: SUMMARY
        with tab1:
            if st.button("Generate Summary"):
                with st.spinner("Synthesizing Summary..."):
                    sys_prompt = "You are a precise AI. Summarize the provided text comprehensively using bullet points."
                    # If video is short, summarize at once. If long, summarize chunks 
                    combined_text = "\n".join(st.session_state.chunks[:3]) # Limit to first few for speed, scale as needed
                    summary = generate_content(f"Transcript:\n{combined_text}", sys_prompt, llm, max_length, temperature)
                    st.success("Summary Generated!")
                    st.info(summary)
                    
        # TAB 2: STUDY PACK & QUIZ
        with tab2:
            if st.button("Generate Study Pack"):
                with st.spinner("Generating Flashcards and Quiz..."):
                    sys_prompt = "You are an expert educator. Extract key concepts and create a multiple-choice quiz."
                    user_prompt = f"Based on this transcript, provide:\n1. 5 Key Flashcard Terms with Definitions\n2. A 3-question Multiple Choice Quiz with an answer key at the very bottom.\n\nTranscript:\n{st.session_state.chunks[0]}..."
                    
                    study_pack = generate_content(user_prompt, sys_prompt, llm, 1200, temperature)
                    st.success("Study Pack Ready!")
                    st.markdown(study_pack)

        # TAB 3: LANGCHAIN RAG CHAT
        with tab3:
            st.markdown("### Ask questions about the video!")
            
            # Display Chat History
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat Input
            if prompt := st.chat_input("E.g., What was the main argument about X?"):
                # 1. Add user message to UI
                st.chat_message("user").markdown(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # 2. Retrieve relevant chunks using FAISS
                with st.spinner("Searching video context..."):
                    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                # 3. Formulate RAG prompt
                system_instruction = "You are a helpful AI tutor answering questions based ONLY on the provided video context. If the answer is not in the context, say 'I cannot find the answer in the video'."
                rag_prompt = f"Video Context:\n{context}\n\nUser Question: {prompt}\n\nAnswer:"
                
                # 4. Generate Answer
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = generate_content(rag_prompt, system_instruction, llm, max_length, temperature)
                        st.markdown(answer)
                
                # 5. Save to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()