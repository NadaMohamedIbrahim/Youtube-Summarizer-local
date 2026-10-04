import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer  # <-- ADDED FOR EXACT TOKENIZATION

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=" Local YouTube Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE THE MODEL & TOKENIZER ---
@st.cache_resource(show_spinner=False)
def load_model(context_size=16384):
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    # 1. Download model if not cached
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # 2. Initialize the Llama CPP engine with Qwen
    llm = Llama(
        model_path=model_path,
        n_ctx=context_size,  # Max context window
        n_threads=4,         # Adjust this based on your CPU cores
        n_gpu_layers=-1,
        verbose=False        # Hides the generation logs from terminal
    )
    
    # 3. Initialize the exact tokenizer for Qwen to perfectly count tokens
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    return llm, tokenizer

# --- HELPER FUNCTIONS ---
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        qs = parse_qs(parsed.query)
        if qs.get("v"):
            return qs["v"][0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]

    raise ValueError("No video ID found in URL.")

# --- MODIFIED: PERFECT TOKEN-BASED CHUNKING ---
def get_text_chunks(text, tokenizer, max_tokens=10000):
    """
    Uses the model's actual tokenizer to split text accurately.
    10,000 max tokens ensures it fits comfortably within the 16,384 limit.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
        
    return chunks

def summarize_chunks(chunks, llm, max_len, temperature):
    chunk_summaries = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        messages = [
            {"role": "system", "content": "You are a precise AI assistant. Summarize the following transcript exactly as it is written. DO NOT make up information. DO NOT hallucinate. If the transcript is in Arabic, reply in Arabic."},
            {"role": "user", "content": f"Here is the transcript:\n\n{chunk}\n\nPlease summarize this:"}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_len,
            temperature=temperature,
            repeat_penalty=1.15 
        )
        
        summary = response["choices"][0]["message"]["content"].strip()
        chunk_summaries.append(summary)
        
        progress_bar.progress((i + 1) / len(chunks))

    return chunk_summaries

def synthesize_summaries(summaries, llm, max_len, temperature):
    if len(summaries) == 1:
        return summaries[0]

    combined_text = "\n\n".join(summaries)
    
    messages = [
        {"role": "system", "content": "You are a precise AI assistant. Combine these summaries into a final overview. Stick strictly to the facts provided. DO NOT add outside information."},
        {"role": "user", "content": f"Partial Summaries:\n\n{combined_text}\n\nProvide the final comprehensive summary:"}
    ]
    
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_len,
        temperature=temperature,
        repeat_penalty=1.15  # <--- THIS STOPS REPETITION LOOPS
    )
    
    return response["choices"][0]["message"]["content"].strip()

# --- MAIN UI ---
def main():
    st.title("🎬 Local YouTube Summarizer")
    st.markdown("Summarize videos completely offline using **Qwen 2.5 0.5B**.")

    # Load Model (Cached) - MODIFIED to also unpack tokenizer
    with st.spinner("Loading Qwen Model and Tokenizer... (Downloads on first run)"):
        llm, tokenizer = load_model()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        max_length = st.slider(
            "Max Summary Tokens", 
            min_value=200, max_value=1500, value=800, step=100
        )
        
        temperature = st.slider(
            "Temperature (Creativity)", 
            min_value=0.0, max_value=1.0, value=0.3, step=0.1
        )
        
        languages = st.text_input("Transcript Languages (comma-separated)", value="en,ar")

    # --- MAIN CONTENT AREA ---
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        st.write("") 
        st.write("")
        summarize_button = st.button("🚀 Summarize Video", use_container_width=True, type="primary")

    if summarize_button and url:
        try:
            with st.status("Processing Video...", expanded=True) as status:
                
                st.write("🔍 Extracting video ID...")
                video_id = extract_video_id(url)
                
                st.write(f"📝 Fetching subtitles...")
                langs_list = [l.strip() for l in languages.split(',')]
                
                # --- FIXED: Restored your original API fetch logic ---
                api = YouTubeTranscriptApi()
                fetched = api.fetch(video_id, languages=langs_list)
                raw_text = "\n".join(snippet.text for snippet in fetched)
                # -----------------------------------------------------

                # --- MODIFIED: Use accurate HuggingFace tokenization ---
                st.write("✂️ Accurately chunking text by tokens...")
                chunks = get_text_chunks(raw_text, tokenizer, max_tokens=10000)
                st.write(f"Created {len(chunks)} exact token chunk(s).")
                
                st.write("🧠 Generating summaries...")
                intermediate_summaries = summarize_chunks(chunks, llm, max_length, temperature)
                
                if len(intermediate_summaries) > 1:
                    st.write("🔗 Combining chunk summaries...")
                    final_summary = synthesize_summaries(intermediate_summaries, llm, max_length, temperature)
                else:
                    final_summary = intermediate_summaries[0]
                
                status.update(label="Summarization Complete!", state="complete", expanded=False)

            st.success("Summary Generated Successfully!")
            st.markdown("### 📋 Final Summary")
            st.info(final_summary)

            with st.expander("Show Raw Transcript"):
                st.text_area("Original Transcript text", raw_text, height=300)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Make sure the video has subtitles available in the requested languages.")

if __name__ == "__main__":
    main()