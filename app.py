from pathlib import Path
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model(folder):
    tok = AutoTokenizer.from_pretrained(folder)
    net = AutoModelForCausalLM.from_pretrained(folder).eval()
    return tok, net

st.set_page_config(page_title="Study Tip Generator", page_icon="📚")
st.title("Study Tip Generator")
st.caption("AI-generated study tips for instructor review (ETMMGA304 Capstone).")

folder = Path("study_tip_gpt2")
if not (folder / "config.json").exists():
    st.error("Run the fine-tuning step and place the exported model beside this app.")
    st.stop()

topic = st.text_input("Enter a study topic", max_chars=120, placeholder="e.g., recursion, matrices, binary search")
if st.button("Generate"):
    if not topic.strip():
        st.warning("Enter a study topic first.")
        st.stop()
    tokenizer, model = load_model(str(folder))
    inputs = tokenizer("Topic: " + topic.strip() + "\nStudy tip:", return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=45,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    tip = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    st.success(tip or "No text was generated. Try a more specific topic.")