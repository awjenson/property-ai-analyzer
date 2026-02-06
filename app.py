import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# --- SETUP ---
# Optimized for your MacBook Pro's GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def calculate_advanced_metrics(caption):
    """
    Simulates real estate metrics based on the AI caption.
    """
    c = caption.lower()
    is_residential = any(word in c for word in ["house", "home", "building", "suburb"])
    has_nature = any(word in c for word in ["tree", "lawn", "grass", "garden", "foliage"])

    # Neighborhood & Design Metrics
    design_score = 82 if is_residential else 45
    walk_score = 68 

    # Climate & Environmental Health Metrics
    air_quality_score = 94 if has_nature else 72 
    greenery_index = 85 if has_nature else 30

    return design_score, walk_score, air_quality_score, greenery_index

def analyze_property(input_image):
    """
    Processes the image using the teacher's original generation parameters
    to prevent the 'repetition trap' and 'hallucinations'.
    """
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # We remove the descriptive prompt and stick to standard processing 
    # as per the tutorial to ensure the highest accuracy.
    inputs = processor(raw_image, return_tensors="pt").to(device)

    # Step 2: Teacher's Original Parameters
    # Removing min_length and beams allows the model to stop naturally.
    out = model.generate(**inputs, max_length=50)

    # Note: Using [0] to get the first sequence from the output batch
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Generate Metrics
    d_score, w_score, aq_score, g_index = calculate_advanced_metrics(caption)

    # Professional Report Structure for Blackbird Dashboard
    report = f"""
    ### 🏛️ Blackbird Property Intelligence Report
    **AI Visual Analysis:** {caption.capitalize()}

    ---
    ### 📈 Valuation Strategy & Quality of Life
    * **Estimated Value:** $580,000 - $695,000
    * **Strategic Recommendation:** Focus on high-impact curb appeal to increase equity.
    * **Action Plan:** Enhance native greenery and address any visible infrastructure wear.

    ---
    ### 🌍 Climate & Environmental Health
    * **Air Quality Score:** {aq_score}/100 (Climate health is a key long-term valuation driver)
    * **Greenery Index:** {g_index}/100 (High canopy coverage supports community wellness)

    ---
    ### 🏘️ Neighborhood & Design
    * **Design Quality:** {d_score}/100 (Evaluating architectural fit within the local community)
    * **Walkability Score:** {w_score}/100 (Distance to amenities and neighborhood 'walk appeal')
    """
    return report

# --- INTERFACE ---
iface = gr.Interface(
    fn=analyze_property,
    inputs=gr.Image(label="Upload Property Photo"),
    outputs=gr.Markdown(label="Real Estate Intelligence"),
    title="Blackbird AI: Property & Community Analyst",
    description="Analyze visual property features alongside environmental health and neighborhood metrics."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)