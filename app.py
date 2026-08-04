import os
import time
import nltk
import spaces
import gradio as gr
from huggingface_hub import HfFileSystem

# 1. Map the runtime directory (Safe from Docker Permission errors)
DOWNLOAD_DIR = "/home/user/app/cache/nltk"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

if DOWNLOAD_DIR not in nltk.data.path:
    nltk.data.path.insert(0, DOWNLOAD_DIR)

# 2. HUGGING FACE BUCKET SYNC
BUCKET_URI = "buckets/KingOfThoughtFleuren/Computational_Consciousness_Engine-storage/nltk_data"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("Connecting to Hugging Face Storage Bucket...")
try:
    # Initialize File System (Authenticates automatically via HF_TOKEN secret)
    fs = HfFileSystem(token=HF_TOKEN)
    
    # If the bucket already has the data, download it directly from your bucket!
    if fs.exists(BUCKET_URI):
        print("Retrieving persistent NLTK data from Bucket...")
        fs.get(BUCKET_URI, DOWNLOAD_DIR, recursive=True)
        print("Bucket retrieval successful!")
        
    # If the bucket is empty, download from NLTK and push it to the bucket!
    else:
        print("Bucket empty. Downloading fresh NLTK data...")
        nltk.download('punkt', download_dir=DOWNLOAD_DIR, quiet=True)
        nltk.download('averaged_perceptron_tagger', download_dir=DOWNLOAD_DIR, quiet=True)
        nltk.download('stopwords', download_dir=DOWNLOAD_DIR, quiet=True)
        nltk.download('wordnet', download_dir=DOWNLOAD_DIR, quiet=True)
        
        if HF_TOKEN:
            print("Uploading fresh NLTK data to Bucket for future persistence...")
            fs.put(DOWNLOAD_DIR, BUCKET_URI, recursive=True)
            print("Upload to Bucket complete!")

except Exception as e:
    print(f"Bucket Sync Warning: {e}")
    print("Falling back to local session download...")
    nltk.download('punkt', download_dir=DOWNLOAD_DIR, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=DOWNLOAD_DIR, quiet=True)
    nltk.download('stopwords', download_dir=DOWNLOAD_DIR, quiet=True)
    nltk.download('wordnet', download_dir=DOWNLOAD_DIR, quiet=True)

time.sleep(0.1)

# --- Clean Native Application Imports ---
try:
    from computational_consciousness_engine.runner import run_simulation
    from computational_consciousness_engine.text_runner import run_text_mapping_demo
    from computational_consciousness_engine.input_mapping.intent_parser_nltk import IntentParserNLTK
    from computational_consciousness_engine.math_formalization import CCMathFormalizer
    print("Imported engine modules successfully!")
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Could not find the engine: {e}")
    raise
# --- Application imports (after nltk path insertion) ---
import gradio as gr

# Ensure your local package is importable; try a robust import with fallback
try:
    from computational_consciousness_engine.runner import run_simulation
    from computational_consciousness_engine.text_runner import run_text_mapping_demo
    from computational_consciousness_engine.input_mapping.intent_parser_nltk import IntentParserNLTK
    from computational_consciousness_engine.math_formalization import CCMathFormalizer
    print("Imported computational_consciousness_engine package modules successfully.")
except Exception as e:
    # Try to ensure repo root is on sys.path and retry once
    print("Initial import of computational_consciousness_engine failed:", e)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from computational_consciousness_engine.runner import run_simulation
        from computational_consciousness_engine.text_runner import run_text_mapping_demo
        from computational_consciousness_engine.input_mapping.intent_parser_nltk import IntentParserNLTK
        from computational_consciousness_engine.math_formalization import CCMathFormalizer
        print("Imported computational_consciousness_engine after sys.path insert.")
    except Exception as e2:
        print("Failed to import computational_consciousness_engine package after sys.path insert:", e2)
        # Re-raise so the Space shows the error clearly
        raise

# Instantiate parsers/formalizers
intent_parser = IntentParserNLTK()
formalizer = CCMathFormalizer()

# --- App logic functions ---
def interpret_math_request(text):
    try:
        intent = intent_parser.parse(text)
    except Exception as e:
        return {"error": f"Intent parsing failed: {e}"}

    try:
        op = intent.get("operation")
        if op == "limit":
            return formalizer.formalize_equilibrium_limit()
        if op == "surface_area":
            return formalizer.formalize_conal_manifold_geometry(5.0, 1.0)
        if op == "mutation_count":
            return {"meaning": "Mutation count is determined by MStringVectorizer."}
        return {"meaning": "Unknown operation", "intent": intent}
    except Exception as e:
        return {"error": f"Formalization failed: {e}", "intent": intent}

@spaces.GPU
def simulate_generations(num_generations, steps_per_gen):
    import io
    import sys

def simulate_generations(num_generations, steps_per_gen):
    import io
    import sys
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        run_simulation(num_generations=int(num_generations), steps_per_gen=int(steps_per_gen))
    except Exception as e:
        buffer.write(f"\n[ERROR] Simulation failed: {e}\n")
    finally:
        sys.stdout = old_stdout

    return buffer.getvalue()

def map_text_to_traversal(text):
    """
    Correctly set the module-level PHILOSOPHICAL_TEXT variable and run the demo.
    Uses module import to mutate the variable in-place.
    """
    import io
    import sys
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        import computational_consciousness_engine.text_runner as tr
        # Set the module-level variable used by the demo
        tr.PHILOSOPHICAL_TEXT = text
        tr.run_text_mapping_demo()
    except Exception as e:
        buffer.write(f"\n[ERROR] Mapping failed: {e}\n")
    finally:
        sys.stdout = old_stdout

    return buffer.getvalue()

# --- Gradio UI ---
with gr.Blocks(title="Computational Consciousness Engine") as demo:
    gr.Markdown(
        """
        # 🧠 Computational Consciousness Engine  
        ### Axiom‑Driven Synthetic Cognition • PMCA Substrate • Conal Geometry • Mutation Dynamics  
        ---
        """
    )

    with gr.Tab("Math Interpreter (NLTK → PMCA)"):
        math_input = gr.Textbox(label="Ask a math question")
        math_output = gr.JSON(label="PMCA Result")
        math_button = gr.Button("Interpret")
        math_button.click(interpret_math_request, math_input, math_output)

    with gr.Tab("0 → -1 → 0 Simulation"):
        gr.Markdown(
            """\
            ### Generational Cycle Runner

            Execute full PMCA traversal cycles, including conal unfolding, mutation absorption,
            equilibrium evaluation, and Q‑operator transitions.
            """
        )

        with gr.Row():
            num_generations = gr.Slider(1, 20, value=8, label="Generations", interactive=True)
            steps = gr.Slider(10, 300, value=100, label="Steps per Generation", interactive=True)

        sim_button = gr.Button("🚀 Run Simulation", variant="primary")
        sim_output = gr.Textbox(
            label="Simulation Output",
            lines=35,
            show_copy_button=True
        )

        sim_button.click(simulate_generations, [num_generations, steps], sim_output)

    with gr.Tab("Text → Traversal Space Mapper"):
        gr.Markdown(
            """\
            ### Natural Language → Chaos Shards → Traversal Atlas

            Map raw text into the PMCA chaos pool, absorb novelty, unfold the manifold,
            and generate the 2D/3D traversal atlas.
            """
        )

        input_text = gr.Textbox(
            label="Input Text",
            lines=10,
            placeholder="Paste philosophical or analytical text here...",
            show_copy_button=True
        )

        map_button = gr.Button("🧭 Map Text to Traversal Space", variant="primary")
        map_output = gr.Textbox(
            label="Traversal Output",
            lines=35,
            show_copy_button=True
        )

        map_button.click(map_text_to_traversal, input_text, map_output)

    gr.Markdown("---")
    gr.Markdown(
        "### 🔧 Engine Version: 1.0.0 • PMCA Substrate Active • Conal Geometry Verified\n"
        "Built for Hugging Face Spaces • Gradio 4.31 • Python 3.12"
    )

# Launch
if __name__ == "__main__":
    demo.launch()
