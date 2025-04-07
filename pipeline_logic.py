# ==============================================================================
# pipeline_logic.py
#
# Backend logic for AI Storyteller Pipeline. Contains functions for each step.
# Designed to be imported and called by a frontend like Streamlit (app.py).
# Manages state via passed arguments and module-level globals for KG/DB handles,
# assuming initialize_state resets these globals appropriately for each new story run.
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import json
import os
import copy
import time
import re
import uuid
import base64

# Third-party libraries
try:
    import google.generativeai as genai
    import chromadb
    import networkx as nx
    from dotenv import load_dotenv
    from gtts import gTTS # Use gTTS for free TTS
    import openai
    import requests
except ImportError as e:
    # This error will be caught by app.py if run via Streamlit
    print(f"CRITICAL: Failed to import necessary libraries: {e}")
    print("Ensure all dependencies in requirements.txt are installed in the correct environment.")
    # Avoid exiting here if imported as module, let the caller handle it
    raise # Re-raise the import error

# ==============================================================================
# Configuration & Global Initialization (Run once on import)
# ==============================================================================

print("Initializing pipeline_logic module...")

# --- Load API Keys ---
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

if not google_api_key:
    # Let app.py handle raising the final error if needed
    print("CRITICAL WARNING: GOOGLE_API_KEY not found in .env file.")
    # Avoid raising error here directly to allow app.py to potentially start and show error

# --- Configure Google Generative AI ---
try:
    if google_api_key: # Only configure if key exists
        genai.configure(api_key=google_api_key)
        print("pipeline_logic: Google Generative AI SDK configured.")
    else:
        print("pipeline_logic: Skipping Google GenAI configuration (no API key).")
except Exception as e:
    print(f"pipeline_logic: Error configuring Google Generative AI SDK: {e}")
    # Allow app.py to handle this potentially

# --- Configure OpenAI API Key ---
if not openai_api_key:
    print("pipeline_logic: Warning: OPENAI_API_KEY not found. Image generation disabled.")
    openai.api_key = None
else:
    openai.api_key = openai_api_key
    print("pipeline_logic: OpenAI API Key configured.")

# --- Global variables for DB/KG/Embeddings ---
# These are initialized once on import, then potentially modified/reset by initialize_state
client = None
collection = None
gemini_ef = None
collection_name = "story_elements_kg_v3" # Consistent name
G = None

# --- Initialize ChromaDB Client & Embedding Function (once on import) ---
try:
    chroma_db_path = "./chroma_story_db"
    client = chromadb.PersistentClient(path=chroma_db_path)
    print(f"pipeline_logic: ChromaDB client initialized (persistent path: {chroma_db_path}).")

    if google_api_key: # Only create embedding function if key is available
         gemini_ef = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(
             api_key=google_api_key,
             task_type="RETRIEVAL_DOCUMENT"
             )
         print("pipeline_logic: Gemini embedding function created.")
    else:
         print("pipeline_logic: Warning: Skipping embedding function creation (no Google API key). DB operations requiring embeddings will fail.")

except Exception as e:
    print(f"pipeline_logic: Error initializing ChromaDB client or Embedding Function: {e}")
    client = None # Ensure client is None if init fails
    gemini_ef = None

# --- Initialize Knowledge Graph (once on import) ---
G = nx.DiGraph()
print("pipeline_logic: Initialized empty Knowledge Graph.")


# ==============================================================================
# Helper Functions
# ==============================================================================

def add_to_vector_db(text_content, metadata, doc_id):
    """Adds or updates a document in the ChromaDB collection."""
    global collection # Needs access to the current collection object
    if collection is None:
        print("Error: ChromaDB collection is not available. Cannot add vector.")
        return
    if not isinstance(text_content, str): return
    cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
    if not cleaned_text: return
    try:
        # Ensure metadata values are ChromaDB compatible (str, int, float, bool)
        safe_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                 safe_metadata[k] = v
            # else: print(f"Warning: Skipping non-primitive metadata key '{k}' for Chroma ID {doc_id}")
        collection.upsert(
            documents=[cleaned_text],
            metadatas=[safe_metadata],
            ids=[doc_id]
        )
    except Exception as e:
        print(f"Error upserting Chroma ID {doc_id}: {e}")
        print(f"Metadata causing error: {metadata}") # Log metadata on error


def call_llm(prompt, model_name="gemini-1.5-flash-latest", max_retries=3, delay=10):
    """Calls the Google Gemini API with retry logic."""
    if not google_api_key: return "ERROR: Google API Key not configured."
    # print(f"\n--- Calling LLM ({model_name}) ---")
    generation_config = {"temperature": 0.75}
    try:
        model = genai.GenerativeModel(model_name=model_name)
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                if response.parts: result = "".join(part.text for part in response.parts)
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                     reason = response.candidates[0].finish_reason; safety = response.candidates[0].safety_ratings
                     print(f"Warning: LLM call potentially blocked/finished. Reason: {reason}, Safety: {safety}")
                     result = response.text if hasattr(response, 'text') else ""
                elif hasattr(response, 'text'): result = response.text
                else: block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"; print(f"Warning: LLM Response blocked/empty. Reason: {block_reason}"); result = ""
                time.sleep(0.5); return result.strip() # Shorter delay?
            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"Rate limit exceeded. Retrying in {delay * (attempt + 1)}s... (Attempt {attempt + 1}/{max_retries})")
                if attempt + 1 == max_retries: return f"ERROR: Rate limit exceeded after {max_retries} attempts."
                time.sleep(delay * (attempt + 1))
            except google.api_core.exceptions.InvalidArgument as e: print(f"ERROR: Invalid Argument (400): {e}"); return f"ERROR: Invalid Argument - {e}"
            except Exception as e:
                error_type = type(e).__name__; print(f"ERROR: LLM call failed ({error_type}) (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt + 1 == max_retries: return f"ERROR: LLM call failed after {max_retries} attempts - {e}"
                time.sleep(delay)
    except Exception as e: print(f"ERROR: Failed to initialize Gemini model '{model_name}': {e}"); return f"ERROR: Model configuration failed - {e}"
    return "ERROR: Max retries reached or unexpected failure in LLM call."

# --- gTTS Audio Generation ---
def generate_audio_gtts(text_to_speak, output_filename="story_audio_gtts.mp3", lang='en'):
    """Generates audio from text using gTTS and saves to MP3."""
    print(f"\n-> Generating audio file via gTTS: {output_filename}...")
    if not text_to_speak: print("No text provided for gTTS generation."); return False
    try:
        tts = gTTS(text=text_to_speak, lang=lang, slow=False)
        tts.save(output_filename)
        print(f"Audio content written to file '{output_filename}'")
        return True
    except Exception as e: print(f"Error during gTTS audio generation: {e}"); return False

# --- DALL-E Cover Generation ---
def generate_cover_image(image_prompt, output_filename="book_cover.png"):
    """Generates an image using DALL-E 3 based on a prompt."""
    if not openai.api_key: print("OpenAI API key missing. Skipping image generation."); return False
    if not image_prompt: print("No prompt for image generation."); return False
    print(f"\n-> Generating cover image (DALL-E 3)...")
    try:
        response = openai.images.generate(model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1, response_format="url")
        image_url = response.data[0].url; print(f"Image generated by DALL-E 3.")
        img_response = requests.get(image_url, stream=True); img_response.raise_for_status()
        with open(output_filename, 'wb') as f:
            for chunk in img_response.iter_content(chunk_size=8192): f.write(chunk)
        print(f"Image downloaded and saved as '{output_filename}'")
        return True
    except openai.OpenAIError as e: print(f"Error generating image via OpenAI: {e}"); return False
    except requests.exceptions.RequestException as e: print(f"Error downloading image: {e}"); return False
    except Exception as e: print(f"Error processing image: {e}"); return False

# --- Outline Formatting Helper ---
def format_outline_for_display(outline):
    """Formats the outline dictionary into markdown for Streamlit display."""
    if not outline or not isinstance(outline.get("episodes"), list): return "Error: Invalid outline data."
    display_text = f"**Overall Arc:** {outline.get('overall_arc', 'N/A')}\n\n"
    display_text += "**Chapters Proposed:**\n"
    for episode in outline.get("episodes", []):
         title = episode.get('title', 'Untitled')
         ep_num = episode.get('episode_num', '?')
         goal = episode.get('summary_goal', 'N/A')
         focus = ", ".join(episode.get('character_focus', ['N/A']))
         # Optionally format beats nicely
         # beats = "\n".join([f"    - {beat}" for beat in episode.get('key_beats', ['N/A'])])
         display_text += f"### Chapter {ep_num}: {title}\n"
         display_text += f"*   **Goal:** {goal}\n"
         display_text += f"*   **Focus:** {focus}\n"
         # display_text += f"*   **Key Beats:**\n{beats}\n" # Add if desired
         display_text += "---\n" # Separator
    return display_text

# ==============================================================================
# Cell 4: Core Pipeline Functions
# ==============================================================================

def expand_concept(input_brief):
    """Takes the initial user input and expands it using Gemini."""
    print(f"\n>>> Expanding Concept...")
    prompt = f"""Expand this concept into a structured JSON: {{ "title_concept": string, "genre": string, "logline": string, "characters": [{{ "name": string, "description": string, "initial_state": string, "arc_goal": string }}], "setting": {{ "primary": string, "secondary": string | null }}, "core_conflict": string, "initial_plot_hook": string }}. Concept: "{input_brief}". Generate ONLY the valid JSON object. 'secondary' location can be null."""
    response = call_llm(prompt, model_name="gemini-1.5-flash-latest")
    if response is None or response.startswith("ERROR:"): return None
    try:
        cleaned = response.strip();
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        data = json.loads(cleaned)
        # Add basic validation here if needed before returning
        return data
    except Exception as e: print(f"Error decoding concept expansion: {e}\nResponse: {response}"); return None

def generate_outline(story_seed):
    """Generates a multi-chapter outline using Gemini."""
    print(f"\n>>> Generating Outline...")
    if not story_seed: return None
    try: seed_json = json.dumps(story_seed, indent=2)
    except TypeError as e: print(f"Error converting seed to JSON: {e}"); return None
    prompt = f"""Generate a coherent multi-chapter JSON outline (3-5 chapters ideal) for a mini-arc based on this seed: {seed_json}. Output JSON format: {{ "overall_arc": string, "episodes": [{{ "episode_num": int, "title": string, "summary_goal": string, "key_beats": [string], "character_focus": [string] }}] }}. Ensure 'episodes' list is not empty. Generate ONLY the JSON object."""
    response = call_llm(prompt, model_name="gemini-1.5-flash-latest")
    if response is None or response.startswith("ERROR:"): return None
    try:
        cleaned = response.strip();
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        outline_data = json.loads(cleaned)
        episodes = outline_data.get("episodes")
        if not isinstance(episodes, list) or not episodes: raise ValueError("Outline JSON invalid/empty episodes")
        validated_eps = [] # Validate and renumber
        for i, ep in enumerate(episodes):
            if isinstance(ep, dict):
                ep['episode_num'] = i + 1; ep.setdefault('title', f'Ch {i+1}'); ep.setdefault('summary_goal', ''); ep.setdefault('key_beats', []); ep.setdefault('character_focus', []); validated_eps.append(ep)
        outline_data["episodes"] = validated_eps
        if not outline_data["episodes"]: raise ValueError("No valid episodes after validation.")
        print(f"Outline generated ({len(validated_eps)} chapters).")
        return outline_data
    except Exception as e: print(f"Error decoding/validating outline: {e}\nResponse: {response}"); return None

def initialize_state(story_seed, story_outline):
    """Resets state, KG, DB and initializes based on seed/outline. Returns state dicts."""
    global G, collection, client, collection_name, gemini_ef # Ensure access to globals
    print("\n>>> Initializing State, KG & Vector DB...")
    if not isinstance(story_seed, dict): print("Error: Invalid story_seed."); return None, None, None, None
    if not isinstance(story_outline, dict): story_outline = {} # Allow proceeding without outline if needed

    # Reset state components
    character_db = {}
    lore_bible = {}
    episode_summaries = {}
    G = nx.DiGraph() # Reset Graph

    # Reset ChromaDB Collection
    if client and collection_name and gemini_ef:
        try:
            try: client.delete_collection(name=collection_name) # Attempt delete
            except Exception: pass # Ignore if it doesn't exist
            collection = client.get_or_create_collection(name=collection_name, embedding_function=gemini_ef) # Create/get anew
            print(f"ChromaDB collection '{collection_name}' reset/created.")
        except Exception as e: print(f"CRITICAL ERROR re-initializing ChromaDB: {e}"); raise
    else: print("CRITICAL WARNING: Chroma client/collection name/embedding function not available. DB operations will fail."); collection = None # Ensure collection is None

    # Populate state (Character DB, Lore Bible, KG Nodes, Initial Vectors)
    # (Same population logic as previous full code example...)
    print("Initializing Characters...")
    for char_data in story_seed.get("characters", []):
        if isinstance(char_data, dict) and isinstance(char_data.get("name"), str) and char_data["name"]:
            name = char_data["name"]; desc = char_data.get("description", "N/A"); state = char_data.get("initial_state", "Unknown"); goal = char_data.get("arc_goal", "N/A")
            character_db[name] = {"description": desc, "current_state": state, "arc_goal": goal, "relationships": {}, "knowledge": []}
            G.add_node(name, type='character', description=desc, state=state, goal=goal, episode_introduced=0)
            initial_desc_vdb = f"Character Profile: {name}. Desc: {desc}. Initial State: {state}. Goal: {goal}"
            add_to_vector_db(initial_desc_vdb, {"type": "character_profile", "character_name": name, "episode": 0}, f"char_{name}_ep0")
        else: print(f"Warning: Skipping invalid character data: {char_data}")

    print("Initializing Setting & Lore...")
    lore_bible = {"setting": story_seed.get("setting", {}), "conflict": story_seed.get("core_conflict", ""), "plot_hook": story_seed.get("initial_plot_hook", "")}
    setting_obj = lore_bible.get('setting', {}); primary_loc = setting_obj.get('primary'); secondary_loc = setting_obj.get('secondary')
    primary_loc_node = None;
    if isinstance(primary_loc, list) and primary_loc: primary_loc_node = str(primary_loc[0]); print(f"Warning: Primary location list, using: {primary_loc_node}")
    elif isinstance(primary_loc, str) and primary_loc: primary_loc_node = primary_loc
    secondary_loc_node = None;
    if isinstance(secondary_loc, list) and secondary_loc: secondary_loc_node = str(secondary_loc[0]); print(f"Warning: Secondary location list, using: {secondary_loc_node}")
    elif isinstance(secondary_loc, str) and secondary_loc: secondary_loc_node = secondary_loc
    if primary_loc_node: G.add_node(primary_loc_node, type='location', episode_introduced=0)
    if secondary_loc_node: G.add_node(secondary_loc_node, type='location', episode_introduced=0)
    setting_desc_vdb = f"Setting: Primary - {primary_loc_node or 'N/A'}. Secondary - {secondary_loc_node or 'N/A'}."
    conflict_desc_vdb = f"Conflict: {lore_bible.get('conflict', 'N/A')}"; hook_desc_vdb = f"Plot Hook: {lore_bible.get('plot_hook', 'N/A')}"
    add_to_vector_db(setting_desc_vdb, {"type": "setting", "episode": 0}, "setting_ep0"); add_to_vector_db(conflict_desc_vdb, {"type": "conflict", "episode": 0}, "conflict_ep0"); add_to_vector_db(hook_desc_vdb, {"type": "plot_hook", "episode": 0}, "hook_ep0")

    # Create plot_tracker
    plot_tracker = []
    # ... (Plot tracker creation logic as before, ensuring defaults) ...
    plot_tracker_episodes = story_outline.get("episodes", [])
    if isinstance(plot_tracker_episodes, list):
        for i, ep_data in enumerate(plot_tracker_episodes):
            if isinstance(ep_data, dict):
                 ep_data['episode_num'] = i + 1; new_ep_data = copy.deepcopy(ep_data); new_ep_data["status"] = "pending"; new_ep_data.setdefault('title', f'Chapter {i+1}'); new_ep_data.setdefault('summary_goal', 'Develop plot.'); new_ep_data.setdefault('key_beats', []); new_ep_data.setdefault('character_focus', []); plot_tracker.append(new_ep_data)
            else: print(f"Warning: Skipping invalid episode data for tracker: {ep_data}")

    print(f"Init Complete: Chars={len(character_db)}, KG Nodes={G.number_of_nodes()}, Chapters={len(plot_tracker)}, VDB Count={collection.count() if collection else 'N/A'}")
    # Return the state dicts/lists needed by the app
    return character_db, lore_bible, episode_summaries, plot_tracker


def assemble_context(episode_num, outline_data, character_db, lore_bible, episode_summaries):
    """Assembles context using RAG retrieval and KG facts."""
    global G, collection # Access globals
    # print(f"\n>>> Assembling Context for Chapter {episode_num}...") # Less verbose
    if not isinstance(outline_data, list): return None
    episode_outline = next((ep for ep in outline_data if isinstance(ep, dict) and ep.get("episode_num") == episode_num), None)
    if not episode_outline: return None

    context = f"--- CONTEXT FOR CHAPTER {episode_num} ---\n"; context += f"\n**Chapter Outline:**\nTitle: {episode_outline.get('title', '?')}\nGoal: {episode_outline.get('summary_goal', 'N/A')}\nKey Beats: {'; '.join(episode_outline.get('key_beats', []))}\n"
    context += "\n**Live Character States:**\n"; focused_chars_list = episode_outline.get('character_focus', [])
    chars_to_include = set(item.split(":")[0].strip() for item in focused_chars_list if isinstance(item, str) and item); focused_char_nodes = []
    for char_name, char_data in character_db.items():
        if char_name in chars_to_include:
            if isinstance(char_name, (str, int, float, tuple)): focused_char_nodes.append(char_name)
            else: continue
            context += f"- {char_name}: State = {char_data.get('current_state', '?')}"
            knowledge = char_data.get('knowledge', [])
            if knowledge: context += f" | Knows = {knowledge[-1]}\n" # Only latest knowledge shown
            else: context += "\n"
    context += "\n**Knowledge Graph Facts:**\n"; kg_facts = [] # Query KG
    for node_name in focused_char_nodes:
        if G.has_node(node_name):
            attrs = G.nodes[node_name]; kg_facts.append(f"- {node_name} [State: {attrs.get('state', '?')}]")
            for key, value in attrs.items():
                 if key.startswith('has_') and value: item_name = key[4:].replace('_',' '); kg_facts.append(f"  - Has: {item_name}")
            for neighbor, edge_data in G.adj[node_name].items(): rel = edge_data.get('relationship', '?'); kg_facts.append(f"  - {node_name} -> {rel} -> {neighbor}")
    context += "\n".join(kg_facts) + "\n" if kg_facts else "(No relevant KG facts found)\n"
    context += "\n**Retrieved Relevant History (RAG):**\n"; # Query RAG
    retrieved_context = "(RAG Disabled/Failed)"
    if collection and collection.count() > 0:
        try:
            query_text = f"Ch {episode_num} {episode_outline.get('title', '')}. Goal: {episode_outline.get('summary_goal', '')}. Chars: {', '.join(focused_char_nodes)}. Beats: {', '.join(episode_outline.get('key_beats',[]))}"
            results = collection.query(query_texts=[query_text], n_results=5, include=['documents', 'metadatas', 'distances']) # Reduced n_results
            if results and results.get('documents') and results['documents'][0]:
                retrieved_items = [];
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}; dist = results['distances'][0][i] if results.get('distances') and results['distances'][0] else None
                    if meta.get("type") == "summary" and meta.get("episode") == episode_num - 1: continue # Skip prev summary
                    if dist is not None and dist > 0.75: continue # Stricter relevance filter
                    prefix = f"- [Ep {meta.get('episode','?')}, {meta.get('type','?')}, Rel={1-dist:.2f}]: " if dist is not None else f"- [Ep {meta.get('episode','?')}]: "
                    retrieved_items.append(prefix + doc)
                retrieved_context = "\n".join(retrieved_items) if retrieved_items else "(No highly relevant history found via RAG)"
            else: retrieved_context = "(RAG returned no results)"
        except Exception as e: print(f"Error RAG query: {e}"); retrieved_context = "(RAG Error)"
    else: retrieved_context = "(Vector DB empty or unavailable)"
    context += retrieved_context + "\n"
    context += "\n**Previous Chapter Summary:**\n"; prev_ep_num = episode_num - 1 # Add Prev Summary
    context += episode_summaries.get(prev_ep_num, "(N/A - First chapter or summary missing)") + "\n"
    context += "--- END CONTEXT ---"
    # print(f"Context Assembled (~{len(context)} chars)") # Verbose logging
    return context

def generate_chapter_content(episode_num, context):
    """Generates narrative prose for a story chapter using Gemini."""
    print(f"\n>>> Generating Chapter {episode_num} Content...")
    if context is None or context.startswith("ERROR:") or not isinstance(context, str): return None
    # Shortened prompt preamble for efficiency
    prompt = f"""You are a novelist writing Chapter {episode_num}. Use the CONTEXT provided. Write engaging narrative prose (PAST TENSE, THIRD-PERSON LIMITED on focused chars). Follow Key Beats. Incorporate descriptions, actions, dialogue ("Quote," attribution.). Be consistent with ALL context (KG facts, RAG, states). Show, don't tell.\n{context}\nBegin writing Chapter {episode_num} now:"""
    content = call_llm(prompt, model_name="gemini-1.5-pro-latest"); # Pro recommended for quality
    if content and content.startswith("ERROR:"): return None
    elif not content: print("Warning: Chapter generation returned empty."); return None
    return content

def update_state_and_summarize(episode_num, generated_content, character_db, plot_tracker, episode_summaries):
    """Updates state, summarizes, adds summary to Vector DB, and updates KG based on chapter content."""
    global G, collection # Access globals
    print(f"\n>>> Updating State for Chapter {episode_num}...")
    content_available = generated_content and not generated_content.startswith("ERROR:")
    for ep in plot_tracker: # Mark status
        if isinstance(ep, dict) and ep.get("episode_num") == episode_num: ep["status"] = "generated" if content_available else "failed_generation"; break
    if not content_available: print("Skipping updates (chapter gen failed)."); return

    # --- Summarization & Add to Vector DB ---
    summary = "Summary generation failed."
    try:
        max_len = 6000; content_clip = generated_content[:max_len] + "..." if len(generated_content) > max_len else generated_content
        summary_prompt = f"""Summarize key plot advancements & character changes in this chapter excerpt (2-4 concise sentences) for NEXT chapter's context. Excerpt: ```\n{content_clip}\n``` Generate only the summary text."""
        summary_response = call_llm(summary_prompt, model_name="gemini-1.5-flash-latest")
        if summary_response and not summary_response.startswith("ERROR:"):
            summary = summary_response.strip(); add_to_vector_db(summary, {"type": "summary", "episode": episode_num}, f"summary_ep{episode_num}")
            # print(f"Summary: {summary}") # Optional print
        else: print(f"Warning: Summary gen failed. Resp: {summary_response}")
    except Exception as e: print(f"Error summary/indexing: {e}")
    episode_summaries[episode_num] = summary

    # --- State & KG Update (LLM Analysis) ---
    # print("Attempting state/KG update analysis...") # Optional print
    try:
        max_len_state = 6000; content_clip_state = generated_content[:max_len_state] + "..." if len(generated_content) > max_len_state else generated_content
        update_prompt = f"""Analyze chapter excerpt & current states. Output JSON of significant changes ONLY. Keys=char names. Values=dict with optional keys: "updated_state"(str), "new_knowledge"(list[str]), "new_relationships"(list[[subj,pred,obj]]), "gained_items"(list[str]). Focus on EXPLICIT events. States: {json.dumps(character_db)}. Excerpt: ```\n{content_clip_state}\n``` Only JSON output (or {{}} if no changes)."""
        suggestion_json = call_llm(update_prompt, model_name="gemini-1.5-flash-latest")
        if suggestion_json and not suggestion_json.startswith("ERROR:"):
            cleaned = suggestion_json.strip(); # Clean response
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            suggested_updates = {} if not cleaned else json.loads(cleaned)
            if isinstance(suggested_updates, dict) and suggested_updates:
                print("Applying suggested state/KG updates:")
                for char_name, updates in suggested_updates.items():
                    if isinstance(char_name, str) and char_name in character_db and G.has_node(char_name):
                        # print(f"- Updating {char_name}:") # Verbose print
                        # Update Character DB (Live State)
                        if updates.get("updated_state"): character_db[char_name]["current_state"] = updates["updated_state"] #; print(f"  DB State Updated")
                        if updates.get("new_knowledge"): new = [k for k in updates["new_knowledge"] if isinstance(k,str) and k and k not in character_db[char_name].get("knowledge",[])]; character_db[char_name].setdefault("knowledge",[]).extend(new) #; if new: print("  DB Knowledge Added")
                        # Update Knowledge Graph
                        if updates.get("updated_state"): G.nodes[char_name]['state'] = updates["updated_state"] #; print("  KG State Updated")
                        if updates.get("new_relationships"):
                            for rel in updates["new_relationships"]:
                                if isinstance(rel,list) and len(rel)==3: subj,pred,obj = str(rel[0]),str(rel[1]),str(rel[2]); G.add_edge(subj, obj, relationship=pred, episode_formed=episode_num) #; print(f"  KG Edge: {subj}-{pred}->{obj}")
                        if updates.get("gained_items"):
                            for item in updates["gained_items"]:
                                if isinstance(item,str) and item: item_nn = item; G.add_node(item_nn, type='item') if not G.has_node(item_nn) else None; G.add_edge(char_name, item_nn, relationship='has', episode_gained=episode_num); G.nodes[char_name]['has_'+re.sub(r'\W+','_',item.lower())] = True #; print(f"  KG Edge: {char_name}-has->{item_nn}")
            # else: print("No significant state/KG changes suggested.") # Optional print
        # else: print(f"State/KG update analysis failed. Response: {suggestion_json}") # Optional print
    except Exception as e: print(f"Error during state/KG update: {e}")
    # print("-" * 10) # Optional separator

# ==============================================================================
# End of pipeline_logic.py
# ==============================================================================