# ==============================================================================
# story_pipeline.py
#
# AI Storyteller Pipeline with Interactive Outline, RAG (ChromaDB),
# Knowledge Graph (NetworkX), Novel-Style Generation, TTS (gTTS - FREE),
# and Cover Generation (DALL-E 3).
# Designed for local execution (e.g., in VS Code).
# ==============================================================================

# ==============================================================================
# Cell 1: Imports and Setup
# ==============================================================================
import json
import os
import copy
import time
import re
import uuid
import base64

# Third-party libraries (ensure installed via requirements.txt)
try:
    import google.generativeai as genai
    import chromadb
    import networkx as nx
    from dotenv import load_dotenv
    from gtts import gTTS # Use gTTS for free TTS
    import openai
    import requests # For downloading DALL-E image URL
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure all dependencies are installed from requirements.txt in your virtual environment.")
    print("Run: pip install -r requirements.txt")
    exit(1)

print("Loading environment and libraries...")

# ==============================================================================
# Cell 2: Configuration & Initialization
# ==============================================================================

# --- Load API Keys from .env file ---
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file with your key.")

# --- Configure Google Generative AI ---
try:
    genai.configure(api_key=google_api_key)
    print("Google Generative AI SDK configured.")
except Exception as e:
    print(f"Error configuring Google Generative AI SDK: {e}")
    raise

# --- Configure OpenAI API Key ---
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in .env file. Image generation will be disabled.")
    openai.api_key = None
else:
    openai.api_key = openai_api_key
    print("OpenAI API Key configured.")

# --- Global variables for DB/KG/Embeddings (initialized below) ---
client = None
collection = None
gemini_ef = None
collection_name = "story_elements_kg_v3" # Use a distinct name for this version
G = None

# --- Initialize ChromaDB (Persistent Client) ---
try:
    chroma_db_path = "./chroma_story_db" # Creates folder in project dir
    client = chromadb.PersistentClient(path=chroma_db_path)
    print(f"ChromaDB using persistent storage at: {chroma_db_path}")

    # Define embedding function (requires Google API key to be configured)
    gemini_ef = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=google_api_key,
        task_type="RETRIEVAL_DOCUMENT"
        )

    # Note: Collection creation/retrieval moved to initialize_state for cleaner reset

except Exception as e:
    print(f"Error initializing ChromaDB Persistent Client or Embedding Function: {e}")
    raise

# --- Initialize Knowledge Graph ---
G = nx.DiGraph() # Directed graph - Initialized here, reset in initialize_state
print("Initialized empty Knowledge Graph (NetworkX DiGraph).")

# ==============================================================================
# Cell 3: Helper Functions (ChromaDB, LLM Call, TTS, Image Gen)
# ==============================================================================

def add_to_vector_db(text_content, metadata, doc_id):
    """Adds or updates a document in the ChromaDB collection."""
    global collection # Access the globally defined collection
    if collection is None:
        print("Error: ChromaDB collection not initialized. Cannot add vector.")
        return
    if not isinstance(text_content, str): # Ensure input is string
        # print(f"Warning: Skipping non-string content for Chroma ID {doc_id}")
        return
    cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
    if not cleaned_text:
        # print(f"Warning: Skipping empty document for ID {doc_id}")
        return
    try:
        collection.upsert(
            documents=[cleaned_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        # print(f"Upserted Chroma ID: {doc_id}, Type: {metadata.get('type')}") # Verbose
    except Exception as e:
        print(f"Error upserting Chroma ID {doc_id}: {e}")


def call_llm(prompt, model_name="gemini-1.5-flash-latest", max_retries=3, delay=10):
    """Calls the Google Gemini API with retry logic."""
    # print(f"\n--- Calling LLM ({model_name}) ---") # Less verbose logging
    generation_config = {"temperature": 0.75} # Slightly higher creativity

    try:
        model = genai.GenerativeModel(model_name=model_name)
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                    # safety_settings=... # Add if needed
                    )

                # Handle response access carefully
                if response.parts:
                     result = "".join(part.text for part in response.parts)
                # Check candidates for potential blocks even if parts exists
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                     reason = response.candidates[0].finish_reason
                     safety_ratings = response.candidates[0].safety_ratings
                     print(f"Warning: LLM call potentially blocked or finished unexpectedly. Reason: {reason}")
                     # print(f"Safety Ratings: {safety_ratings}") # More verbose
                     result = response.text if hasattr(response, 'text') else "" # Still try to get text
                elif hasattr(response, 'text'):
                     result = response.text
                else: # No parts, no text, likely fully blocked
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    print(f"Warning: LLM Response blocked or empty. Reason: {block_reason}")
                    result = ""

                time.sleep(1) # Small delay after success
                return result.strip()

            # Specific exception handling from google.api_core is useful
            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"Rate limit likely exceeded. Retrying in {delay * (attempt + 1)}s... (Attempt {attempt + 1}/{max_retries})")
                if attempt + 1 == max_retries: return f"ERROR: Rate limit exceeded after {max_retries} attempts."
                time.sleep(delay * (attempt + 1)) # Exponential backoff
            except google.api_core.exceptions.InvalidArgument as e:
                print(f"ERROR: Invalid Argument (400) - Check prompt or parameters: {e}")
                return f"ERROR: Invalid Argument - {e}" # Don't retry
            except Exception as e: # Catch other potential API or network errors
                error_type = type(e).__name__
                print(f"ERROR: LLM call failed ({error_type}) (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt + 1 == max_retries: return f"ERROR: LLM call failed after {max_retries} attempts - {e}"
                time.sleep(delay)
    except Exception as e:
        # Errors during model instantiation
        print(f"ERROR: Failed to initialize Gemini model '{model_name}': {e}")
        return f"ERROR: Model configuration failed - {e}"

    return "ERROR: Max retries reached or unexpected failure in LLM call."

# --- generate_audio function using gTTS ---
def generate_audio_gtts(text_to_speak, output_filename="story_audio_gtts.mp3", lang='en'):
    """Generates audio from text using gTTS and saves to MP3."""
    print(f"\n-> Generating audio file via gTTS: {output_filename}...")
    if not text_to_speak:
        print("No text provided for gTTS generation.")
        return False
    try:
        # Create gTTS object
        tts = gTTS(text=text_to_speak, lang=lang, slow=False)
        # Save the audio file
        tts.save(output_filename)
        print(f"Audio content written to file '{output_filename}'")
        return True # Indicate success
    except Exception as e:
        print(f"Error during gTTS audio generation: {e}")
        return False

# --- Cover Generation function ---
def generate_cover_image(image_prompt, output_filename="book_cover.png"):
    """Generates an image using DALL-E 3 based on a prompt."""
    if not openai.api_key: print("OpenAI API key missing. Skipping image generation."); return False
    if not image_prompt: print("No prompt for image generation."); return False

    print(f"\n-> Generating cover image (DALL-E 3)...")
    try:
        response = openai.images.generate(
            model="dall-e-3", prompt=image_prompt,
            size="1024x1024", quality="standard", n=1,
            response_format="url" # Easier for direct download
        )
        image_url = response.data[0].url
        print(f"Image generated by DALL-E 3.")

        # Download and Save
        img_response = requests.get(image_url, stream=True)
        img_response.raise_for_status() # Raise error for bad responses (4xx or 5xx)
        with open(output_filename, 'wb') as f:
            for chunk in img_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded and saved as '{output_filename}'")
        return True

    except openai.OpenAIError as e: # Catch specific OpenAI errors
        print(f"Error generating image via OpenAI: {e}")
        return False
    except requests.exceptions.RequestException as e: # Catch download errors
         print(f"Error downloading image: {e}")
         return False
    except Exception as e: # Catch other errors like file saving
        print(f"Error processing image: {e}")
        return False

# ==============================================================================
# Cell 4: Pipeline Module Functions (KG+RAG Version)
# ==============================================================================

# --- Concept Expansion ---
def expand_concept(input_brief):
    """Takes the initial user input and expands it using Gemini."""
    print(f"\n-> Expanding Concept: {input_brief}")
    prompt = f"""
    You are a creative story assistant. Expand this concept into a structured JSON format containing:
    - title_concept (string)
    - genre (string)
    - logline (string) - Should clearly state the core premise.
    - characters (list of objects, each with 'name' (string), 'description' (string), 'initial_state' (string), 'arc_goal' (string))
    - setting (object with 'primary' (string), 'secondary' (string or null) locations - values must be strings or null)
    - core_conflict (string)
    - initial_plot_hook (string)

    Concept: "{input_brief}"

    Generate only the JSON object, starting with {{ and ending with }}. Do not include ```json markdown or any other text before or after the JSON. Ensure the logline is compelling and all required fields are present. 'secondary' location can be null if not applicable.
    """
    response = call_llm(prompt, model_name="gemini-1.5-flash-latest")
    if response is None or response.startswith("ERROR:"): return None
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-3]
        data = json.loads(cleaned_response) # Use cleaned_response
        # Validation
        required_keys = ['title_concept', 'genre', 'logline', 'characters', 'setting', 'core_conflict', 'initial_plot_hook']
        if not all(k in data for k in required_keys):
             print(f"Warning: Concept JSON missing required keys. Found: {list(data.keys())}")
        setting = data.get('setting', {})
        if not isinstance(setting.get('primary'), str): print("Warning: Primary setting location not a string.")
        # Allow secondary to be string or None
        if not (isinstance(setting.get('secondary'), str) or setting.get('secondary') is None):
             print("Warning: Secondary setting location is not a string or null.")
        return data
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding JSON from concept expansion: {e}\nLLM Response: {response}")
        return None

# --- Outline Generation ---
def generate_outline(story_seed):
    """Generates a multi-chapter outline using Gemini, letting it determine the chapter count."""
    print(f"\n-> Generating Initial Outline Proposal...")
    if not story_seed: print("Error: Cannot generate outline without story seed."); return None
    try: seed_json = json.dumps(story_seed, indent=2)
    except TypeError as e: print(f"Error converting story_seed to JSON: {e}"); return None

    prompt = f"""
    You are a master story outliner. Based on the following story seed, propose a coherent multi-chapter outline (suggesting a reasonable number of chapters, typically 3-5, to tell a complete mini-arc based on the logline).

    The output MUST be a JSON object with:
    - overall_arc (string describing the planned progression across the proposed chapters)
    - episodes (list of objects, each with 'episode_num' (integer, starting from 1), 'title' (string), 'summary_goal' (string), 'key_beats' (list of strings), 'character_focus' (list of strings containing character names))

    Ensure the outline develops the characters and conflict logically across the proposed chapters, creating a satisfying narrative arc from beginning to end for this specific story concept.

    Story Seed:
    {seed_json}

    Generate only the JSON object, starting with {{ and ending with }}. Do not include ```json markdown or any other text before or after the JSON. Ensure the 'episodes' list is not empty.
    """
    response = call_llm(prompt, model_name="gemini-1.5-flash-latest")
    if response is None or response.startswith("ERROR:"): return None
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-3]
        outline_data = json.loads(cleaned_response)
        # Validation
        episodes = outline_data.get("episodes")
        if not isinstance(episodes, list) or not episodes:
             print("Error: Outline JSON invalid or 'episodes' list is empty.")
             print(f"LLM Response: {response}")
             return None
        # Validate/fix episode numbers and structure
        validated_episodes = []
        for i, ep in enumerate(episodes):
             if isinstance(ep, dict):
                 ep['episode_num'] = i + 1 # Enforce sequential numbering
                 ep.setdefault('title', f'Chapter {i+1}')
                 ep.setdefault('summary_goal', 'Develop the plot.')
                 ep.setdefault('key_beats', [])
                 ep.setdefault('character_focus', [])
                 validated_episodes.append(ep)
             else:
                 print(f"Warning: Skipping invalid episode data in outline: {ep}")
        outline_data["episodes"] = validated_episodes
        if not outline_data["episodes"]:
            print("Error: No valid episodes found in outline after validation.")
            return None
        print(f"Generated outline proposal with {len(outline_data['episodes'])} chapters.")
        return outline_data
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Error decoding/validating outline JSON: {e}\nLLM Response: {response}")
        return None

# --- Interactive Outline Review ---
def review_and_confirm_outline(story_seed, initial_outline):
    """Displays the outline and asks the user for confirmation or feedback."""
    current_outline = initial_outline
    max_retries = 2

    for attempt in range(max_retries + 1):
        print("\n" + "="*15 + f" Outline Proposal (Attempt {attempt + 1}/{max_retries + 1}) " + "="*15)
        if not current_outline or not isinstance(current_outline.get("episodes"), list):
             print("Error: Invalid outline data for review."); return None

        print(f"Overall Arc Concept: {current_outline.get('overall_arc', 'N/A')}")
        print("-" * 40)
        for episode in current_outline.get("episodes", []):
             print(f"Chapter {episode.get('episode_num', '?')}: {episode.get('title', 'Untitled')}")
             print(f"  Goal: {episode.get('summary_goal', 'N/A')}")
             print("  Focus: " + ", ".join(episode.get('character_focus', ['N/A'])))
             print("-" * 20)
        print("=" * 60)

        while True:
            try:
                 feedback = input("Enter 'yes' to accept, 'no' to reject, or provide feedback to modify: ").strip().lower()
            except EOFError:
                 print("\nInput ended unexpectedly. Rejecting outline.")
                 return None

            if feedback == 'yes': print("Outline accepted!"); return current_outline
            elif feedback == 'no': print("Outline rejected."); return None
            elif feedback:
                if attempt < max_retries:
                    print("Attempting regeneration with feedback...")
                    try: seed_json = json.dumps(story_seed, indent=2)
                    except TypeError as e: print(f"Error preparing seed JSON: {e}"); break

                    regen_prompt = f"""
                    You are a master story outliner revising a previous outline based on user feedback.
                    Original Story Seed: {seed_json}
                    Previous Outline: {json.dumps(current_outline, indent=2)}
                    User Feedback: "{feedback}"
                    Generate a NEW, revised multi-chapter outline (JSON object, same structure: 'overall_arc', 'episodes' list) incorporating the feedback while maintaining coherence. Ensure 'episodes' is not empty.
                    Generate only the JSON object starting with {{ and ending with }}. No ```json markdown.
                    """
                    new_outline_response = call_llm(regen_prompt, model_name="gemini-1.5-flash-latest")
                    if new_outline_response and not new_outline_response.startswith("ERROR:"):
                        try:
                             # Re-use validation logic from generate_outline
                             cleaned_response = new_outline_response.strip()
                             if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[7:]
                             if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-3]
                             new_outline_data = json.loads(cleaned_response)
                             episodes = new_outline_data.get("episodes")
                             if isinstance(episodes, list) and episodes:
                                  validated_eps = []
                                  for i, ep in enumerate(episodes):
                                       if isinstance(ep, dict):
                                           ep['episode_num'] = i + 1
                                           ep.setdefault('title', f'Chapter {i+1}')
                                           ep.setdefault('summary_goal', 'Develop plot.')
                                           ep.setdefault('key_beats', [])
                                           ep.setdefault('character_focus', [])
                                           validated_eps.append(ep)
                                  new_outline_data["episodes"] = validated_eps
                                  if new_outline_data["episodes"]: # Check if any valid episodes remain
                                       current_outline = new_outline_data # Update current outline
                                       print("Regeneration complete. Review new proposal.")
                                       break # Break inner while to re-display
                                  else: print("Regeneration failed: No valid episodes in response structure.")
                             else: print("Regeneration failed: Invalid structure in LLM response.")
                        except (json.JSONDecodeError, AttributeError, KeyError) as e:
                             print(f"Error decoding/validating regenerated outline: {e}\nLLM Response: {new_outline_response}")
                        # If validation fails, loop back to ask for input again for the *current* outline
                    else: print(f"Regeneration failed (LLM error). Try again or enter 'no'. LLM Response: {new_outline_response}")
                else: print("Max edit attempts reached. Please enter 'yes' or 'no'.")
            else: print("Invalid input. Enter 'yes', 'no', or feedback.")

    print("Outline review failed after multiple attempts.")
    return None

# --- State Initialization (Adds to KG & Vector DB) ---
# --- Includes the fix for ChromaDB Collection Re-creation ---
def initialize_state(story_seed, story_outline):
    """Initializes in-memory state, KG, and Vector DB."""
    global G, collection, client, collection_name, gemini_ef # Declare globals needed
    print("\n-> Initializing State, KG & Vector DB...")
    if not isinstance(story_seed, dict): return None, None, None, None
    if not isinstance(story_outline, dict): story_outline = {}

    # Reset state components
    character_db = {}
    lore_bible = {}
    episode_summaries = {}
    G = nx.DiGraph()

    # --- Cleanly handle ChromaDB Collection Re-creation ---
    try:
        # Try deleting first (might fail if it doesn't exist, which is fine)
        try:
            print(f"Attempting to delete existing ChromaDB collection '{collection_name}'...")
            client.delete_collection(name=collection_name)
            print(f"Collection '{collection_name}' deleted.")
        except Exception:
            # Catch error if collection didn't exist - no problem
            # print(f"Collection '{collection_name}' did not exist or couldn't be deleted (proceeding).")
            pass # Ignore deletion error

        # Now create/get the collection anew and assign to global var
        print(f"Creating/getting ChromaDB collection '{collection_name}'...")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=gemini_ef # Use the globally defined embedding function
        )
        print(f"ChromaDB collection '{collection_name}' ready.")

    except Exception as e:
        print(f"CRITICAL ERROR initializing ChromaDB collection '{collection_name}': {e}")
        print("Cannot proceed without a valid collection.")
        raise # Re-raise the exception to stop the script

    # --- Populate character_db, KG nodes/attributes, and Vector DB ---
    print("Initializing Characters...")
    for char_data in story_seed.get("characters", []):
        if isinstance(char_data, dict) and isinstance(char_data.get("name"), str) and char_data["name"]:
            name = char_data["name"]; desc = char_data.get("description", "N/A"); state = char_data.get("initial_state", "Unknown"); goal = char_data.get("arc_goal", "N/A")
            character_db[name] = {"description": desc, "current_state": state, "arc_goal": goal, "relationships": {}, "knowledge": []}
            G.add_node(name, type='character', description=desc, state=state, goal=goal, episode_introduced=0)
            initial_desc_vdb = f"Character Profile: {name}. Description: {desc}. Initial State: {state}. Goal: {goal}"
            add_to_vector_db(initial_desc_vdb, {"type": "character_profile", "character_name": name, "episode": 0}, f"char_{name}_ep0")
        else: print(f"Warning: Skipping invalid character data: {char_data}")

    # --- Populate lore_bible, KG nodes/attributes, and Vector DB ---
    print("Initializing Setting & Lore...")
    lore_bible = {"setting": story_seed.get("setting", {}), "conflict": story_seed.get("core_conflict", ""), "plot_hook": story_seed.get("initial_plot_hook", "")}
    setting_obj = lore_bible.get('setting', {}); primary_loc = setting_obj.get('primary'); secondary_loc = setting_obj.get('secondary')
    primary_loc_node = None; # Handle list/null for locations
    if isinstance(primary_loc, list) and primary_loc: primary_loc_node = str(primary_loc[0]); print(f"Warning: Primary location list, using: {primary_loc_node}")
    elif isinstance(primary_loc, str) and primary_loc: primary_loc_node = primary_loc
    secondary_loc_node = None;
    if isinstance(secondary_loc, list) and secondary_loc: secondary_loc_node = str(secondary_loc[0]); print(f"Warning: Secondary location list, using: {secondary_loc_node}")
    elif isinstance(secondary_loc, str) and secondary_loc: secondary_loc_node = secondary_loc
    if primary_loc_node: G.add_node(primary_loc_node, type='location', episode_introduced=0) # Add nodes to KG
    if secondary_loc_node: G.add_node(secondary_loc_node, type='location', episode_introduced=0)
    setting_desc_vdb = f"Setting Details: Primary - {primary_loc_node or 'N/A'}. Secondary - {secondary_loc_node or 'N/A'}."
    conflict_desc_vdb = f"Core Conflict Summary: {lore_bible.get('conflict', 'N/A')}"; hook_desc_vdb = f"Initial Plot Hook Summary: {lore_bible.get('plot_hook', 'N/A')}"
    add_to_vector_db(setting_desc_vdb, {"type": "setting", "episode": 0}, "setting_ep0"); add_to_vector_db(conflict_desc_vdb, {"type": "conflict", "episode": 0}, "conflict_ep0"); add_to_vector_db(hook_desc_vdb, {"type": "plot_hook", "episode": 0}, "hook_ep0")

    # --- Create plot_tracker ---
    plot_tracker = []; plot_tracker_episodes = story_outline.get("episodes", [])
    if isinstance(plot_tracker_episodes, list):
        for i, ep_data in enumerate(plot_tracker_episodes):
            if isinstance(ep_data, dict):
                 ep_data['episode_num'] = i + 1; new_ep_data = copy.deepcopy(ep_data); new_ep_data["status"] = "pending"; new_ep_data.setdefault('title', f'Chapter {i+1}'); new_ep_data.setdefault('summary_goal', 'Develop plot.'); new_ep_data.setdefault('key_beats', []); new_ep_data.setdefault('character_focus', []); plot_tracker.append(new_ep_data)
            else: print(f"Warning: Skipping invalid episode data for tracker: {ep_data}")
    else: print("Warning: 'episodes' key invalid in outline. Plot tracker empty.")

    # --- Final Print Statements for Initialization ---
    print(f"Initialized Characters: {list(character_db.keys())}")
    print(f"Initialized KG Nodes: {list(G.nodes())}")
    print(f"Plot Tracker Initialized with {len(plot_tracker)} chapters.")
    # Safely get count after ensuring collection is valid
    try:
        print(f"Initial Vector DB count: {collection.count()}")
    except Exception as e:
        print(f"Error getting collection count after initialization: {e}")

    return character_db, lore_bible, episode_summaries, plot_tracker


# --- Context Assembly (Uses KG & RAG) ---
def assemble_context(episode_num, outline_data, character_db, lore_bible, episode_summaries):
    """Assembles context using RAG retrieval and KG facts."""
    global G, collection # Access globals
    print(f"\n-> Assembling Context (KG+RAG) for Chapter {episode_num}...")
    if not isinstance(outline_data, list): print("Error: Invalid outline data."); return None
    episode_outline = next((ep for ep in outline_data if isinstance(ep, dict) and ep.get("episode_num") == episode_num), None)
    if not episode_outline: print(f"Error: Outline for chapter {episode_num} not found."); return None

    # --- 1. Core Context: Current Chapter Outline ---
    context = f"--- Context for Generating Chapter {episode_num} ---\n"; context += f"\n**Chapter Outline & Goal:**\n"; context += f"Title: {episode_outline.get('title', 'Untitled')}\n"; context += f"Goal: {episode_outline.get('summary_goal', 'N/A')}\n"; key_beats = episode_outline.get('key_beats', []); context += "Key Beats to Cover:\n"
    if key_beats: [context := context + f"- {beat}\n" for beat in key_beats]
    else: context += "- (No specific beats listed)\n"

    # --- 2. Current Character Structured States ---
    context += "\n**Current Character Information (Live State):**\n"; focused_chars_list = episode_outline.get('character_focus', [])
    chars_to_include = set(item.split(":")[0].strip() for item in focused_chars_list if isinstance(item, str) and item); included_any_char = False; focused_char_nodes = []
    for char_name, char_data in character_db.items():
        if char_name in chars_to_include:
            if isinstance(char_name, (str, int, float, tuple)): focused_char_nodes.append(char_name)
            else: print(f"Warning: Character name '{char_name}' not hashable, skipping."); continue
            included_any_char = True; context += f"  * {char_name}:\n"; context += f"    - Current State: {char_data.get('current_state', 'Unknown')}\n"; knowledge = char_data.get('knowledge', [])
            if knowledge: context += f"    - Latest Knowledge: {knowledge[-1]}\n"
    if not included_any_char and chars_to_include: context += f"(Warning: Focused characters {', '.join(chars_to_include)} not in DB)\n"

    # --- 3. KG: Retrieve Facts about Focused Characters ---
    context += "\n**Relevant Facts from Knowledge Graph:**\n"; kg_facts = []; processed_nodes = set()
    for node_name in focused_char_nodes:
        if node_name in processed_nodes: continue; processed_nodes.add(node_name)
        if G.has_node(node_name):
            attrs = G.nodes[node_name]; kg_facts.append(f"- {node_name} [Type: {attrs.get('type', '?')}, State: {attrs.get('state', '?')}]")
            for key, value in attrs.items():
                 if key.startswith('has_') and value: item_name = key[4:].replace('_',' '); kg_facts.append(f"  - Possesses: {item_name}")
            for neighbor, edge_data in G.adj[node_name].items():
                rel = edge_data.get('relationship', 'related_to'); ep_formed = edge_data.get('episode_formed'); rel_detail = f" (ep {ep_formed})" if ep_formed is not None else ""
                kg_facts.append(f"  - {node_name} --[{rel}{rel_detail}]--> {neighbor}")
        else: kg_facts.append(f"- ('{node_name}' not in KG)")
    if kg_facts: context += "\n".join(kg_facts) + "\n"; 
    else: context += "(No specific KG facts retrieved)\n"

    # --- 4. RAG: Retrieve Relevant History/Lore ---
    context += "\n**Relevant History & Context (Retrieved via RAG):**\n"; query_text = f"Ch {episode_num} '{episode_outline.get('title', '')}'. Goal: {episode_outline.get('summary_goal', '')}. Chars: {', '.join(focused_char_nodes)}. Beats: {', '.join(key_beats)}"
    retrieved_context = "";
    try:
        if collection.count() > 0: # Only query if collection has items
             results = collection.query(query_texts=[query_text], n_results=6, include=['documents', 'metadatas', 'distances'])
             if results and results.get('documents') and results['documents'][0]:
                 retrieved_items = [];
                 for i, doc in enumerate(results['documents'][0]):
                     meta = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}; dist = results['distances'][0][i] if results.get('distances') and results['distances'][0] else None
                     item_type = meta.get('type', 'unknown'); item_ep = meta.get('episode', '?')
                     if item_type == "summary" and item_ep == episode_num - 1: continue # Skip immediate previous summary
                     if dist is not None and dist > 0.8: continue # Relevance filter
                     prefix = f"- [From Ep {item_ep}, Type: {item_type}, Rel: {1-dist:.2f}]: " if dist is not None else f"- [From Ep {item_ep}, Type: {item_type}]: "
                     retrieved_items.append(prefix + doc)
                 if retrieved_items: retrieved_context = "\n".join(retrieved_items) + "\n"; print(f"Retrieved {len(retrieved_items)} items via RAG.")
                 else: retrieved_context = "(No highly relevant items found via RAG query)\n"
             else: retrieved_context = "(RAG retrieval returned no results)\n"
        else: retrieved_context = "(Vector DB is empty, skipping RAG query)\n"
    except Exception as e: print(f"Error during RAG retrieval: {e}"); retrieved_context = "(Error during RAG retrieval)\n"
    context += retrieved_context

    # --- 5. Add Previous Chapter Summary (Directly) ---
    context += "\n**Direct Summary of Previous Chapter:**\n"; prev_ep_num = episode_num - 1
    if prev_ep_num > 0 and prev_ep_num in episode_summaries: context += f"Summary of Chapter {prev_ep_num}: {episode_summaries[prev_ep_num]}\n"
    elif episode_num == 1: context += "This is the first chapter.\n";
    else: context += f"(No direct summary available for Chapter {prev_ep_num})\n"

    print(f"Assembled Context Length: ~{len(context)} chars"); return context

# --- Chapter Generation (Novel Format) ---
def generate_chapter_content(episode_num, context):
    """Generates narrative prose for a story chapter using Gemini."""
    print(f"\n-> Generating Chapter {episode_num} Content...")
    if context is None or context.startswith("ERROR:") or not isinstance(context, str): return None
    prompt = f"""You are a novelist writing Chapter {episode_num}. Use the provided CONTEXT (outline, current states, KG facts, RAG history, previous summary). Write in engaging, descriptive narrative prose (PAST TENSE, THIRD-PERSON LIMITED focusing on characters in 'Current Character Information'). Follow outline beats. Incorporate setting, actions, dialogue ("Quote," attribution.). Be consistent with ALL context. Show, don't tell. Write the actual scene. --- CONTEXT ---\n{context}\n--- END CONTEXT ---\nBegin writing Chapter {episode_num} now:"""
    content = call_llm(prompt, model_name="gemini-1.5-pro-latest"); # Use Pro for better narrative
    if content and content.startswith("ERROR:"): return None
    elif not content: print("Warning: Chapter generation returned empty."); return None
    return content

# --- State Update (Updates KG & Vector DB) ---
def update_state_and_summarize(episode_num, generated_content, character_db, plot_tracker, episode_summaries):
    """Updates state, summarizes, adds summary to Vector DB, and updates KG based on chapter content."""
    global G, collection; print(f"\n-> Updating State, Summarizing, Indexing & Updating KG for Chapter {episode_num}...")
    content_available = generated_content and not generated_content.startswith("ERROR:")
    episode_updated_in_tracker = False # Mark status in tracker
    for ep in plot_tracker:
        if isinstance(ep, dict) and ep.get("episode_num") == episode_num: ep["status"] = "generated" if content_available else "failed_generation"; episode_updated_in_tracker = True; break
    if not episode_updated_in_tracker: print(f"Warning: Could not update status for Chapter {episode_num} in tracker.")
    if not content_available: print("Skipping updates due to chapter generation failure."); return
    summary = "Summary generation failed."; # Summarization & Add to Vector DB
    try:
        max_summary_content_len = 8000; content_for_summary = generated_content
        if len(content_for_summary) > max_summary_content_len: content_for_summary = generated_content[:4000] + "\n...\n" + generated_content[-4000:]
        summary_prompt = f"""Read the chapter excerpt. Summarize key plot advancements & character changes in 2-4 concise sentences for the NEXT chapter's context. Excerpt: ```\n{content_for_summary}\n``` Generate only the summary text."""
        summary_response = call_llm(summary_prompt, model_name="gemini-1.5-flash-latest")
        if summary_response and not summary_response.startswith("ERROR:"):
            summary = summary_response.strip(); print(f"Generated Summary: {summary}"); add_to_vector_db(summary, {"type": "summary", "episode": episode_num}, f"summary_ep{episode_num}")
        else: print(f"Warning: Summary generation failed. LLM Response: {summary_response}")
    except Exception as e: print(f"Error during summary/indexing: {e}")
    episode_summaries[episode_num] = summary
    print("Attempting state and KG update analysis..."); # State & KG Update (LLM Analysis)
    state_updates_applied = False; kg_updates_applied = False
    try:
        max_state_content_len = 8000; content_for_state = generated_content
        if len(content_for_state) > max_state_content_len: content_for_state = generated_content[:4000] + "\n...\n" + generated_content[-4000:]
        update_analysis_prompt = f"""Analyze the chapter excerpt for Chapter {episode_num} and current states. Identify significant changes. Output JSON: Keys are character names. Values are objects with optional keys: "updated_state"(string), "new_knowledge"(list[str]), "new_relationships"(list[list[str, str, str]]), "gained_items"(list[str]). Focus ONLY on explicit events. Current States: {json.dumps(character_db, indent=2)}\nChapter Excerpt: ```\n{content_for_state}\n```\nGenerate ONLY the JSON object. If no changes, output {{}}. No ```json markdown."""
        state_update_suggestion_json = call_llm(update_analysis_prompt, model_name="gemini-1.5-flash-latest")
        if state_update_suggestion_json and not state_update_suggestion_json.startswith("ERROR:"):
            cleaned_response = state_update_suggestion_json.strip();
            if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-3]
            suggested_updates = {} if not cleaned_response else json.loads(cleaned_response)
            if isinstance(suggested_updates, dict) and suggested_updates:
                print("Applying suggested state & KG updates:")
                for char_name, updates in suggested_updates.items():
                    if isinstance(char_name, str) and char_name in character_db and G.has_node(char_name):
                        print(f"- Updating {char_name}:") # Update Character DB
                        if updates.get("updated_state") and isinstance(updates["updated_state"], str): character_db[char_name]["current_state"] = updates["updated_state"]; print(f"  DB State -> {updates['updated_state']}"); state_updates_applied = True
                        if updates.get("new_knowledge") and isinstance(updates["new_knowledge"], list):
                             current_knowledge = character_db[char_name].get("knowledge", []); new_items = [k for k in updates["new_knowledge"] if isinstance(k, str) and k and k not in current_knowledge]
                             if new_items: character_db[char_name].setdefault("knowledge", []).extend(new_items); print(f"  DB Knowledge Added: {new_items}"); state_updates_applied = True
                        # Update Knowledge Graph
                        if updates.get("updated_state") and isinstance(updates["updated_state"], str): G.nodes[char_name]['state'] = updates["updated_state"]; print(f"  KG Attr 'state' -> {updates['updated_state']}"); kg_updates_applied = True
                        if updates.get("new_relationships") and isinstance(updates["new_relationships"], list):
                            for rel in updates["new_relationships"]:
                                if isinstance(rel, list) and len(rel) == 3:
                                    subj, pred, obj = str(rel[0]), str(rel[1]), str(rel[2])
                                    if G.has_node(subj) and G.has_node(obj): G.add_edge(subj, obj, relationship=pred, episode_formed=episode_num); print(f"  KG Edge Added: {subj} -[{pred}]-> {obj}"); kg_updates_applied = True
                                    else: print(f"  KG Edge Skipped: Node missing for {subj} or {obj}")
                        if updates.get("gained_items") and isinstance(updates["gained_items"], list):
                             for item in updates["gained_items"]:
                                 if isinstance(item, str) and item:
                                     item_node_name = item;
                                     if not G.has_node(item_node_name): G.add_node(item_node_name, type='item')
                                     G.add_edge(char_name, item_node_name, relationship='has', episode_gained=episode_num); print(f"  KG Edge Added: {char_name} -[has]-> {item_node_name}"); attr_name = 'has_' + re.sub(r'\W+', '_', item.lower()); G.nodes[char_name][attr_name] = True; kg_updates_applied = True
                    else: print(f"  Warning: Character '{char_name}' not in DB/KG or name invalid.")
                if not state_updates_applied and not kg_updates_applied: print("LLM suggested updates, but none were applicable or valid.")
            else: print("No significant state/KG changes detected by LLM.")
        else: print(f"State/KG update analysis failed. LLM Response: {state_update_suggestion_json}")
    except (json.JSONDecodeError, AttributeError) as e: print(f"Error decoding state/KG update JSON: {e}\nLLM Response: {state_update_suggestion_json}")
    except Exception as e: print(f"Error during state/KG update process: {e}")
    if not state_updates_applied: print("Character DB state update via LLM analysis did not result in changes.")
    if not kg_updates_applied: print("Knowledge Graph update via LLM analysis did not result in changes.")
    print("-" * 20)

# ==============================================================================
# Cell 5: Pipeline Runner Function (Interactive KG+RAG + Novel + gTTS + DALL-E)
# ==============================================================================

def run_interactive_pipeline(input_brief):
    """Runs the interactive storytelling pipeline with KG, RAG, Novel Format, TTS, Cover."""
    global G, collection # Allow access/reset
    print("--- Starting AI Storytelling Pipeline (Interactive + KG + RAG + Novel + gTTS + DALL-E) ---")
    print(f"Input Concept: {input_brief}"); print("-" * 50)
    G = nx.DiGraph(); # Reset KG # Collection cleared in initialize_state

    # Initialize return values
    generated_chapters = {}; confirmed_outline = None; final_state = None
    audio_generated = False; cover_generated = False

    # 1. Expand Concept
    story_seed = expand_concept(input_brief)
    if not story_seed: print("Halted: Concept Expansion failed."); return None, None, None, False, False
    try:
        with open("story_seed.json", "w", encoding='utf-8') as f: json.dump(story_seed, f, indent=2); print("Saved story_seed.json")
    except Exception as e: print(f"Error saving story_seed.json: {e}")
    time.sleep(1)

    # 2. Generate Initial Outline Proposal
    initial_outline = generate_outline(story_seed)
    if not initial_outline: print("Halted: Initial Outline Generation failed."); return None, story_seed, None, False, False

    # 3. Review and Confirm Outline
    confirmed_outline = review_and_confirm_outline(story_seed, initial_outline)
    if not confirmed_outline: print("Halted: Outline rejected."); return None, story_seed, None, False, False
    try:
        with open("story_outline_confirmed.json", "w", encoding='utf-8') as f: json.dump(confirmed_outline, f, indent=2); print("Saved confirmed outline.")
    except Exception as e: print(f"Error saving confirmed outline: {e}")

    # 4. Initialize State (populates KG & Vector DB)
    # Ensure this function returns 4 values now
    character_db, lore_bible, episode_summaries, plot_tracker = initialize_state(story_seed, confirmed_outline)
    if plot_tracker is None: print("Halted: State Initialization failed."); return None, confirmed_outline, None, False, False

    # 5. Determine Chapters to Process
    if not plot_tracker: print("Error: Plot tracker empty."); return None, confirmed_outline, character_db, False, False
    chapter_numbers_to_process = sorted([ep.get("episode_num") for ep in plot_tracker if isinstance(ep, dict) and "episode_num" in ep])
    if not chapter_numbers_to_process: print("Error: No valid chapter numbers in tracker."); return None, confirmed_outline, character_db, False, False
    print(f"\nProceeding to generate {len(chapter_numbers_to_process)} chapters: {chapter_numbers_to_process}"); print("-" * 50)

    # 6. Loop through chapters
    for ch_num in chapter_numbers_to_process:
        print(f"\n=== Processing Chapter {ch_num} ===")
        current_plot_tracker_state = [ep for ep in plot_tracker if isinstance(ep, dict)]
        context = assemble_context(ch_num, current_plot_tracker_state, character_db, lore_bible, episode_summaries)
        if context is None: print(f"Error: Failed context assembly for Chapter {ch_num}. Skipping."); generated_chapters[ch_num] = f"ERROR: Context assembly failed."; continue
        chapter_content = generate_chapter_content(ch_num, context)
        chapter_filename = f"chapter_{ch_num}_content.txt"
        content_to_save = chapter_content if chapter_content else f"ERROR: Chapter generation failed for Chapter {ch_num}."
        try:
            with open(chapter_filename, "w", encoding='utf-8') as f: f.write(content_to_save); print(f"Saved {chapter_filename}")
            generated_chapters[ch_num] = chapter_content if chapter_content else f"ERROR: Chapter generation failed."
        except Exception as e: print(f"Error saving chapter {chapter_filename}: {e}"); generated_chapters[ch_num] = f"ERROR: Chapter saving failed - {e}"
        update_state_and_summarize(ch_num, chapter_content, character_db, plot_tracker, episode_summaries)
        print("Waiting briefly..."); time.sleep(4)

    # --- Pipeline Finished ---
    print("\n--- Story Generation Pipeline Finished ---")
    final_state = {"character_db": character_db, "lore_bible": lore_bible, "episode_summaries": episode_summaries, "plot_tracker": plot_tracker}
    try:
        with open("final_pipeline_state.json", "w", encoding='utf-8') as f: json.dump(final_state, f, indent=2); print("Saved final_pipeline_state.json")
    except Exception as e: print(f"Error saving final_pipeline_state.json: {e}")
    try:
        kg_filename = "final_knowledge_graph.json"; kg_data = nx.node_link_data(G)
        with open(kg_filename, "w", encoding='utf-8') as f: json.dump(kg_data, f, indent=2); print(f"Saved final KG ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    except Exception as e: print(f"Error saving final KG: {e}")
    try: print(f"Final Vector DB Count: {collection.count()}")
    except Exception as e: print(f"Could not get final vector DB count: {e}")

    # --- Post-Generation Options ---
    full_story_text = "" # Initialize before potentially using
    if generated_chapters:
        print("\nCombining chapters for potential audio generation...")
        combined_texts = []
        # Use confirmed_outline here as final_outline_used is not defined in this scope yet
        outline_to_use_for_titles = confirmed_outline if confirmed_outline else {}
        episodes_list_for_titles = outline_to_use_for_titles.get("episodes", [])

        for i in sorted(generated_chapters.keys()):
            title = f"Chapter {i}"
            if isinstance(episodes_list_for_titles, list):
                 ep_detail = next((ep for ep in episodes_list_for_titles if isinstance(ep,dict) and ep.get("episode_num") == i), None)
                 if ep_detail and ep_detail.get("title"): title = f"Chapter {i}: {ep_detail['title']}"

            chapter_content = generated_chapters.get(i) # Use .get for safety
            if chapter_content and not str(chapter_content).startswith("ERROR:"): # Check type just in case
                 combined_texts.append(f"{title}\n\n{chapter_content}")
            else:
                 combined_texts.append(f"\n\n[Content for Chapter {i} could not be generated.]\n\n")
        full_story_text = "\n\n".join(combined_texts).strip()

        # Ask User about Audio Generation (using gTTS function)
        if full_story_text:
            while True:
                try:
                     generate_audio_q = input("Generate audio file for the story (using gTTS)? (yes/no): ").strip().lower()
                except EOFError: generate_audio_q = 'no'; print("Input ended, skipping audio.") # Handle EOF
                if generate_audio_q == 'yes':
                    audio_filename = "generated_story_gtts.mp3"
                    audio_generated = generate_audio_gtts(full_story_text, audio_filename)
                    break
                elif generate_audio_q == 'no': print("Skipping audio generation."); break
                else: print("Please enter 'yes' or 'no'.")

        # Ask User about Cover Generation (using DALL-E function)
        if openai.api_key and story_seed:
             while True:
                try:
                     generate_cover_q = input("Generate a book cover image (using DALL-E 3)? (yes/no): ").strip().lower()
                except EOFError: generate_cover_q = 'no'; print("Input ended, skipping cover.") # Handle EOF
                if generate_cover_q == 'yes':
                    prompt_gen_prompt = f"""Based on the following story seed, generate a concise and evocative DALL-E 3 prompt (max 150 words) for a book cover image. Focus on genre, mood, key characters/elements, and visual style. Story Seed: {json.dumps(story_seed)}"""
                    img_prompt_text = call_llm(prompt_gen_prompt, model_name="gemini-1.5-flash-latest")
                    if not img_prompt_text or img_prompt_text.startswith("ERROR:"):
                        print("Failed to generate image prompt using LLM, using basic prompt.")
                        img_prompt_text = f"Book cover art for a {story_seed.get('genre', 'story')} titled '{story_seed.get('title_concept', 'Untitled')}' about {story_seed.get('logline', 'a story')}. Style: digital painting."
                    print(f"Using image prompt: {img_prompt_text[:200]}...")
                    cover_filename = "generated_book_cover.png"
                    cover_generated = generate_cover_image(img_prompt_text, cover_filename)
                    break
                elif generate_cover_q == 'no': print("Skipping cover generation."); break
                else: print("Please enter 'yes' or 'no'.")

    # Return confirmed outline, not initial
    return generated_chapters, confirmed_outline, final_state, audio_generated, cover_generated


# ==============================================================================
# Cell 6: Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" AI Storyteller Pipeline (Interactive + KG + RAG + Novel + gTTS) ".center(60, "="))
    print("="*60)
    initial_concept = ""
    try:
        print("\nPlease enter the story concept note below.")
        initial_concept = input("Enter your story concept here: ")
    except EOFError: # Handle cases where input might be piped or interrupted
        print("\nNo input received. Exiting.")
        exit()
    except KeyboardInterrupt:
         print("\nExecution interrupted by user. Exiting.")
         exit()

    if not initial_concept or initial_concept.isspace():
        print("\nNo valid concept entered. Halting.")
    else:
        print(f"\nReceived concept: \"{initial_concept}\"")
        print("-" * 30)

        # --- Run the Interactive Pipeline ---
        final_chapters = None # Initialize to None
        final_outline_used = None
        audio_success = False
        cover_success = False
        try:
             # Capture all return values, even if None
             pipeline_results = run_interactive_pipeline(initial_concept)
             if pipeline_results: # Check if pipeline returned anything
                  final_chapters, final_outline_used, final_state_data, audio_success, cover_success = pipeline_results
             else: # Pipeline likely halted early
                  print("Pipeline did not complete successfully.")

             # --- Final Output Display ---
             print("\n" + "="*20 + " FINAL STORY OUTPUT " + "="*20)
             # Check if chapters were actually generated
             if final_chapters:
                 # Check if outline data is usable for titles
                 if final_outline_used and isinstance(final_outline_used.get("episodes"), list):
                     print("\nDisplaying generated chapter content (truncated):")
                     print("-" * 50)
                     outline_map = {ep.get("episode_num"): ep for ep in final_outline_used["episodes"] if isinstance(ep, dict)}
                     for ch_num in sorted(final_chapters.keys()):
                         chapter_content = final_chapters[ch_num]
                         details = outline_map.get(ch_num)
                         title = details.get("title", f"Chapter {ch_num}") if details else f"Chapter {ch_num}"
                         print(f"\n--- {title.upper()} ---")
                         print("-" * len(title))
                         if chapter_content and not str(chapter_content).startswith("ERROR:"):
                              lines = chapter_content.splitlines()
                              preview_len = 10
                              if len(lines) > (preview_len * 2 + 2):
                                   print("\n".join(lines[:preview_len])); print("\n[... chapter content truncated ...]\n"); print("\n".join(lines[-preview_len:]))
                              else: print(chapter_content)
                              print(f"\n(Full content saved in: chapter_{ch_num}_content.txt)")
                         else: print(f"\n!!! CHAPTER GENERATION FAILED for Chapter {ch_num} !!!\n(Reason: {chapter_content})")
                         print("-" * 50)
                 else: # Chapters exist but outline is bad/missing
                      print("\nGenerated Chapter Files (outline data missing for titles/display):")
                      for ch_num, content in sorted(final_chapters.items()):
                          status = "Error/Failed" if str(content).startswith("ERROR:") else "Generated"
                          print(f"- chapter_{ch_num}_content.txt ({status})")

                 # Always show status of additional files if chapters were attempted
                 print("\n--- Additional Files Status ---")
                 print(f"- Audio (gTTS): {'generated_story_gtts.mp3' if audio_success else 'Skipped or Failed'}")
                 print(f"- Cover Image (DALL-E): {'generated_book_cover.png' if cover_success else 'Skipped or Failed'}")

             else: # Pipeline ran but generated no chapters
                 print("\nPipeline did not complete chapter generation.")
                 print("Check error messages and file browser for partial outputs (like JSON files).")

        except KeyboardInterrupt:
             print("\nExecution interrupted by user during pipeline run.")
        except Exception as main_e: # Catch any other unexpected errors during the run
             print("\n" + "="*60)
             print(" AN UNEXPECTED ERROR OCCURRED DURING PIPELINE EXECUTION ".center(60, "!"))
             print(f"Error Type: {type(main_e).__name__}")
             print(f"Error Details: {main_e}")
             import traceback
             print("\n--- Traceback ---")
             traceback.print_exc() # Print detailed traceback for debugging
             print("="*60)
             print("Pipeline execution halted prematurely.")

    # --- Final message ---
    print("\n" + "="*60)
    print(" Script Finished ".center(60,"="))
    print("="*60)