# ==================================================
# app.py - Streamlit Frontend for KUKUTales
# ==================================================
import streamlit as st
import json
import time
import os
import traceback # For better error reporting
import base64
import openai
import networkx as nx # Needed if displaying KG info

# --- Import functions & Initialize Components ---
# ASSUMES pipeline_logic.py has been refactored (no input()/exit())
# and sets up necessary global variables like G, client, collection upon import,
# AND that initialize_state resets them correctly on each run.
# This reliance on globals is NOT ideal but adapts the previous structure.
try:
    from pipeline_logic import (
        # Core pipeline functions
        expand_concept, generate_outline, initialize_state, assemble_context,
        generate_chapter_content, update_state_and_summarize,
        # Helper/Utility functions
        call_llm, generate_audio_gtts, generate_cover_image,
        # Needed if initialize/update calls it directly & not encapsulated
        add_to_vector_db
        # Globals we might need to access if not passed around (use with caution)
        # G, client as chroma_client, collection as chroma_collection # Better to pass state
    )
    # Import the specific formatter helper if it's separate
    try:
        from pipeline_logic import format_outline_for_display
    except ImportError:
        # Basic fallback formatter if not found in pipeline_logic
        def format_outline_for_display(outline):
            if not outline or not isinstance(outline.get("episodes"), list): return "Error: Invalid outline."
            text = f"**Overall Arc:** {outline.get('overall_arc', 'N/A')}\n\n**Chapters:**\n"
            for ep in outline.get("episodes", []):
                 text += f"### Ch {ep.get('episode_num', '?')}: {ep.get('title', 'Untitled')}\n"
                 text += f"*   Goal: {ep.get('summary_goal', 'N/A')}\n"
                 text += f"*   Focus: {', '.join(ep.get('character_focus', ['N/A']))}\n---\n"
            return text
    print("Successfully imported from pipeline_logic.")

except ImportError as e:
    st.error(f"Failed import from pipeline_logic.py: {e}. Ensure file exists and is correct."); st.stop()
except Exception as e:
    st.error(f"Error during import or initial setup: {e}"); st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="KUKUTales Storyteller")

# --- Initialize Session State ---
# Stages: input -> generating_outline -> review_outline -> regenerating_outline -> initializing -> generating_chapters -> finished
if 'stage' not in st.session_state: st.session_state.stage = 'input'
if 'concept' not in st.session_state: st.session_state.concept = ""
if 'story_seed' not in st.session_state: st.session_state.story_seed = None
if 'outline_proposal' not in st.session_state: st.session_state.outline_proposal = None
if 'confirmed_outline' not in st.session_state: st.session_state.confirmed_outline = None
if 'final_chapters' not in st.session_state: st.session_state.final_chapters = {}
if 'full_story_text' not in st.session_state: st.session_state.full_story_text = ""
# State Management Objects - results from initialize_state
if 'character_db' not in st.session_state: st.session_state.character_db = None
if 'lore_bible' not in st.session_state: st.session_state.lore_bible = None
if 'episode_summaries' not in st.session_state: st.session_state.episode_summaries = {}
if 'plot_tracker' not in st.session_state: st.session_state.plot_tracker = None
# We won't store the whole KG/Collection in session state, assume logic file handles it via globals/reinit
# UI / Status Flags
if 'error_message' not in st.session_state: st.session_state.error_message = None
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'audio_file' not in st.session_state: st.session_state.audio_file = None # Store filename
if 'cover_file' not in st.session_state: st.session_state.cover_file = None # Store filename
if 'current_chapter_generating' not in st.session_state: st.session_state.current_chapter_generating = None
if 'chapters_generated_count' not in st.session_state: st.session_state.chapters_generated_count = 0

# --- Helper Functions ---
def add_log(message):
    """Adds a timestamped message to the log in session state."""
    st.session_state.log_messages.append(f"{time.strftime('%H:%M:%S')}: {message}")
    max_log_lines = 50; st.session_state.log_messages = st.session_state.log_messages[-max_log_lines:]

def reset_state_for_new_story():
    """Resets session state for a new story."""
    add_log("Resetting state for new story...")
    keys_to_reset = ['stage', 'concept', 'story_seed', 'outline_proposal', 'confirmed_outline',
                     'final_chapters', 'error_message', 'log_messages', 'full_story_text',
                     'audio_file', 'cover_file', 'character_db', 'lore_bible',
                     'episode_summaries', 'plot_tracker', #'knowledge_graph', #'chroma_collection', # Don't delete KG/Collection handles if logic file manages them
                     'current_chapter_generating', 'chapters_generated_count']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize stage
    st.session_state.stage = 'input'
    # Optionally explicitly call a reset function in pipeline_logic if needed
    # e.g., pipeline_logic.reset_globals()

# ==================================================
# UI Rendering Logic
# ==================================================

# --- Logo and Title ---
# Create columns for layout
top_col1, top_col2 = st.columns([1, 4]) # Adjust ratio as needed
with top_col1:
    try:
        st.image("kukutales_logo.png", width=150) # Add your logo file path/name
    except Exception:
        st.write("KUKUTales") # Fallback text if logo fails
with top_col2:
    st.title("AI Storyteller")
    st.caption("Generate multi-chapter stories from a concept, with interactive outlining, memory, audio, and cover art.")

st.divider()

# --- Display Error Messages ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    if st.button("Try Again From Start"):
        reset_state_for_new_story(); st.rerun()
    st.stop() # Stop rendering further if there's a blocking error

# === Input Stage ===
if st.session_state.stage == 'input':
    st.header("1. Plant Your Story Seed")
    concept = st.text_area(
        "Enter your story concept:", height=100, key="concept_widget",
        value=st.session_state.concept, # Preserve input
        placeholder="E.g., A steampunk detective investigates a series of impossible clockwork crimes..."
    )
    st.session_state.concept = concept # Update state

    if st.button("🌱 Generate Outline Proposal", type="primary", disabled=(not concept)):
        st.session_state.error_message = None; st.session_state.log_messages = []
        add_log(f"Received concept: {concept}"); add_log("Starting pipeline...")
        st.session_state.stage = 'generating_outline'; st.rerun()

# === Generating Outline Stage ===
elif st.session_state.stage == 'generating_outline':
    st.header("Processing...")
    with st.spinner("🧠 Expanding concept & drafting outline..."):
        add_log("Expanding concept...");
        try:
            # Use concept stored in session state
            concept_to_use = st.session_state.get('concept', '')
            if not concept_to_use: raise ValueError("Concept is empty.")

            story_seed = expand_concept(concept_to_use)
            if story_seed:
                st.session_state.story_seed = story_seed; add_log("Concept expanded.")
                add_log("Generating outline...");
                outline = generate_outline(story_seed)
                if outline:
                    st.session_state.outline_proposal = outline
                    add_log(f"Outline generated ({len(outline.get('episodes',[]))} chapters).")
                    st.session_state.stage = 'review_outline'
                else: raise ValueError("Outline generation failed (returned None or empty).")
            else: raise ValueError("Concept expansion failed (returned None or empty).")
        except Exception as e:
            st.session_state.error_message = f"Error during outline gen: {e}"; add_log(f"ERROR: {e}")
            st.session_state.stage = 'input'
    st.rerun()

# === Review Outline Stage ===
elif st.session_state.stage == 'review_outline':
    st.header("2. Review the Blueprint")
    st.info("Confirm the proposed story arc and chapter goals, or suggest revisions.")
    outline = st.session_state.outline_proposal
    if outline:
        display_md = format_outline_for_display(outline)
        st.markdown(display_md, unsafe_allow_html=True)
        st.divider()
        st.write("**Your Action:**")
        # Use columns for better layout
        col_feedback, col_actions = st.columns([3,1])
        with col_feedback:
            feedback = st.text_area("Revision Feedback (optional):", key="feedback_widget", height=100)
        with col_actions:
            accept_btn = st.button("✅ Accept & Write", type="primary", use_container_width=True)
            revise_btn = st.button("🔄 Revise Outline", disabled=(not feedback), use_container_width=True)
            reject_btn = st.button("❌ Discard & Start Over", use_container_width=True)

        if accept_btn:
            add_log("Outline accepted."); st.session_state.confirmed_outline = outline
            st.session_state.stage = 'initializing'; st.session_state.error_message = None; st.rerun()
        if reject_btn:
            add_log("Outline rejected."); st.warning("Outline rejected. Starting over.")
            reset_state_for_new_story(); st.rerun()
        if revise_btn: # Button only enabled if feedback has text
            add_log(f"User feedback: {feedback}"); st.session_state.stage = 'regenerating_outline'; st.session_state.error_message = None; st.rerun()
    else:
        st.error("Outline data missing. Please start over."); reset_state_for_new_story(); st.rerun()

# === Regenerating Outline Stage ===
elif st.session_state.stage == 'regenerating_outline':
    st.header("Processing Feedback...")
    with st.spinner("🛠️ Incorporating feedback and revising outline..."):
        feedback = st.session_state.get("feedback_widget", "")
        seed = st.session_state.story_seed
        previous_outline = st.session_state.outline_proposal
        if not feedback or not seed or not previous_outline:
            st.error("Missing data for regeneration. Returning to review.")
            st.session_state.stage = 'review_outline'; st.rerun()

        add_log("Attempting regeneration...")
        try:
            regen_prompt = f"""Revise story outline based on feedback. Seed: {json.dumps(seed)}. Prev Outline: {json.dumps(previous_outline)}. Feedback: "{feedback}". Generate NEW JSON outline (same structure). Only JSON output. Ensure 'episodes' not empty."""
            response = call_llm(regen_prompt, model_name="gemini-1.5-flash-latest")
            if response and not response.startswith("ERROR:"):
                cleaned = response.strip(); # Basic cleaning
                if cleaned.startswith("```json"): cleaned = cleaned[7:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                new_outline = json.loads(cleaned)
                # Validation
                episodes = new_outline.get("episodes")
                if isinstance(episodes, list) and episodes:
                    validated_eps = []
                    for i, ep in enumerate(episodes):
                         if isinstance(ep, dict): ep['episode_num'] = i+1; validated_eps.append(ep)
                    new_outline["episodes"] = validated_eps
                    if new_outline["episodes"]:
                        st.session_state.outline_proposal = new_outline # Update proposal
                        add_log("Outline regenerated."); st.session_state.stage = 'review_outline'
                    else: raise ValueError("Regen outline has no valid episodes.")
                else: raise ValueError("Regen outline invalid structure.")
            else: raise ValueError(f"LLM call failed for regen: {response}")
        except Exception as e:
            st.session_state.error_message = f"Regeneration failed: {e}"; add_log(f"ERROR: {e}")
            st.session_state.stage = 'review_outline' # Go back to previous valid outline on error
    st.rerun()

# === Initializing Stage ===
elif st.session_state.stage == 'initializing':
    st.header("Setting the Stage...")
    with st.spinner("📚 Preparing story state, knowledge graph, and vector memory..."):
        add_log("Initializing state...")
        try:
            # Assumes initialize_state resets KG/Collection via globals or internal logic
            results = initialize_state(st.session_state.story_seed, st.session_state.confirmed_outline)
            if results:
                # Store the returned state components
                st.session_state.character_db, st.session_state.lore_bible, \
                st.session_state.episode_summaries, st.session_state.plot_tracker = results[:4]
                # We don't store KG/Collection in session state, assume logic file manages them
                if not st.session_state.plot_tracker: raise ValueError("Init failed: No plot tracker.")
                add_log("State initialized."); st.session_state.stage = 'generating_chapters'
                # Clear previous run outputs
                st.session_state.final_chapters = {}; st.session_state.full_story_text = ""; st.session_state.audio_file = None; st.session_state.cover_file = None
                st.session_state.chapters_generated_count = 0
            else: raise ValueError("initialize_state returned None.")
        except Exception as e:
            st.session_state.error_message = f"Initialization Error: {e}"; add_log(f"ERROR: {e}"); st.session_state.stage = 'input'
    st.rerun()

# === Generating Chapters Stage ===
elif st.session_state.stage == 'generating_chapters':
    st.header("3. Writing the Story...")

    # Retrieve necessary state components from session_state
    plot_tracker = st.session_state.plot_tracker
    character_db = st.session_state.character_db
    lore_bible = st.session_state.lore_bible
    episode_summaries = st.session_state.episode_summaries
    # Note: We are NOT passing G or collection here, assuming pipeline_logic handles them globally

    if not plot_tracker or character_db is None or lore_bible is None or episode_summaries is None:
         st.error("Critical state missing. Cannot generate chapters."); reset_state_for_new_story(); st.rerun()

    chapters_to_process = sorted([
         ep.get("episode_num") for ep in plot_tracker if isinstance(ep, dict) and ep.get("status") == "pending"
    ])

    if not chapters_to_process:
         add_log("All chapters already processed."); st.session_state.stage = 'finished'; st.rerun()

    # Get the next chapter number
    ch_num = chapters_to_process[0]
    st.session_state.current_chapter_generating = ch_num
    total_chapters = len(plot_tracker)

    # Display progress and logs
    st.progress((st.session_state.chapters_generated_count) / total_chapters, text=f"Working on Chapter {ch_num}/{total_chapters}...")
    log_placeholder = st.empty() # For dynamic log updates
    log_placeholder.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key=f"log_area_{ch_num}")

    # Run generation for ONE chapter per rerun
    try:
        add_log(f"=== Processing Chapter {ch_num} ==="); log_placeholder.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key=f"log_upd1_{ch_num}")
        add_log(f"  Assembling context..."); log_placeholder.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key=f"log_upd2_{ch_num}")
        # Call assemble_context (assumes it accesses global G/collection if needed)
        context = assemble_context(ch_num, plot_tracker, character_db, lore_bible, episode_summaries)

        if context:
            add_log(f"  Generating content..."); log_placeholder.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key=f"log_upd3_{ch_num}")
            content = generate_chapter_content(ch_num, context)
            st.session_state.final_chapters[ch_num] = content if content else f"ERROR: Chapter {ch_num} generation failed."
            if not content or str(content).startswith("ERROR:"): add_log(f"  Error: Chapter {ch_num} generation failed.")
            else: add_log(f"  Chapter {ch_num} content generated.")

            add_log(f"  Updating state..."); log_placeholder.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key=f"log_upd4_{ch_num}")
            # Call update_state (assumes it accesses global G/collection if needed and modifies state dicts/lists in place)
            update_state_and_summarize(ch_num, content, character_db, plot_tracker, episode_summaries)
            add_log(f"  State updated for Chapter {ch_num}.")
            st.session_state.chapters_generated_count += 1
        else:
            add_log(f"  Error: Context assembly failed. Skipping generation."); log_placeholder.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key=f"log_upd5_{ch_num}")
            st.session_state.final_chapters[ch_num] = f"ERROR: Context assembly failed for Chapter {ch_num}."
            for ep in plot_tracker: # Mark as failed
                if isinstance(ep, dict) and ep.get("episode_num") == ch_num: ep["status"] = "failed_context"; break

        # Trigger next iteration or finish
        chapters_remaining = [ep.get("episode_num") for ep in plot_tracker if isinstance(ep, dict) and ep.get("status") == "pending"]
        if not chapters_remaining:
             st.session_state.stage = 'finished'; st.session_state.current_chapter_generating = None; add_log("All chapters processed.")
        else:
             st.session_state.stage = 'generating_chapters' # Stay in this stage for next chapter
        time.sleep(0.5) # Short pause
        st.rerun()

    except Exception as e:
         st.error(f"Error during chapter {ch_num} generation: {e}")
         add_log(f"ERROR Chap {ch_num}: {e}\n{traceback.format_exc()}")
         st.session_state.error_message = f"Failed generating chapter {ch_num}: {e}"
         # Mark chapter as failed
         for ep in plot_tracker:
              if isinstance(ep, dict) and ep.get("episode_num") == ch_num: ep["status"] = "failed_generation"; break
         # Decide how to proceed - go to finished state to show partial results
         st.session_state.stage = 'finished'; st.session_state.current_chapter_generating = None
         st.rerun()


# === Finished Stage ===
elif st.session_state.stage == 'finished':
    st.header("🎉 Story Complete! 🎉")

    if st.session_state.error_message and "Failed generating chapter" in st.session_state.error_message:
        st.error(f"Note: An error occurred during generation: {st.session_state.error_message}. Results may be incomplete.")

    # --- Display Book Cover ---
    st.subheader("Book Cover")
    cover_placeholder = st.empty()
    if st.session_state.cover_file and os.path.exists(st.session_state.cover_file):
         cover_placeholder.image(st.session_state.cover_file, caption="Generated Book Cover", use_column_width=True)
    else:
         can_gen_cover = bool(openai.api_key and st.session_state.story_seed)
         btn_disabled = not can_gen_cover
         btn_label = "🖼️ Generate Book Cover (DALL-E 3)" if can_gen_cover else "Configure OpenAI Key & Start Over to Generate Cover"
         if cover_placeholder.button(btn_label, key="finish_gen_cover_btn", disabled=btn_disabled):
             with st.spinner("Generating cover image..."):
                seed = st.session_state.story_seed
                prompt_gen_prompt = f"""Generate a DALL-E 3 prompt for a book cover based on this seed: {json.dumps(seed)}"""
                img_prompt = call_llm(prompt_gen_prompt)
                if not img_prompt or img_prompt.startswith("ERROR:"): img_prompt = f"Cover for {seed.get('genre','story')} titled '{seed.get('title_concept','Untitled')}'"
                st.info(f"Using image prompt: {img_prompt[:150]}...")
                cover_filename = "generated_book_cover.png"
                success = generate_cover_image(img_prompt, cover_filename)
                if success: st.session_state.cover_file = cover_filename; st.rerun()
                else: st.error("Cover generation failed."); st.session_state.cover_file = None

    st.divider()

    # --- Display Full Story Audio Section ---
    st.subheader("Full Story Audio (gTTS)")
    audio_section_placeholder = st.container()

    # Combine text if not already done
    if not st.session_state.full_story_text and st.session_state.final_chapters:
        texts, outline_map = [], {}
        if st.session_state.confirmed_outline: outline_map = {ep.get("episode_num"): ep for ep in st.session_state.confirmed_outline.get("episodes", []) if isinstance(ep, dict)}
        for i in sorted(st.session_state.final_chapters.keys()):
             title = outline_map.get(i, {}).get('title', f"Chapter {i}")
             content = st.session_state.final_chapters[i]
             if content and not str(content).startswith("ERROR:"): texts.append(f"{title}\n\n{content}")
             else: texts.append(f"\n\n[Chapter {i} Failed]\n\n")
        st.session_state.full_story_text = "\n\n".join(texts).strip()

    # Display Audio Button and Player
    if st.session_state.full_story_text:
        if audio_section_placeholder.button("🎧 Generate/Regenerate Full Audio", key="finish_gen_audio_btn"):
            with st.spinner("Generating full story audio (gTTS)..."):
                audio_filename = "generated_story_gtts.mp3"
                success = generate_audio_gtts(st.session_state.full_story_text, audio_filename)
                if success: st.session_state.audio_file = audio_filename; st.success("Audio Generated!")
                else: st.error("Audio generation failed."); st.session_state.audio_file = None
                st.rerun()

        if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
            try:
                audio_bytes = open(st.session_state.audio_file, 'rb').read()
                audio_section_placeholder.audio(audio_bytes, format='audio/mp3')
            except Exception as e: audio_section_placeholder.warning(f"Could not load audio: {e}")

        # Download full text
        audio_section_placeholder.download_button("Download Full Story Text", st.session_state.full_story_text, "full_story.txt", "text/plain")
    else:
         audio_section_placeholder.warning("No chapter content generated to create audio or text file.")

    st.divider()

    # --- Display Chapters (Using Tabs) ---
    st.subheader("Story Chapters")
    final_chapters = st.session_state.final_chapters
    confirmed_outline = st.session_state.confirmed_outline

    if final_chapters:
        chapter_titles, outline_map = [], {}
        if confirmed_outline and isinstance(confirmed_outline.get("episodes"), list):
             valid_episodes = sorted([ep for ep in confirmed_outline["episodes"] if isinstance(ep, dict)], key=lambda x: x.get('episode_num', 0))
             outline_map = {ep.get("episode_num"): ep for ep in valid_episodes}
             chapter_titles = [ f"Ch {ep.get('episode_num', '?')}: {ep.get('title', 'Untitled')}" for ep in valid_episodes ]
        else: chapter_titles = [f"Chapter {i}" for i in sorted(final_chapters.keys())]

        if not chapter_titles:
            st.warning("Content generated, but no chapter titles found.")
            for ch_num in sorted(final_chapters.keys()): st.markdown(final_chapters.get(ch_num, ""))
        else:
            tabs = st.tabs(chapter_titles)
            for i, tab in enumerate(tabs):
                try: ch_num = sorted(final_chapters.keys())[i]
                except IndexError: st.error("Tab/Chapter mismatch."); continue
                with tab:
                    content = final_chapters.get(ch_num, "Content missing.")
                    if content and not str(content).startswith("ERROR:"):
                        st.markdown(content, unsafe_allow_html=True)
                        st.download_button(f"DL Ch {ch_num} Text", content, f"chapter_{ch_num}_content.txt", "text/plain", key=f"dl_ch_{ch_num}")
                    else: st.error(f"Chapter generation failed: {content}")
    else:
        st.warning("No chapters were generated.")

    st.divider()
    # Button to start over
    if st.button("🔄 Start New Story"):
        reset_state_for_new_story()
        st.rerun()

    # Optional: Display Logs
    with st.expander("Show Processing Logs"):
         st.text_area("Logs", "\n".join(st.session_state.log_messages), height=200, key="log_display_final", disabled=True)


# ==================================================
# Streamlit runs the script top-to-bottom on each interaction.
# No explicit main execution block needed.
# ==================================================