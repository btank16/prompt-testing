import streamlit as st
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv

from gemini_client import GeminiClient
from perplexity_client import PerplexityAPIClient
from openai_client import OpenAIClient
from content_extractor import extract_website, extract_pdf_from_upload, extract_pdf_from_url
from utils import (
    detect_variables,
    substitute_variables,
    extract_schema_from_stored_format,
    normalize_response,
    detect_provider_from_model,
    build_response_format,
    expand_json_to_rows,
)
from batch_helpers import (
    render_prompt_inputs,
    render_variable_table,
    render_run_controls,
    execute_batch,
    build_results_dataframe,
    render_results,
)

load_dotenv()

st.set_page_config(page_title="LLM Prompt Tester", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "gemini_client": None,
    "gemini_connected": False,
    "gemini_results_df": None,
    "gemini_variable_df": None,
    "perplexity_client": None,
    "perplexity_connected": False,
    "perplexity_results_df": None,
    "perplexity_variable_df": None,
    "openai_client": None,
    "openai_connected": False,
    "openai_results_df": None,
    "openai_variable_df": None,
    # Content extraction state
    "extracted_content": "",
    "extracted_source": "",
    "extracted_metadata": {},
    # Search & Analyze state
    "sa_results_df": None,
    "sa_variable_df": None,
    "sa_pipeline_log": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Client init helpers
# ---------------------------------------------------------------------------
def _init_gemini(api_key: str):
    try:
        st.session_state.gemini_client = GeminiClient(api_key)
        st.session_state.gemini_connected = True
    except Exception as e:
        st.session_state.gemini_client = None
        st.session_state.gemini_connected = False
        st.error(f"Gemini init failed: {e}")


def _init_perplexity(api_key: str):
    try:
        st.session_state.perplexity_client = PerplexityAPIClient(api_key)
        st.session_state.perplexity_connected = True
    except Exception as e:
        st.session_state.perplexity_client = None
        st.session_state.perplexity_connected = False
        st.error(f"Perplexity init failed: {e}")


def _init_openai(api_key: str):
    try:
        st.session_state.openai_client = OpenAIClient(api_key)
        st.session_state.openai_connected = True
    except Exception as e:
        st.session_state.openai_client = None
        st.session_state.openai_connected = False
        st.error(f"OpenAI init failed: {e}")


# Auto-load keys from .env on first run
if st.session_state.gemini_client is None:
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key:
        _init_gemini(env_key)

if st.session_state.perplexity_client is None:
    env_key = os.getenv("PERPLEXITY_API_KEY", "")
    if env_key:
        _init_perplexity(env_key)

if st.session_state.openai_client is None:
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        _init_openai(env_key)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("LLM Prompt Tester")

    # --- API Keys ---
    st.subheader("API Keys")

    with st.expander("Gemini", expanded=not st.session_state.gemini_connected):
        gem_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            key="gem_key_input",
        )
        if st.button("Connect", key="gem_connect"):
            if gem_key:
                _init_gemini(gem_key)
            else:
                st.warning("Enter an API key")
        if st.session_state.gemini_connected:
            st.success("Connected")
        else:
            st.error("Not connected")

    with st.expander("Perplexity", expanded=not st.session_state.perplexity_connected):
        pplx_key = st.text_input(
            "Perplexity API Key",
            type="password",
            value=os.getenv("PERPLEXITY_API_KEY", ""),
            key="pplx_key_input",
        )
        if st.button("Connect", key="pplx_connect"):
            if pplx_key:
                _init_perplexity(pplx_key)
            else:
                st.warning("Enter an API key")
        if st.session_state.perplexity_connected:
            st.success("Connected")
        else:
            st.error("Not connected")

    with st.expander("OpenAI", expanded=not st.session_state.openai_connected):
        oai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="oai_key_input",
        )
        if st.button("Connect", key="oai_connect"):
            if oai_key:
                _init_openai(oai_key)
            else:
                st.warning("Enter an API key")
        if st.session_state.openai_connected:
            st.success("Connected")
        else:
            st.error("Not connected")

    st.divider()

    # --- Batch Settings ---
    st.subheader("Batch Settings")
    delay_between = st.number_input(
        "Delay between requests (s)", min_value=0.0, max_value=10.0, value=0.5, step=0.1
    )
    max_workers = st.number_input("Max concurrent workers", min_value=1, max_value=20, value=5)

    st.divider()

    # --- Load Config ---
    st.subheader("Load Config")
    prompts_dir = os.path.join(os.path.dirname(__file__), "Good_prompts")
    config_files = []
    if os.path.isdir(prompts_dir):
        config_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith(".json")])

    selected_config = st.selectbox("Good_prompts/", ["(none)"] + config_files)

    if st.button("Load selected") and selected_config != "(none)":
        path = os.path.join(prompts_dir, selected_config)
        try:
            with open(path) as f:
                cfg = json.load(f)
            st.session_state["_loaded_config"] = cfg
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load: {e}")

    uploaded_config = st.file_uploader("Or upload config JSON", type=["json"])
    if uploaded_config is not None:
        try:
            cfg = json.load(uploaded_config)
            st.session_state["_loaded_config"] = cfg
            st.rerun()
        except Exception as e:
            st.error(f"Failed to parse uploaded config: {e}")

# ---------------------------------------------------------------------------
# Apply loaded config (runs after rerun)
# ---------------------------------------------------------------------------
loaded = st.session_state.pop("_loaded_config", None)
if loaded:
    model_name = loaded.get("model", "")
    provider = detect_provider_from_model(model_name)
    tab_key = {"gemini": "gemini", "perplexity": "perplexity", "openai": "openai"}.get(
        provider, "gemini"
    )

    # Prompt & system prompt
    st.session_state[f"{tab_key}_system_prompt"] = loaded.get("system_prompt", "")
    st.session_state[f"{tab_key}_prompt_template"] = loaded.get(
        "prompt", loaded.get("prompt_template", "")
    )

    # JSON schema — try various config key names
    schema = None
    jf = loaded.get("json_format", loaded.get("expected_json", ""))
    if jf:
        schema = extract_schema_from_stored_format(jf)
    if not schema:
        js = loaded.get("json_schema", "")
        if js:
            if isinstance(js, str) and js.strip():
                try:
                    schema = json.loads(js)
                except json.JSONDecodeError:
                    pass
            elif isinstance(js, dict):
                schema = js
    if schema:
        st.session_state[f"{tab_key}_use_schema"] = True
        st.session_state[f"{tab_key}_schema_text"] = json.dumps(schema, indent=2)

    # Variable data
    var_data = loaded.get("variable_data")
    if var_data:
        st.session_state[f"{tab_key}_variable_df"] = pd.DataFrame(var_data)

    # --- Provider-specific fields ---
    if provider == "perplexity":
        st.session_state["pplx_model"] = model_name
        st.session_state["pplx_url"] = loaded.get("url", "")

        sp = loaded.get("search_params", {})
        st.session_state["pplx_domain_filter"] = sp.get("domain_filter", "")
        recency = sp.get("recency_filter", "none")
        if recency in ("none", "hour", "day", "week", "month"):
            st.session_state["pplx_recency_filter"] = recency
        ctx = sp.get("context_size", "low")
        if ctx in ("low", "medium", "high"):
            st.session_state["pplx_context_size"] = ctx
        st.session_state["pplx_after_date"] = sp.get("after_date", "")
        st.session_state["pplx_before_date"] = sp.get("before_date", "")
        st.session_state["pplx_return_images"] = sp.get("return_images", False)
        st.session_state["pplx_return_questions"] = sp.get("return_questions", False)

        loc = loaded.get("location", {})
        st.session_state["pplx_latitude"] = str(loc.get("latitude", ""))
        st.session_state["pplx_longitude"] = str(loc.get("longitude", ""))
        st.session_state["pplx_country"] = str(loc.get("country", ""))

        llm = loaded.get("llm_params", {})
        temp = llm.get("temperature")
        if temp not in (None, ""):
            st.session_state["pplx_temperature"] = float(temp)
        max_tok = llm.get("max_tokens")
        if max_tok not in (None, ""):
            st.session_state["pplx_max_tokens"] = int(max_tok)
        tp = llm.get("top_p")
        if tp not in (None, ""):
            st.session_state["pplx_top_p"] = float(tp)
        fp = llm.get("frequency_penalty")
        if fp not in (None, ""):
            st.session_state["pplx_freq_penalty"] = float(fp)
        pp = llm.get("presence_penalty")
        if pp not in (None, ""):
            st.session_state["pplx_pres_penalty"] = float(pp)

    elif provider == "openai":
        st.session_state["oai_model"] = model_name
        llm = loaded.get("llm_params", {})
        max_tok = llm.get("max_tokens")
        if max_tok not in (None, ""):
            st.session_state["oai_max_tokens"] = int(max_tok)
        re_val = llm.get("reasoning_effort")
        if re_val in ("minimal", "low", "medium", "high"):
            st.session_state["oai_reasoning"] = re_val
        vb_val = llm.get("verbosity")
        if vb_val in ("low", "medium", "high"):
            st.session_state["oai_verbosity"] = vb_val

    elif provider == "gemini":
        st.session_state["gem_model"] = model_name
        llm = loaded.get("llm_params", {})
        temp = llm.get("temperature")
        if temp is not None:
            st.session_state["gem_temperature"] = float(temp)
        tp = llm.get("top_p")
        if tp is not None:
            st.session_state["gem_top_p"] = float(tp)
        mot = llm.get("max_output_tokens")
        if mot is not None:
            st.session_state["gem_max_tokens"] = int(mot)
        tl = llm.get("thinking_level")
        if tl in ("none", "low", "medium", "high"):
            st.session_state["gem_thinking"] = tl

# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------
tab_gemini, tab_perplexity, tab_openai, tab_extract, tab_search_analyze = st.tabs(
    ["Gemini Batch", "Perplexity", "OpenAI", "Content Extraction", "Search & Analyze"]
)

# ========================  GEMINI BATCH TAB  ================================
with tab_gemini:

    # Model & LLM params
    with st.expander("Model & LLM Parameters", expanded=False):
        gem_model = st.selectbox(
            "Model",
            ["gemini-3-pro-preview", "gemini-3-flash-preview"],
            key="gem_model",
        )
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            gem_temperature = st.slider(
                "Temperature", 0.0, 2.0, 1.0, 0.05, key="gem_temperature"
            )
            gem_top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.01, key="gem_top_p")
        with gcol2:
            gem_max_tokens = st.number_input(
                "Max output tokens",
                min_value=1,
                max_value=65536,
                value=16384,
                key="gem_max_tokens",
            )
            gem_thinking = st.selectbox(
                "Thinking level",
                ["none", "low", "medium", "high"],
                key="gem_thinking",
            )

    # Grounding tools
    with st.expander("Grounding Tools", expanded=False):
        gem_google_search = st.checkbox(
            "Grounding with Google Search",
            key="gem_google_search",
            help="Connect Gemini to real-time Google Search results for grounded answers with citations.",
        )
        gem_url_context = st.checkbox(
            "URL Context",
            key="gem_url_context",
            help="Gemini will fetch and analyze content from URLs included in your prompt.",
        )

    # Shared prompt / variable / schema UI
    inputs = render_prompt_inputs("gemini")
    st.divider()
    edited_df = render_variable_table("gemini", inputs["variables"])
    st.divider()

    # Save config
    save_config = {
        "model": gem_model,
        "prompt_template": inputs["prompt_template"],
        "system_prompt": inputs["system_prompt"],
        "json_schema": inputs["schema_text"] if inputs["use_schema"] else "",
        "llm_params": {
            "temperature": gem_temperature,
            "max_output_tokens": gem_max_tokens,
            "top_p": gem_top_p,
            "thinking_level": gem_thinking,
        },
        "grounding": {
            "google_search": gem_google_search,
            "url_context": gem_url_context,
        },
        "batch_settings": {"delay_between": delay_between, "max_workers": int(max_workers)},
        "variable_data": edited_df.to_dict(orient="records") if inputs["variables"] else [],
    }

    run_clicked = render_run_controls(
        "gemini",
        edited_df,
        inputs["variables"],
        delay_between,
        st.session_state.gemini_connected,
        save_config,
    )

    if run_clicked:
        if not st.session_state.gemini_client:
            st.error("Connect your API key first.")
        elif not inputs["prompt_template"].strip():
            st.error("Enter a prompt template.")
        else:
            client = st.session_state.gemini_client

            json_schema = None
            if inputs["use_schema"] and inputs["schema_text"].strip():
                try:
                    json_schema = json.loads(inputs["schema_text"])
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON schema: {e}")
                    st.stop()

            rows = (
                edited_df.fillna("").to_dict(orient="records")
                if inputs["variables"]
                else [{}]
            )

            def _gemini_call(idx, row_vars):
                prompt = (
                    substitute_variables(inputs["prompt_template"], row_vars)
                    if row_vars
                    else inputs["prompt_template"]
                )
                start = time.time()
                try:
                    resp = client.generate_content(
                        model=gem_model,
                        prompt=prompt,
                        system_prompt=(
                            inputs["system_prompt"]
                            if inputs["system_prompt"].strip()
                            else None
                        ),
                        json_schema=json_schema,
                        temperature=gem_temperature,
                        top_p=gem_top_p,
                        max_output_tokens=gem_max_tokens,
                        thinking_level=(
                            gem_thinking if gem_thinking != "none" else None
                        ),
                        use_google_search=gem_google_search,
                        use_url_context=gem_url_context,
                    )
                    return idx, row_vars, resp, time.time() - start, None
                except Exception as e:
                    return idx, row_vars, None, time.time() - start, str(e)

            results = execute_batch(_gemini_call, rows, delay_between, max_workers)
            st.session_state["gemini_results_df"] = build_results_dataframe(
                results, inputs["variables"]
            )

    render_results("gemini")

# ========================  PERPLEXITY TAB  ==================================
with tab_perplexity:

    # Model select
    pplx_model = st.selectbox(
        "Model",
        ["sonar", "sonar-pro", "sonar-reasoning", "sonar-deep-research"],
        key="pplx_model",
    )

    # Search parameters
    with st.expander("Search Parameters", expanded=False):
        pplx_url = st.text_input(
            "Reference URL", key="pplx_url", placeholder="https://example.com"
        )
        pplx_domain_filter = st.text_input(
            "Domain filter (comma-separated, max 3)",
            key="pplx_domain_filter",
            placeholder="example.com, another.com",
        )

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            pplx_recency = st.selectbox(
                "Recency filter",
                ["none", "hour", "day", "week", "month"],
                key="pplx_recency_filter",
            )
            pplx_context_size = st.selectbox(
                "Search context size",
                ["low", "medium", "high"],
                key="pplx_context_size",
            )
        with pcol2:
            pplx_after = st.text_input("After date (MM/DD/YYYY)", key="pplx_after_date")
            pplx_before = st.text_input("Before date (MM/DD/YYYY)", key="pplx_before_date")

        pcol3, pcol4 = st.columns(2)
        with pcol3:
            pplx_return_images = st.checkbox("Return images", key="pplx_return_images")
            pplx_return_questions = st.checkbox(
                "Return related questions", key="pplx_return_questions"
            )
        with pcol4:
            st.markdown("**Location**")
            pplx_lat = st.text_input("Latitude", key="pplx_latitude")
            pplx_lon = st.text_input("Longitude", key="pplx_longitude")
            pplx_country = st.text_input("Country code", key="pplx_country")

    # LLM Parameters
    with st.expander("LLM Parameters", expanded=False):
        plcol1, plcol2 = st.columns(2)
        with plcol1:
            pplx_temperature = st.slider(
                "Temperature", 0.0, 2.0, 0.2, 0.05, key="pplx_temperature"
            )
            pplx_top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.01, key="pplx_top_p")
            pplx_max_tokens = st.number_input(
                "Max tokens",
                min_value=1,
                max_value=128000,
                value=4096,
                key="pplx_max_tokens",
            )
        with plcol2:
            pplx_freq_penalty = st.slider(
                "Frequency penalty", -2.0, 2.0, 0.0, 0.1, key="pplx_freq_penalty"
            )
            pplx_pres_penalty = st.slider(
                "Presence penalty", -2.0, 2.0, 0.0, 0.1, key="pplx_pres_penalty"
            )

    # Shared prompt / variable / schema UI
    inputs = render_prompt_inputs("perplexity")
    st.divider()
    edited_df = render_variable_table("perplexity", inputs["variables"])
    st.divider()

    # Save config
    save_config = {
        "model": pplx_model,
        "prompt": inputs["prompt_template"],
        "system_prompt": inputs["system_prompt"],
        "url": pplx_url,
        "search_params": {
            "domain_filter": pplx_domain_filter,
            "recency_filter": pplx_recency,
            "context_size": pplx_context_size,
            "after_date": pplx_after,
            "before_date": pplx_before,
            "return_images": pplx_return_images,
            "return_questions": pplx_return_questions,
        },
        "location": {
            "latitude": pplx_lat,
            "longitude": pplx_lon,
            "country": pplx_country,
        },
        "llm_params": {
            "temperature": pplx_temperature,
            "max_tokens": pplx_max_tokens,
            "top_p": pplx_top_p,
            "frequency_penalty": pplx_freq_penalty,
            "presence_penalty": pplx_pres_penalty,
        },
        "use_json": inputs["use_schema"],
        "expected_json": inputs["schema_text"] if inputs["use_schema"] else "",
        "variable_data": edited_df.to_dict(orient="records") if inputs["variables"] else [],
    }

    run_clicked = render_run_controls(
        "perplexity",
        edited_df,
        inputs["variables"],
        delay_between,
        st.session_state.perplexity_connected,
        save_config,
    )

    if run_clicked:
        if not st.session_state.perplexity_client:
            st.error("Connect your Perplexity API key first.")
        elif not inputs["prompt_template"].strip():
            st.error("Enter a prompt template.")
        else:
            client = st.session_state.perplexity_client

            # Parse schema
            response_format = None
            if inputs["use_schema"] and inputs["schema_text"].strip():
                try:
                    raw_schema = json.loads(inputs["schema_text"])
                    response_format = build_response_format("perplexity", raw_schema)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON schema: {e}")
                    st.stop()

            # Build search params
            domain_list = None
            if pplx_domain_filter.strip():
                domain_list = [
                    d.strip() for d in pplx_domain_filter.split(",") if d.strip()
                ][:3]

            recency = pplx_recency if pplx_recency != "none" else None

            user_location = None
            if pplx_lat.strip() or pplx_lon.strip() or pplx_country.strip():
                user_location = {}
                if pplx_lat.strip():
                    user_location["latitude"] = float(pplx_lat)
                if pplx_lon.strip():
                    user_location["longitude"] = float(pplx_lon)
                if pplx_country.strip():
                    user_location["country"] = pplx_country.strip()

            rows = (
                edited_df.fillna("").to_dict(orient="records")
                if inputs["variables"]
                else [{}]
            )

            def _perplexity_call(idx, row_vars):
                prompt = (
                    substitute_variables(inputs["prompt_template"], row_vars)
                    if row_vars
                    else inputs["prompt_template"]
                )
                messages = []
                if inputs["system_prompt"].strip():
                    messages.append(
                        {"role": "system", "content": inputs["system_prompt"]}
                    )
                messages.append({"role": "user", "content": prompt})

                start = time.time()
                try:
                    raw = client.chat_completion(
                        model=pplx_model,
                        messages=messages,
                        response_format=response_format,
                        url=pplx_url if pplx_url.strip() else None,
                        search_domain_filter=domain_list,
                        search_recency_filter=recency,
                        search_after_date_filter=(
                            pplx_after if pplx_after.strip() else None
                        ),
                        search_before_date_filter=(
                            pplx_before if pplx_before.strip() else None
                        ),
                        search_context_size=pplx_context_size,
                        return_images=True if pplx_return_images else None,
                        return_related_questions=(
                            True if pplx_return_questions else None
                        ),
                        user_location=user_location,
                        temperature=pplx_temperature,
                        max_tokens=pplx_max_tokens,
                        top_p=pplx_top_p,
                        frequency_penalty=(
                            pplx_freq_penalty if pplx_freq_penalty != 0.0 else None
                        ),
                        presence_penalty=(
                            pplx_pres_penalty if pplx_pres_penalty != 0.0 else None
                        ),
                    )
                    resp = normalize_response("perplexity", raw)
                    return idx, row_vars, resp, time.time() - start, None
                except Exception as e:
                    return idx, row_vars, None, time.time() - start, str(e)

            results = execute_batch(_perplexity_call, rows, delay_between, max_workers)
            st.session_state["perplexity_results_df"] = build_results_dataframe(
                results, inputs["variables"], include_citations=True
            )

    render_results("perplexity", include_citations=True)

# ========================  OPENAI TAB  ======================================
with tab_openai:

    # Model select
    oai_model = st.selectbox(
        "Model",
        ["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        key="oai_model",
    )

    is_gpt5 = oai_model in ("gpt-5", "gpt-5-mini", "gpt-5-nano")

    # Model-specific parameters
    with st.expander("Model Parameters", expanded=False):
        ocol1, ocol2 = st.columns(2)
        with ocol1:
            oai_reasoning = st.selectbox(
                "Reasoning effort",
                ["minimal", "low", "medium", "high"],
                index=2,
                key="oai_reasoning",
            )
            oai_verbosity = st.selectbox(
                "Verbosity",
                ["low", "medium", "high"],
                index=1,
                key="oai_verbosity",
            )
        with ocol2:
            oai_seed = st.number_input(
                "Seed (0 = none)", min_value=0, value=0, key="oai_seed"
            )
            oai_logprobs = st.checkbox(
                "Logprobs", disabled=is_gpt5, key="oai_logprobs"
            )
            oai_top_logprobs = None
            if oai_logprobs and not is_gpt5:
                oai_top_logprobs = st.number_input(
                    "Top logprobs",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="oai_top_logprobs",
                )

    # LLM Parameters
    with st.expander("LLM Parameters", expanded=False):
        if is_gpt5:
            st.info(
                "GPT-5 models only support max tokens. Temperature is fixed at 1.0; "
                "top-p, frequency penalty, and presence penalty are not supported."
            )
        olcol1, olcol2 = st.columns(2)
        with olcol1:
            oai_max_tokens = st.number_input(
                "Max tokens",
                min_value=1,
                max_value=128000,
                value=4096,
                key="oai_max_tokens",
            )
            if not is_gpt5:
                oai_temperature = st.slider(
                    "Temperature", 0.0, 2.0, 1.0, 0.05, key="oai_temperature"
                )
                oai_top_p = st.slider(
                    "Top-p", 0.0, 1.0, 1.0, 0.01, key="oai_top_p"
                )
        with olcol2:
            if not is_gpt5:
                oai_freq_penalty = st.slider(
                    "Frequency penalty",
                    -2.0,
                    2.0,
                    0.0,
                    0.1,
                    key="oai_freq_penalty",
                )
                oai_pres_penalty = st.slider(
                    "Presence penalty",
                    -2.0,
                    2.0,
                    0.0,
                    0.1,
                    key="oai_pres_penalty",
                )

    # Shared prompt / variable / schema UI
    inputs = render_prompt_inputs("openai")
    st.divider()
    edited_df = render_variable_table("openai", inputs["variables"])
    st.divider()

    # Save config
    _oai_llm_params = {
        "max_tokens": oai_max_tokens,
        "reasoning_effort": oai_reasoning,
        "verbosity": oai_verbosity,
    }
    if not is_gpt5:
        _oai_llm_params.update(
            {
                "temperature": oai_temperature,
                "top_p": oai_top_p,
                "frequency_penalty": oai_freq_penalty,
                "presence_penalty": oai_pres_penalty,
            }
        )

    save_config = {
        "model": oai_model,
        "prompt_template": inputs["prompt_template"],
        "system_prompt": inputs["system_prompt"],
        "json_schema": inputs["schema_text"] if inputs["use_schema"] else "",
        "llm_params": _oai_llm_params,
        "seed": oai_seed if oai_seed > 0 else None,
        "variable_data": edited_df.to_dict(orient="records") if inputs["variables"] else [],
    }

    run_clicked = render_run_controls(
        "openai",
        edited_df,
        inputs["variables"],
        delay_between,
        st.session_state.openai_connected,
        save_config,
    )

    if run_clicked:
        if not st.session_state.openai_client:
            st.error("Connect your OpenAI API key first.")
        elif not inputs["prompt_template"].strip():
            st.error("Enter a prompt template.")
        else:
            client = st.session_state.openai_client

            # Parse schema
            response_format = None
            if inputs["use_schema"] and inputs["schema_text"].strip():
                try:
                    raw_schema = json.loads(inputs["schema_text"])
                    response_format = build_response_format("openai", raw_schema)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON schema: {e}")
                    st.stop()

            rows = (
                edited_df.fillna("").to_dict(orient="records")
                if inputs["variables"]
                else [{}]
            )

            def _openai_call(idx, row_vars):
                prompt = (
                    substitute_variables(inputs["prompt_template"], row_vars)
                    if row_vars
                    else inputs["prompt_template"]
                )
                messages = []
                if inputs["system_prompt"].strip():
                    messages.append(
                        {"role": "system", "content": inputs["system_prompt"]}
                    )
                messages.append({"role": "user", "content": prompt})

                kwargs = {
                    "model": oai_model,
                    "messages": messages,
                    "max_tokens": oai_max_tokens,
                    "reasoning_effort": oai_reasoning,
                    "verbosity": oai_verbosity,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                if oai_seed > 0:
                    kwargs["seed"] = oai_seed
                if oai_logprobs and not is_gpt5:
                    kwargs["logprobs"] = True
                    if oai_top_logprobs is not None:
                        kwargs["top_logprobs"] = oai_top_logprobs

                # Non-GPT-5 standard params
                if not is_gpt5:
                    kwargs["temperature"] = oai_temperature
                    kwargs["top_p"] = oai_top_p
                    kwargs["frequency_penalty"] = oai_freq_penalty
                    kwargs["presence_penalty"] = oai_pres_penalty

                start = time.time()
                try:
                    raw = client.chat_completion(**kwargs)
                    resp = normalize_response("openai", raw)
                    return idx, row_vars, resp, time.time() - start, None
                except Exception as e:
                    return idx, row_vars, None, time.time() - start, str(e)

            results = execute_batch(_openai_call, rows, delay_between, max_workers)
            st.session_state["openai_results_df"] = build_results_dataframe(
                results, inputs["variables"]
            )

    render_results("openai")

# ========================  CONTENT EXTRACTION TAB  ============================
with tab_extract:
    st.header("Content Extraction")
    st.caption(
        "Extract content from websites or PDFs. Use extracted text as a variable "
        "in prompt templates across all provider tabs."
    )

    # ---- Extraction settings ----
    with st.expander("Extraction Settings", expanded=False):
        max_chars = st.number_input(
            "Max content length (characters)",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=5000,
            key="extract_max_chars",
            help="Content exceeding this limit will be truncated.",
        )

    # ---- Source selection ----
    source_type = st.radio(
        "Extraction source",
        ["Website", "PDF Upload", "PDF from URL"],
        horizontal=True,
        key="extract_source_type",
    )

    # ---- Source-specific inputs ----
    if source_type == "Website":
        extract_url = st.text_input(
            "URL to extract",
            placeholder="https://example.com/article",
            key="web_extract_url",
        )

        if st.button("Extract Website", key="extract_web_btn", type="primary"):
            if not extract_url.strip():
                st.error("Enter a URL.")
            else:
                with st.spinner("Extracting website content..."):
                    result = extract_website(extract_url.strip())
                if result["success"]:
                    content = result["content"]
                    if len(content) > max_chars:
                        content = content[:max_chars]
                        st.warning(
                            f"Content truncated from {len(result['content']):,} "
                            f"to {max_chars:,} characters."
                        )
                    st.session_state.extracted_content = content
                    st.session_state.extracted_source = extract_url
                    st.session_state.extracted_metadata = {
                        "type": "website",
                        "title": result.get("title", ""),
                        "word_count": len(content.split()),
                        "elapsed": result.get("elapsed", 0),
                    }
                    st.success(
                        f"Extracted {len(content.split()):,} words in "
                        f"{result.get('elapsed', 0)}s"
                    )
                else:
                    st.error(f"Extraction failed: {result['error']}")

    elif source_type == "PDF Upload":
        pdf_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="pdf_upload",
            help="Max file size: 50 MB",
        )

        if st.button("Extract PDF", key="extract_pdf_upload_btn", type="primary"):
            if pdf_file is None:
                st.error("Upload a PDF file.")
            else:
                with st.spinner("Extracting PDF content..."):
                    result = extract_pdf_from_upload(pdf_file.read(), pdf_file.name)
                if result["success"]:
                    content = result["content"]
                    if len(content) > max_chars:
                        content = content[:max_chars]
                        st.warning(
                            f"Content truncated from {len(result['content']):,} "
                            f"to {max_chars:,} characters."
                        )
                    st.session_state.extracted_content = content
                    st.session_state.extracted_source = pdf_file.name
                    st.session_state.extracted_metadata = {
                        "type": "pdf",
                        "title": pdf_file.name,
                        "page_count": result.get("page_count", 0),
                        "word_count": len(content.split()),
                        "elapsed": result.get("elapsed", 0),
                    }
                    st.success(
                        f"Extracted {len(content.split()):,} words "
                        f"({result.get('page_count', 0)} pages) in "
                        f"{result.get('elapsed', 0)}s"
                    )
                else:
                    st.error(f"PDF extraction failed: {result['error']}")

    else:  # PDF from URL
        pdf_url = st.text_input(
            "PDF URL",
            placeholder="https://example.com/document.pdf",
            key="pdf_extract_url",
        )

        if st.button("Extract PDF from URL", key="extract_pdf_url_btn", type="primary"):
            if not pdf_url.strip():
                st.error("Enter a PDF URL.")
            else:
                with st.spinner("Extracting PDF from URL..."):
                    result = extract_pdf_from_url(pdf_url.strip())
                if result["success"]:
                    content = result["content"]
                    if len(content) > max_chars:
                        content = content[:max_chars]
                        st.warning(
                            f"Content truncated from {len(result['content']):,} "
                            f"to {max_chars:,} characters."
                        )
                    st.session_state.extracted_content = content
                    st.session_state.extracted_source = pdf_url
                    st.session_state.extracted_metadata = {
                        "type": "pdf",
                        "title": result.get("title", ""),
                        "page_count": result.get("page_count", 0),
                        "word_count": len(content.split()),
                        "elapsed": result.get("elapsed", 0),
                    }
                    st.success(
                        f"Extracted {len(content.split()):,} words "
                        f"({result.get('page_count', 0)} pages) in "
                        f"{result.get('elapsed', 0)}s"
                    )
                else:
                    st.error(f"PDF extraction failed: {result['error']}")

    # ---- Preview extracted content ----
    if st.session_state.extracted_content:
        st.divider()
        meta = st.session_state.extracted_metadata
        info_parts = [f"Source: {st.session_state.extracted_source}"]
        if meta.get("title"):
            info_parts.append(f"Title: {meta['title']}")
        if meta.get("page_count"):
            info_parts.append(f"Pages: {meta['page_count']}")
        info_parts.append(f"Words: {meta.get('word_count', 0):,}")
        info_parts.append(f"Time: {meta.get('elapsed', 0)}s")
        st.caption(" | ".join(info_parts))

        st.text_area(
            "Extracted Content (preview)",
            value=st.session_state.extracted_content[:5000],
            height=300,
            disabled=True,
            key="extract_preview",
        )

        # ---- Export buttons ----
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.download_button(
                "Download as Markdown",
                data=st.session_state.extracted_content,
                file_name="extracted_content.md",
                mime="text/markdown",
                key="extract_dl_md",
            )
        with ecol2:
            st.download_button(
                "Download as Text",
                data=st.session_state.extracted_content,
                file_name="extracted_content.txt",
                mime="text/plain",
                key="extract_dl_txt",
            )
        with ecol3:
            if st.button("Clear Extracted Content", key="clear_extract"):
                st.session_state.extracted_content = ""
                st.session_state.extracted_source = ""
                st.session_state.extracted_metadata = {}
                st.rerun()

        # ---- Send to provider tab ----
        st.divider()
        st.subheader("Use as LLM Context")
        st.caption(
            "Inject extracted content into a provider's variable table. "
            "Then use the variable name (e.g. `{extracted_content}`) in your prompt template."
        )

        icol1, icol2 = st.columns(2)
        with icol1:
            target_provider = st.selectbox(
                "Target provider tab",
                ["gemini", "perplexity", "openai"],
                key="inject_target",
            )
        with icol2:
            var_name = st.text_input(
                "Variable name",
                value="extracted_content",
                key="inject_var_name",
            )

        if st.button("Send to Tab", key="inject_btn", type="primary"):
            if not var_name.strip():
                st.error("Enter a variable name.")
            else:
                vn = var_name.strip()
                df_key = f"{target_provider}_variable_df"
                existing_df = st.session_state.get(df_key)

                if existing_df is not None and not existing_df.empty:
                    existing_df[vn] = st.session_state.extracted_content
                    st.session_state[df_key] = existing_df
                else:
                    st.session_state[df_key] = pd.DataFrame(
                        {vn: [st.session_state.extracted_content]}
                    )
                st.success(
                    f"Injected into **{target_provider}** as variable "
                    f"`{{{vn}}}`. Switch to the {target_provider.title()} tab "
                    f"and add `{{{vn}}}` to your prompt template."
                )

# ========================  SEARCH & ANALYZE TAB  ==============================
with tab_search_analyze:
    st.header("Search & Analyze")
    st.caption(
        "Pipeline: Perplexity search → crawl URLs with crawl4ai → analyze with LLM. "
        "Use `{variable}` placeholders in both prompts for batch execution."
    )

    # ------------------------------------------------------------------
    # Step 1: Perplexity Search Configuration
    # ------------------------------------------------------------------
    st.subheader("Step 1: Perplexity Search")

    sa_search_system = st.text_area(
        "Search System Prompt",
        height=80,
        placeholder="Optional system instructions for the search query...",
        key="sa_search_system_prompt",
    )

    sa_search_template = st.text_area(
        "Search Prompt Template",
        height=120,
        placeholder=(
            "Enter your search query. Use {variable} placeholders for batch variables.\n"
            "Example: Find recent news about {company} in {industry}"
        ),
        key="sa_search_prompt_template",
    )

    sa_search_vars = detect_variables(sa_search_template)

    with st.expander("Search Parameters", expanded=False):
        sa_search_model = st.selectbox(
            "Search Model",
            ["sonar", "sonar-pro", "sonar-reasoning"],
            key="sa_search_model",
        )
        sacol1, sacol2 = st.columns(2)
        with sacol1:
            sa_domain_filter = st.text_input(
                "Domain filter (comma-separated, max 3)",
                key="sa_domain_filter",
                placeholder="example.com, another.com",
            )
            sa_recency = st.selectbox(
                "Recency filter",
                ["none", "hour", "day", "week", "month"],
                key="sa_recency_filter",
            )
        with sacol2:
            sa_context_size = st.selectbox(
                "Search context size",
                ["low", "medium", "high"],
                index=1,
                key="sa_context_size",
            )
            sa_search_temp = st.slider(
                "Search temperature", 0.0, 2.0, 0.2, 0.05, key="sa_search_temp"
            )

    st.divider()

    # ------------------------------------------------------------------
    # Step 2: Crawl Settings
    # ------------------------------------------------------------------
    st.subheader("Step 2: Crawl Settings")
    sa_crawl_col1, sa_crawl_col2 = st.columns(2)
    with sa_crawl_col1:
        sa_max_crawl_chars = st.number_input(
            "Max content per URL (characters)",
            min_value=1000,
            max_value=200000,
            value=30000,
            step=5000,
            key="sa_max_crawl_chars",
            help="Content from each URL will be truncated to this limit.",
        )
    with sa_crawl_col2:
        sa_crawl_timeout = st.number_input(
            "Crawl timeout per URL (seconds)",
            min_value=10,
            max_value=120,
            value=60,
            key="sa_crawl_timeout",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Step 3: Analysis LLM Configuration
    # ------------------------------------------------------------------
    st.subheader("Step 3: Analysis LLM")

    sa_analysis_provider = st.selectbox(
        "Analysis LLM Provider",
        ["Gemini", "OpenAI", "Perplexity"],
        key="sa_analysis_provider",
    )

    _sa_provider_models = {
        "Gemini": ["gemini-3-pro-preview", "gemini-3-flash-preview"],
        "OpenAI": ["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        "Perplexity": ["sonar", "sonar-pro", "sonar-reasoning"],
    }
    sa_analysis_model = st.selectbox(
        "Analysis Model",
        _sa_provider_models[sa_analysis_provider],
        key="sa_analysis_model",
    )

    sa_analysis_system = st.text_area(
        "Analysis System Prompt",
        height=80,
        placeholder="Optional system instructions for the analysis LLM...",
        key="sa_analysis_system_prompt",
    )

    sa_analysis_template = st.text_area(
        "Analysis Prompt Template",
        height=150,
        placeholder=(
            "The crawled content is automatically available as {crawled_content} and "
            "the search response as {search_response}.\n"
            "Example: Analyze the following content about {company}:\n\n{crawled_content}"
        ),
        key="sa_analysis_prompt_template",
    )

    sa_analysis_vars = detect_variables(sa_analysis_template)

    # Show detected variables from BOTH prompts (excluding auto-injected ones)
    _auto_vars = {"crawled_content", "search_response", "search_citations"}
    all_sa_vars_set = set(sa_search_vars) | set(sa_analysis_vars)
    user_sa_vars = sorted(all_sa_vars_set - _auto_vars)
    if sa_search_vars or sa_analysis_vars:
        parts = []
        if sa_search_vars:
            parts.append(
                "Search: " + "  ".join([f"`{{{v}}}`" for v in sa_search_vars])
            )
        if sa_analysis_vars:
            parts.append(
                "Analysis: " + "  ".join([f"`{{{v}}}`" for v in sa_analysis_vars])
            )
        st.caption("Detected variables — " + "  |  ".join(parts))
        st.caption(
            "Auto-injected variables available in analysis prompt: "
            "`{crawled_content}`, `{search_response}`, `{search_citations}`"
        )

    with st.expander("Analysis LLM Parameters", expanded=False):
        alcol1, alcol2 = st.columns(2)
        with alcol1:
            sa_analysis_temp = st.slider(
                "Temperature", 0.0, 2.0, 0.7, 0.05, key="sa_analysis_temp"
            )
            sa_analysis_max_tokens = st.number_input(
                "Max tokens",
                min_value=1,
                max_value=128000,
                value=8192,
                key="sa_analysis_max_tokens",
            )
        with alcol2:
            sa_analysis_top_p = st.slider(
                "Top-p", 0.0, 1.0, 0.9, 0.01, key="sa_analysis_top_p"
            )

    sa_use_schema = st.checkbox(
        "Use JSON schema for analysis output",
        key="sa_use_schema",
    )
    sa_schema_text = ""
    if sa_use_schema:
        sa_schema_text = st.text_area(
            "JSON Schema",
            height=200,
            placeholder='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}',
            key="sa_schema_text",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Variable Table (shared across both prompts)
    # ------------------------------------------------------------------
    st.subheader("Variable Input Data")
    if user_sa_vars:
        existing_sa_df = st.session_state.get("sa_variable_df")
        if existing_sa_df is not None and not existing_sa_df.empty:
            for v in user_sa_vars:
                if v not in existing_sa_df.columns:
                    existing_sa_df[v] = ""
            existing_sa_df = existing_sa_df[
                [v for v in user_sa_vars if v in existing_sa_df.columns]
            ]
            sa_df_input = existing_sa_df
        else:
            sa_df_input = pd.DataFrame({v: [""] for v in user_sa_vars})

        sa_rcol1, sa_rcol2, sa_rcol3 = st.columns([1, 1, 2])
        with sa_rcol1:
            sa_add_count = st.number_input(
                "Rows to add",
                min_value=1,
                max_value=500,
                value=5,
                key="sa_add_count",
            )
        with sa_rcol2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Add {sa_add_count} rows", key="sa_add_rows"):
                new_rows = pd.DataFrame(
                    {v: [""] * int(sa_add_count) for v in user_sa_vars}
                )
                sa_df_input = pd.concat([sa_df_input, new_rows], ignore_index=True)
                st.session_state["sa_variable_df"] = sa_df_input
                st.rerun()
        with sa_rcol3:
            sa_csv_file = st.file_uploader(
                "Upload CSV", type=["csv"], key="sa_csv_upload"
            )
        if sa_csv_file is not None:
            try:
                uploaded_sa_df = pd.read_csv(sa_csv_file, dtype=str).fillna("")
                missing_cols = [
                    v for v in user_sa_vars if v not in uploaded_sa_df.columns
                ]
                if missing_cols:
                    st.warning(
                        f"CSV missing columns: {', '.join(missing_cols)}. "
                        "They will be added empty."
                    )
                    for mc in missing_cols:
                        uploaded_sa_df[mc] = ""
                sa_df_input = uploaded_sa_df[
                    [v for v in user_sa_vars if v in uploaded_sa_df.columns]
                ]
            except Exception as e:
                st.error(f"CSV parse error: {e}")

        sa_edited_df = st.data_editor(
            sa_df_input,
            num_rows="dynamic",
            use_container_width=True,
            key="sa_var_editor",
        )
        st.session_state["sa_variable_df"] = sa_edited_df
    else:
        st.info(
            "Add `{variable}` placeholders to your search or analysis prompt "
            "to enable batch testing, or run a single pipeline with no variables."
        )
        sa_edited_df = pd.DataFrame({"_row": ["single call"]})

    st.divider()

    # ------------------------------------------------------------------
    # Run Controls
    # ------------------------------------------------------------------
    sa_num_rows = len(sa_edited_df) if user_sa_vars else 1
    st.caption(
        f"Total pipelines: **{sa_num_rows}** (each runs: search → crawl → analyze)"
    )

    # Check which providers are connected
    _sa_provider_connected = {
        "Gemini": st.session_state.gemini_connected,
        "OpenAI": st.session_state.openai_connected,
        "Perplexity": st.session_state.perplexity_connected,
    }
    sa_can_run = (
        st.session_state.perplexity_connected
        and _sa_provider_connected[sa_analysis_provider]
    )

    sa_col_run, sa_col_save = st.columns([1, 1])
    with sa_col_run:
        sa_run_clicked = st.button(
            "Run Pipeline",
            type="primary",
            disabled=not sa_can_run,
            key="sa_run",
        )
        if not st.session_state.perplexity_connected:
            st.caption("Connect Perplexity API key for search step.")
        if not _sa_provider_connected[sa_analysis_provider]:
            st.caption(
                f"Connect {sa_analysis_provider} API key for analysis step."
            )
    with sa_col_save:
        sa_save_config = {
            "type": "search_and_analyze",
            "search": {
                "model": sa_search_model,
                "system_prompt": sa_search_system,
                "prompt_template": sa_search_template,
                "domain_filter": sa_domain_filter,
                "recency_filter": sa_recency,
                "context_size": sa_context_size,
                "temperature": sa_search_temp,
            },
            "crawl": {
                "max_chars": sa_max_crawl_chars,
                "timeout": sa_crawl_timeout,
            },
            "analysis": {
                "provider": sa_analysis_provider,
                "model": sa_analysis_model,
                "system_prompt": sa_analysis_system,
                "prompt_template": sa_analysis_template,
                "temperature": sa_analysis_temp,
                "max_tokens": sa_analysis_max_tokens,
                "top_p": sa_analysis_top_p,
                "use_schema": sa_use_schema,
                "schema_text": sa_schema_text if sa_use_schema else "",
            },
            "variable_data": (
                sa_edited_df.to_dict(orient="records") if user_sa_vars else []
            ),
        }
        st.download_button(
            "Save Config",
            data=json.dumps(sa_save_config, indent=2),
            file_name="search_analyze_config.json",
            mime="application/json",
            key="sa_save",
        )

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------
    if sa_run_clicked:
        if not st.session_state.perplexity_client:
            st.error("Connect your Perplexity API key first (needed for search).")
        elif not sa_search_template.strip():
            st.error("Enter a search prompt template.")
        elif not sa_analysis_template.strip():
            st.error("Enter an analysis prompt template.")
        else:
            pplx_client = st.session_state.perplexity_client

            # Resolve analysis LLM client
            _sa_analysis_clients = {
                "Gemini": st.session_state.gemini_client,
                "OpenAI": st.session_state.openai_client,
                "Perplexity": st.session_state.perplexity_client,
            }
            analysis_client = _sa_analysis_clients[sa_analysis_provider]
            if not analysis_client:
                st.error(
                    f"Connect your {sa_analysis_provider} API key for analysis."
                )
                st.stop()

            # Parse analysis schema
            sa_analysis_schema = None
            if sa_use_schema and sa_schema_text.strip():
                try:
                    sa_analysis_schema = json.loads(sa_schema_text)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON schema: {e}")
                    st.stop()

            # Build search params
            sa_domain_list = None
            if sa_domain_filter.strip():
                sa_domain_list = [
                    d.strip()
                    for d in sa_domain_filter.split(",")
                    if d.strip()
                ][:3]
            sa_recency_val = sa_recency if sa_recency != "none" else None

            rows = (
                sa_edited_df.fillna("").to_dict(orient="records")
                if user_sa_vars
                else [{}]
            )

            # --- Pipeline function for each row ---
            def _sa_pipeline(idx, row_vars):
                pipeline_start = time.time()
                log = {"row": idx, "steps": {}}

                # === STEP 1: Perplexity Search ===
                search_prompt = (
                    substitute_variables(sa_search_template, row_vars)
                    if row_vars
                    else sa_search_template
                )
                messages = []
                if sa_search_system.strip():
                    messages.append(
                        {"role": "system", "content": sa_search_system}
                    )
                messages.append({"role": "user", "content": search_prompt})

                step1_start = time.time()
                try:
                    search_raw = pplx_client.chat_completion(
                        model=sa_search_model,
                        messages=messages,
                        search_domain_filter=sa_domain_list,
                        search_recency_filter=sa_recency_val,
                        search_context_size=sa_context_size,
                        temperature=sa_search_temp,
                    )
                    search_resp = normalize_response("perplexity", search_raw)
                    search_content = search_resp.get("content", "")
                    citations = search_resp.get("citations") or []
                    log["steps"]["search"] = {
                        "status": "ok",
                        "elapsed": round(time.time() - step1_start, 2),
                        "citations_found": len(citations),
                    }
                except Exception as e:
                    return (
                        idx,
                        row_vars,
                        None,
                        time.time() - pipeline_start,
                        f"Search failed: {e}",
                    )

                # === STEP 2: Crawl URLs ===
                crawled_parts = []
                crawl_errors = []
                step2_start = time.time()
                for url in citations:
                    try:
                        result = extract_website(url, timeout=sa_crawl_timeout)
                        if result["success"]:
                            content = result["content"]
                            if len(content) > sa_max_crawl_chars:
                                content = content[:sa_max_crawl_chars]
                            crawled_parts.append(
                                f"--- Source: {url} ---\n{content}"
                            )
                        else:
                            crawl_errors.append(
                                f"{url}: {result.get('error', 'unknown error')}"
                            )
                    except Exception as e:
                        crawl_errors.append(f"{url}: {e}")

                combined_crawled = "\n\n".join(crawled_parts)
                log["steps"]["crawl"] = {
                    "status": "ok" if crawled_parts else "partial",
                    "elapsed": round(time.time() - step2_start, 2),
                    "urls_attempted": len(citations),
                    "urls_succeeded": len(crawled_parts),
                    "urls_failed": len(crawl_errors),
                    "errors": crawl_errors[:5],
                }

                # === STEP 3: Analysis LLM ===
                # Inject auto-variables into row_vars for analysis prompt
                analysis_vars = dict(row_vars) if row_vars else {}
                analysis_vars["crawled_content"] = combined_crawled
                analysis_vars["search_response"] = search_content
                analysis_vars["search_citations"] = "; ".join(citations)

                analysis_prompt = substitute_variables(
                    sa_analysis_template, analysis_vars
                )

                step3_start = time.time()
                try:
                    provider_lower = sa_analysis_provider.lower()

                    if provider_lower == "gemini":
                        resp = analysis_client.generate_content(
                            model=sa_analysis_model,
                            prompt=analysis_prompt,
                            system_prompt=(
                                sa_analysis_system
                                if sa_analysis_system.strip()
                                else None
                            ),
                            json_schema=sa_analysis_schema,
                            temperature=sa_analysis_temp,
                            top_p=sa_analysis_top_p,
                            max_output_tokens=sa_analysis_max_tokens,
                        )
                    elif provider_lower == "openai":
                        a_messages = []
                        if sa_analysis_system.strip():
                            a_messages.append(
                                {
                                    "role": "system",
                                    "content": sa_analysis_system,
                                }
                            )
                        a_messages.append(
                            {"role": "user", "content": analysis_prompt}
                        )
                        a_kwargs = {
                            "model": sa_analysis_model,
                            "messages": a_messages,
                            "max_tokens": sa_analysis_max_tokens,
                        }
                        if sa_analysis_schema:
                            a_kwargs["response_format"] = build_response_format(
                                "openai", sa_analysis_schema
                            )
                        # GPT-5 models need reasoning_effort
                        if sa_analysis_model.startswith("gpt-5"):
                            a_kwargs["reasoning_effort"] = "medium"
                        else:
                            a_kwargs["temperature"] = sa_analysis_temp
                            a_kwargs["top_p"] = sa_analysis_top_p
                        raw = analysis_client.chat_completion(**a_kwargs)
                        resp = normalize_response("openai", raw)
                    else:  # perplexity
                        a_messages = []
                        if sa_analysis_system.strip():
                            a_messages.append(
                                {
                                    "role": "system",
                                    "content": sa_analysis_system,
                                }
                            )
                        a_messages.append(
                            {"role": "user", "content": analysis_prompt}
                        )
                        a_kwargs = {
                            "model": sa_analysis_model,
                            "messages": a_messages,
                            "temperature": sa_analysis_temp,
                            "max_tokens": sa_analysis_max_tokens,
                            "top_p": sa_analysis_top_p,
                        }
                        if sa_analysis_schema:
                            a_kwargs["response_format"] = build_response_format(
                                "perplexity", sa_analysis_schema
                            )
                        raw = analysis_client.chat_completion(**a_kwargs)
                        resp = normalize_response("perplexity", raw)

                    log["steps"]["analysis"] = {
                        "status": "ok",
                        "elapsed": round(time.time() - step3_start, 2),
                    }

                    # Attach pipeline metadata to response
                    resp["_pipeline_log"] = log
                    resp["_citations"] = citations
                    resp["_urls_crawled"] = len(crawled_parts)
                    resp["_urls_failed"] = len(crawl_errors)

                    return (
                        idx,
                        row_vars,
                        resp,
                        time.time() - pipeline_start,
                        None,
                    )
                except Exception as e:
                    return (
                        idx,
                        row_vars,
                        None,
                        time.time() - pipeline_start,
                        f"Analysis failed: {e}",
                    )

            # --- Execute pipeline across all rows ---
            results = execute_batch(
                _sa_pipeline, rows, delay_between, max_workers
            )

            # --- Build results DataFrame ---
            result_rows = []
            pipeline_logs = []
            for idx, row_vars, resp, elapsed, err in sorted(
                results, key=lambda x: x[0]
            ):
                base = {}
                for v in user_sa_vars:
                    base[v] = row_vars.get(v, "")

                base["_status"] = "ok" if err is None else "error"
                base["_duration_s"] = round(elapsed, 2)
                base["_error"] = err or ""

                if resp:
                    content = resp.get("content", "")
                    usage = resp.get("usage", {})
                    base["_prompt_tokens"] = usage.get("prompt_tokens", 0)
                    base["_completion_tokens"] = usage.get(
                        "completion_tokens", 0
                    )
                    base["_citations"] = (
                        "; ".join(resp.get("_citations", []))
                        if resp.get("_citations")
                        else ""
                    )
                    base["_urls_crawled"] = resp.get("_urls_crawled", 0)
                    base["_urls_failed"] = resp.get("_urls_failed", 0)

                    if resp.get("_pipeline_log"):
                        pipeline_logs.append(resp["_pipeline_log"])

                    try:
                        parsed = json.loads(content)
                        expanded = expand_json_to_rows(parsed)
                        for expanded_row in expanded:
                            row = dict(base)
                            row.update(expanded_row)
                            result_rows.append(row)
                    except (json.JSONDecodeError, TypeError):
                        row = dict(base)
                        row["_raw_response"] = content
                        result_rows.append(row)
                else:
                    result_rows.append(base)

            st.session_state["sa_results_df"] = pd.DataFrame(result_rows)
            st.session_state["sa_pipeline_log"] = pipeline_logs

    # ------------------------------------------------------------------
    # Results Display
    # ------------------------------------------------------------------
    sa_rdf = st.session_state.get("sa_results_df")
    if sa_rdf is not None:
        st.divider()
        st.subheader("Results")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        ok_count = (
            (sa_rdf["_status"] == "ok").sum()
            if "_status" in sa_rdf.columns
            else 0
        )
        err_count = (
            (sa_rdf["_status"] == "error").sum()
            if "_status" in sa_rdf.columns
            else 0
        )
        avg_dur = (
            sa_rdf["_duration_s"].mean()
            if "_duration_s" in sa_rdf.columns
            else 0
        )
        total_tokens = (
            sa_rdf["_prompt_tokens"].sum() + sa_rdf["_completion_tokens"].sum()
            if "_prompt_tokens" in sa_rdf.columns
            else 0
        )
        mcol1.metric("Successful", int(ok_count))
        mcol2.metric("Errors", int(err_count))
        mcol3.metric("Avg Duration", f"{avg_dur:.1f}s")
        mcol4.metric("Total Tokens", int(total_tokens))

        st.dataframe(sa_rdf, use_container_width=True, hide_index=True)

        # Pipeline logs
        logs = st.session_state.get("sa_pipeline_log", [])
        if logs:
            with st.expander("Pipeline Execution Details", expanded=False):
                for log_entry in logs:
                    row_idx = log_entry.get("row", "?")
                    steps = log_entry.get("steps", {})
                    search_info = steps.get("search", {})
                    crawl_info = steps.get("crawl", {})
                    analysis_info = steps.get("analysis", {})

                    st.markdown(f"**Row {row_idx}**")
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(
                            f"Search: {search_info.get('elapsed', '?')}s, "
                            f"{search_info.get('citations_found', 0)} citations"
                        )
                    with cols[1]:
                        st.caption(
                            f"Crawl: {crawl_info.get('elapsed', '?')}s, "
                            f"{crawl_info.get('urls_succeeded', 0)}/"
                            f"{crawl_info.get('urls_attempted', 0)} URLs"
                        )
                    with cols[2]:
                        st.caption(
                            f"Analysis: {analysis_info.get('elapsed', '?')}s"
                        )

                    if crawl_info.get("errors"):
                        with st.popover("Crawl errors"):
                            for ce in crawl_info["errors"]:
                                st.text(ce)

        # Export
        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "Download CSV",
                data=sa_rdf.to_csv(index=False),
                file_name="search_analyze_results.csv",
                mime="text/csv",
                key="sa_csv_dl",
            )
        with exp2:
            st.markdown("**Copy TSV** (paste into Excel)")
            st.code(sa_rdf.to_csv(sep="\t", index=False), language=None)
