import streamlit as st
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv

from gemini_client import GeminiClient
from perplexity_client import PerplexityAPIClient
from openai_client import OpenAIClient
from utils import (
    detect_variables,
    substitute_variables,
    extract_schema_from_stored_format,
    normalize_response,
    detect_provider_from_model,
    build_response_format,
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
tab_gemini, tab_perplexity, tab_openai = st.tabs(["Gemini Batch", "Perplexity", "OpenAI"])

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
