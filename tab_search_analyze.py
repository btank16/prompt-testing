"""Search & Analyze tab — extracted from app.py.

Pipeline: Perplexity search → crawl URLs with crawl4ai → analyze with LLM.
"""

import streamlit as st
import pandas as pd
import json
import time

from utils import (
    detect_variables,
    substitute_variables,
    build_prompt,
    parse_schema,
    dataframe_to_rows,
    normalize_response,
    expand_json_to_rows,
)
from llm_helpers import call_llm
from batch_helpers import execute_batch
from content_extractor import extract_website


def render_search_analyze_tab(delay_between, max_workers):
    """Render the full Search & Analyze tab UI and handle pipeline execution."""

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
            try:
                sa_analysis_schema, _ = parse_schema(sa_use_schema, sa_schema_text)
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

            rows = dataframe_to_rows(sa_edited_df, bool(user_sa_vars))

            # Resolve provider name for call_llm
            provider_lower = sa_analysis_provider.lower()

            # Analysis LLM params
            analysis_params = {
                "temperature": sa_analysis_temp,
                "max_tokens": sa_analysis_max_tokens,
                "top_p": sa_analysis_top_p,
            }
            # GPT-5 models need reasoning_effort instead of temperature
            if provider_lower == "openai" and sa_analysis_model.startswith("gpt-5"):
                analysis_params = {
                    "max_tokens": sa_analysis_max_tokens,
                    "reasoning_effort": "medium",
                }

            # --- Pipeline function for each row ---
            def _sa_pipeline(idx, row_vars):
                pipeline_start = time.time()
                log = {"row": idx, "steps": {}}

                # === STEP 1: Perplexity Search ===
                search_prompt = build_prompt(sa_search_template, row_vars)

                search_params = {
                    "search_domain_filter": sa_domain_list,
                    "search_recency_filter": sa_recency_val,
                    "search_context_size": sa_context_size,
                    "temperature": sa_search_temp,
                }

                step1_start = time.time()
                try:
                    search_resp = call_llm(
                        "perplexity", pplx_client, sa_search_model,
                        search_prompt, sa_search_system if sa_search_system.strip() else None,
                        params=search_params,
                    )
                    search_content = search_resp.get("content", "")
                    citations = search_resp.get("citations") or []
                    log["steps"]["search"] = {
                        "status": "ok",
                        "elapsed": round(time.time() - step1_start, 2),
                        "citations_found": len(citations),
                    }
                except Exception as e:
                    return (
                        idx, row_vars, None,
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
                analysis_vars = dict(row_vars) if row_vars else {}
                analysis_vars["crawled_content"] = combined_crawled
                analysis_vars["search_response"] = search_content
                analysis_vars["search_citations"] = "; ".join(citations)

                analysis_prompt = substitute_variables(
                    sa_analysis_template, analysis_vars
                )

                step3_start = time.time()
                try:
                    resp = call_llm(
                        provider_lower, analysis_client, sa_analysis_model,
                        analysis_prompt,
                        sa_analysis_system if sa_analysis_system.strip() else None,
                        sa_analysis_schema, analysis_params,
                    )

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
                        idx, row_vars, resp,
                        time.time() - pipeline_start, None,
                    )
                except Exception as e:
                    return (
                        idx, row_vars, None,
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
