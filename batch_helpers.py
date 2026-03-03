import streamlit as st
import pandas as pd
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import detect_variables, expand_json_to_rows


def render_prompt_inputs(tab_key):
    """Render system prompt, prompt template, variable detection, and JSON schema toggle.

    Returns dict: system_prompt, prompt_template, variables, use_schema, schema_text
    """
    system_prompt = st.text_area(
        "System Prompt",
        height=80,
        placeholder="Optional system instructions...",
        key=f"{tab_key}_system_prompt",
    )

    prompt_template = st.text_area(
        "Prompt Template",
        height=150,
        placeholder="Enter your prompt here. Use {variable} placeholders for batch variables.\n"
        "Example: Tell me about {city} in {country}",
        key=f"{tab_key}_prompt_template",
    )

    variables = detect_variables(prompt_template)
    if variables:
        st.caption("Detected variables: " + "  ".join([f"`{{{v}}}`" for v in variables]))

    use_schema = st.checkbox(
        "Use JSON schema for structured output",
        key=f"{tab_key}_use_schema",
    )
    schema_text = ""
    if use_schema:
        schema_text = st.text_area(
            "JSON Schema",
            height=200,
            placeholder='{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}',
            key=f"{tab_key}_schema_text",
        )

    return {
        "system_prompt": system_prompt,
        "prompt_template": prompt_template,
        "variables": variables,
        "use_schema": use_schema,
        "schema_text": schema_text,
    }


def render_variable_table(tab_key, variables):
    """Render the variable input data editor and CSV upload.

    Returns the edited DataFrame.
    """
    st.subheader("Variable Input Data")

    if variables:
        existing_df = st.session_state.get(f"{tab_key}_variable_df")
        if existing_df is not None and not existing_df.empty:
            for v in variables:
                if v not in existing_df.columns:
                    existing_df[v] = ""
            existing_df = existing_df[[v for v in variables if v in existing_df.columns]]
            df_input = existing_df
        else:
            df_input = pd.DataFrame({v: [""] for v in variables})

        # Bulk add rows + CSV upload
        rcol1, rcol2, rcol3 = st.columns([1, 1, 2])
        with rcol1:
            add_count = st.number_input(
                "Rows to add",
                min_value=1,
                max_value=500,
                value=5,
                key=f"{tab_key}_add_count",
            )
        with rcol2:
            st.markdown("<br>", unsafe_allow_html=True)  # align button with input
            if st.button(f"Add {add_count} rows", key=f"{tab_key}_add_rows"):
                new_rows = pd.DataFrame({v: [""] * int(add_count) for v in variables})
                df_input = pd.concat([df_input, new_rows], ignore_index=True)
                st.session_state[f"{tab_key}_variable_df"] = df_input
                st.rerun()
        with rcol3:
            csv_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key=f"{tab_key}_csv_upload",
            )
        if csv_file is not None:
            try:
                uploaded_df = pd.read_csv(csv_file, dtype=str).fillna("")
                missing_cols = [v for v in variables if v not in uploaded_df.columns]
                if missing_cols:
                    st.warning(
                        f"CSV missing columns: {', '.join(missing_cols)}. They will be added empty."
                    )
                    for mc in missing_cols:
                        uploaded_df[mc] = ""
                df_input = uploaded_df[[v for v in variables if v in uploaded_df.columns]]
            except Exception as e:
                st.error(f"CSV parse error: {e}")

        edited_df = st.data_editor(
            df_input,
            num_rows="dynamic",
            use_container_width=True,
            key=f"{tab_key}_var_editor",
        )
        st.session_state[f"{tab_key}_variable_df"] = edited_df
        return edited_df
    else:
        st.info(
            "Add `{variable}` placeholders to your prompt template to enable batch testing, "
            "or run a single call with no variables."
        )
        return pd.DataFrame({"_row": ["single call"]})


def render_run_controls(tab_key, edited_df, variables, delay, connected, save_config):
    """Render run button, row count caption, and save config download.

    Returns run_clicked bool.
    """
    num_rows = len(edited_df) if variables else 1
    st.caption(
        f"Total API calls: **{num_rows}** | Estimated time: ~{num_rows * (delay + 1):.0f}s"
    )

    col_run, col_save = st.columns([1, 1])
    with col_run:
        run_clicked = st.button(
            "Run Batch",
            type="primary",
            disabled=not connected,
            key=f"{tab_key}_run",
        )
    with col_save:
        st.download_button(
            "Save Config",
            data=json.dumps(save_config, indent=2),
            file_name=f"{tab_key}_config.json",
            mime="application/json",
            key=f"{tab_key}_save",
        )
    return run_clicked


_RETRYABLE_INDICATORS = (
    "503", "429", "overloaded", "rate limit", "resource exhausted",
    "quota", "service unavailable", "too many requests",
)


def _is_retryable(error_str):
    if not error_str:
        return False
    lower = error_str.lower()
    return any(s in lower for s in _RETRYABLE_INDICATORS)


def _call_with_retry(call_fn, idx, row_vars, max_retries=3, base_delay=2.0):
    """Wrap call_fn with exponential backoff for transient errors."""
    for attempt in range(max_retries + 1):
        result = call_fn(idx, row_vars)
        _, _, _, _, err = result
        if err is None or attempt == max_retries or not _is_retryable(err):
            return result
        wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
        time.sleep(wait)
    return result


def execute_batch(call_fn, rows, delay, max_workers, max_retries=3):
    """Execute batch API calls with progress bar and retry on transient errors.

    call_fn(idx, row_vars) should return (idx, row_vars, response, elapsed, error).
    Retries up to max_retries times on 503/429/overloaded errors with exponential backoff.
    Returns list of result tuples.
    """
    results = []
    progress = st.progress(0, text="Starting batch...")
    max_workers = int(max_workers)

    if max_workers <= 1:
        for i, row_vars in enumerate(rows):
            result = _call_with_retry(call_fn, i, row_vars, max_retries=max_retries)
            results.append(result)
            progress.progress((i + 1) / len(rows), text=f"Completed {i + 1}/{len(rows)}")
            if i < len(rows) - 1:
                time.sleep(delay)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, row_vars in enumerate(rows):
                fut = executor.submit(
                    _call_with_retry, call_fn, i, row_vars, max_retries
                )
                futures[fut] = i
                if i < len(rows) - 1:
                    time.sleep(delay / max_workers)

            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                completed += 1
                progress.progress(
                    completed / len(rows), text=f"Completed {completed}/{len(rows)}"
                )

    progress.progress(1.0, text="Done!")
    return results


def build_results_dataframe(results, variables, include_citations=False):
    """Build pandas DataFrame from batch results.

    Parallel arrays in JSON responses are expanded into multiple rows.
    Input variables and metadata are repeated on each expanded row so every
    row is traceable back to the API call that produced it.

    results: list of (idx, row_vars, response, elapsed, error) tuples.
    """
    result_rows = []
    for idx, row_vars, resp, elapsed, err in sorted(results, key=lambda x: x[0]):
        # Base columns shared by every row from this API call
        base = {}
        for v in variables:
            base[v] = row_vars.get(v, "")

        base["_status"] = "ok" if err is None else "error"
        base["_duration_s"] = round(elapsed, 2)
        base["_error"] = err or ""

        if resp:
            content = resp.get("content", "")
            usage = resp.get("usage", {})
            base["_prompt_tokens"] = usage.get("prompt_tokens", 0)
            base["_completion_tokens"] = usage.get("completion_tokens", 0)

            if include_citations:
                citations = resp.get("citations") or []
                base["_citations"] = "; ".join(citations) if citations else ""
                base["_num_sources"] = len(citations)

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

    return pd.DataFrame(result_rows)


def render_results(tab_key, include_citations=False):
    """Render results metrics, dataframe, and export buttons.

    Reads from st.session_state[f"{tab_key}_results_df"].
    """
    rdf = st.session_state.get(f"{tab_key}_results_df")
    if rdf is None:
        return

    st.divider()
    st.subheader("Results")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    ok_count = (rdf["_status"] == "ok").sum() if "_status" in rdf.columns else 0
    err_count = (rdf["_status"] == "error").sum() if "_status" in rdf.columns else 0
    avg_dur = rdf["_duration_s"].mean() if "_duration_s" in rdf.columns else 0
    total_tokens = (
        rdf["_prompt_tokens"].sum() + rdf["_completion_tokens"].sum()
        if "_prompt_tokens" in rdf.columns
        else 0
    )
    mcol1.metric("Successful", int(ok_count))
    mcol2.metric("Errors", int(err_count))
    mcol3.metric("Avg Duration", f"{avg_dur:.1f}s")
    mcol4.metric("Total Tokens", int(total_tokens))

    st.dataframe(rdf, use_container_width=True, hide_index=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "Download CSV",
            data=rdf.to_csv(index=False),
            file_name=f"{tab_key}_results.csv",
            mime="text/csv",
            key=f"{tab_key}_csv_dl",
        )
    with exp2:
        st.markdown("**Copy TSV** (paste into Excel)")
        st.code(rdf.to_csv(sep="\t", index=False), language=None)
