import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import customtkinter as ctk
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import threading
from typing import Dict, Any
import jsonschema
from jsonschema import validate, ValidationError
from perplexity_client import PerplexityAPIClient

load_dotenv()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LLMPromptTesterGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("LLM Prompt Tester - Perplexity Sonar API")
        self.root.geometry("1200x800")

        self.api_client = None
        self.current_response = None
        self.test_history = []

        self.available_models = [
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-deep-research"
        ]

        self.setup_ui()
        self.load_api_key()

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.setup_input_panel(left_panel)
        self.setup_output_panel(right_panel)

        self.setup_control_buttons()

    def setup_input_panel(self, parent):
        # Create scrollable frame for input panel
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        input_label = ctk.CTkLabel(scroll_frame, text="Input Configuration",
                                  font=("Arial", 18, "bold"))
        input_label.pack(pady=10)

        # API Configuration
        api_frame = ctk.CTkFrame(scroll_frame)
        api_frame.pack(fill=tk.X, padx=10, pady=5)

        api_label = ctk.CTkLabel(api_frame, text="API Key Status:")
        api_label.pack(side=tk.LEFT, padx=5)

        self.api_status_label = ctk.CTkLabel(api_frame, text="Not Loaded",
                                            text_color="red")
        self.api_status_label.pack(side=tk.LEFT, padx=5)

        reload_btn = ctk.CTkButton(api_frame, text="Reload API Key",
                                   command=self.load_api_key, width=100)
        reload_btn.pack(side=tk.RIGHT, padx=5)

        # Model Selection
        model_frame = ctk.CTkFrame(scroll_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        model_label = ctk.CTkLabel(model_frame, text="Model:")
        model_label.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar(value=self.available_models[0])
        self.model_dropdown = ctk.CTkComboBox(model_frame,
                                             values=self.available_models,
                                             variable=self.model_var,
                                             width=200)
        self.model_dropdown.pack(side=tk.LEFT, padx=5)

        # Basic Prompts
        prompt_label = ctk.CTkLabel(scroll_frame, text="Prompt Message:")
        prompt_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.prompt_text = ctk.CTkTextbox(scroll_frame, height=120)
        self.prompt_text.pack(fill=tk.BOTH, padx=10, pady=5)

        system_prompt_label = ctk.CTkLabel(scroll_frame, text="System Prompt (Optional):")
        system_prompt_label.pack(anchor=tk.W, padx=10, pady=(5, 0))

        self.system_prompt_text = ctk.CTkTextbox(scroll_frame, height=80)
        self.system_prompt_text.pack(fill=tk.BOTH, padx=10, pady=5)

        # Search Parameters Section
        search_params_label = ctk.CTkLabel(scroll_frame, text="Search Parameters",
                                          font=("Arial", 14, "bold"))
        search_params_label.pack(anchor=tk.W, padx=10, pady=(15, 5))

        # URL Reference
        url_label = ctk.CTkLabel(scroll_frame, text="Reference URL (Optional):")
        url_label.pack(anchor=tk.W, padx=10, pady=(5, 0))

        self.url_entry = ctk.CTkEntry(scroll_frame, placeholder_text="https://example.com")
        self.url_entry.pack(fill=tk.X, padx=10, pady=5)

        # Domain Filtering
        domain_filter_label = ctk.CTkLabel(scroll_frame, text="Domain Filter (max 3, prefix '-' to exclude):")
        domain_filter_label.pack(anchor=tk.W, padx=10, pady=(5, 0))

        self.domain_filter_entry = ctk.CTkEntry(scroll_frame,
                                                placeholder_text="example.com, -spam.com, docs.com")
        self.domain_filter_entry.pack(fill=tk.X, padx=10, pady=5)

        # Search Recency and Context
        search_options_frame = ctk.CTkFrame(scroll_frame)
        search_options_frame.pack(fill=tk.X, padx=10, pady=5)

        recency_label = ctk.CTkLabel(search_options_frame, text="Recency Filter:")
        recency_label.grid(row=0, column=0, padx=5, sticky="w")

        self.recency_var = tk.StringVar(value="none")
        self.recency_dropdown = ctk.CTkComboBox(search_options_frame,
                                                values=["none", "hour", "day", "week", "month"],
                                                variable=self.recency_var,
                                                width=120)
        self.recency_dropdown.grid(row=0, column=1, padx=5)

        context_label = ctk.CTkLabel(search_options_frame, text="Context Size:")
        context_label.grid(row=0, column=2, padx=(20, 5), sticky="w")

        self.context_var = tk.StringVar(value="medium")
        self.context_dropdown = ctk.CTkComboBox(search_options_frame,
                                                values=["low", "medium", "high"],
                                                variable=self.context_var,
                                                width=120)
        self.context_dropdown.grid(row=0, column=3, padx=5)

        # Date Filters
        date_filter_frame = ctk.CTkFrame(scroll_frame)
        date_filter_frame.pack(fill=tk.X, padx=10, pady=5)

        after_date_label = ctk.CTkLabel(date_filter_frame, text="After Date (MM/DD/YYYY):")
        after_date_label.grid(row=0, column=0, padx=5, sticky="w")

        self.after_date_entry = ctk.CTkEntry(date_filter_frame,
                                             placeholder_text="01/01/2025",
                                             width=120)
        self.after_date_entry.grid(row=0, column=1, padx=5)

        before_date_label = ctk.CTkLabel(date_filter_frame, text="Before Date:")
        before_date_label.grid(row=0, column=2, padx=(20, 5), sticky="w")

        self.before_date_entry = ctk.CTkEntry(date_filter_frame,
                                              placeholder_text="12/31/2025",
                                              width=120)
        self.before_date_entry.grid(row=0, column=3, padx=5)

        # Search Options Checkboxes
        options_frame = ctk.CTkFrame(scroll_frame)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.return_images_var = tk.BooleanVar(value=False)
        self.return_images_check = ctk.CTkCheckBox(options_frame,
                                                   text="Return Images",
                                                   variable=self.return_images_var)
        self.return_images_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.return_questions_var = tk.BooleanVar(value=False)
        self.return_questions_check = ctk.CTkCheckBox(options_frame,
                                                      text="Return Related Questions",
                                                      variable=self.return_questions_var)
        self.return_questions_check.grid(row=0, column=1, padx=20, pady=5, sticky="w")

        # Location Settings
        location_label = ctk.CTkLabel(scroll_frame, text="Location (Optional):")
        location_label.pack(anchor=tk.W, padx=10, pady=(5, 0))

        location_frame = ctk.CTkFrame(scroll_frame)
        location_frame.pack(fill=tk.X, padx=10, pady=5)

        lat_label = ctk.CTkLabel(location_frame, text="Latitude:")
        lat_label.grid(row=0, column=0, padx=5, sticky="w")

        self.latitude_entry = ctk.CTkEntry(location_frame, placeholder_text="37.7749", width=100)
        self.latitude_entry.grid(row=0, column=1, padx=5)

        lon_label = ctk.CTkLabel(location_frame, text="Longitude:")
        lon_label.grid(row=0, column=2, padx=(15, 5), sticky="w")

        self.longitude_entry = ctk.CTkEntry(location_frame, placeholder_text="-122.4194", width=100)
        self.longitude_entry.grid(row=0, column=3, padx=5)

        country_label = ctk.CTkLabel(location_frame, text="Country:")
        country_label.grid(row=0, column=4, padx=(15, 5), sticky="w")

        self.country_entry = ctk.CTkEntry(location_frame, placeholder_text="US", width=60)
        self.country_entry.grid(row=0, column=5, padx=5)

        # LLM Parameters Section
        llm_params_label = ctk.CTkLabel(scroll_frame, text="LLM Parameters",
                                       font=("Arial", 14, "bold"))
        llm_params_label.pack(anchor=tk.W, padx=10, pady=(15, 5))

        # Temperature and Max Tokens
        llm_frame1 = ctk.CTkFrame(scroll_frame)
        llm_frame1.pack(fill=tk.X, padx=10, pady=5)

        temp_label = ctk.CTkLabel(llm_frame1, text="Temperature (0-2):")
        temp_label.grid(row=0, column=0, padx=5, sticky="w")

        self.temperature_slider = ctk.CTkSlider(llm_frame1, from_=0, to=2, number_of_steps=20, width=150)
        self.temperature_slider.set(0.2)
        self.temperature_slider.grid(row=0, column=1, padx=5)

        self.temp_value_label = ctk.CTkLabel(llm_frame1, text="0.2")
        self.temp_value_label.grid(row=0, column=2, padx=5)

        self.temperature_slider.configure(command=lambda v: self.temp_value_label.configure(text=f"{v:.1f}"))

        max_tokens_label = ctk.CTkLabel(llm_frame1, text="Max Tokens:")
        max_tokens_label.grid(row=0, column=3, padx=(20, 5), sticky="w")

        self.max_tokens_entry = ctk.CTkEntry(llm_frame1, placeholder_text="1000", width=80)
        self.max_tokens_entry.grid(row=0, column=4, padx=5)

        # Top-p and Penalties
        llm_frame2 = ctk.CTkFrame(scroll_frame)
        llm_frame2.pack(fill=tk.X, padx=10, pady=5)

        top_p_label = ctk.CTkLabel(llm_frame2, text="Top-p (0-1):")
        top_p_label.grid(row=0, column=0, padx=5, sticky="w")

        self.top_p_entry = ctk.CTkEntry(llm_frame2, placeholder_text="1.0", width=60)
        self.top_p_entry.grid(row=0, column=1, padx=5)

        freq_penalty_label = ctk.CTkLabel(llm_frame2, text="Frequency Penalty (-2 to 2):")
        freq_penalty_label.grid(row=0, column=2, padx=(15, 5), sticky="w")

        self.freq_penalty_entry = ctk.CTkEntry(llm_frame2, placeholder_text="0", width=60)
        self.freq_penalty_entry.grid(row=0, column=3, padx=5)

        pres_penalty_label = ctk.CTkLabel(llm_frame2, text="Presence Penalty:")
        pres_penalty_label.grid(row=0, column=4, padx=(15, 5), sticky="w")

        self.pres_penalty_entry = ctk.CTkEntry(llm_frame2, placeholder_text="0", width=60)
        self.pres_penalty_entry.grid(row=0, column=5, padx=5)

        # Response Format Section
        response_format_label = ctk.CTkLabel(scroll_frame, text="JSON Response Format",
                                            font=("Arial", 14, "bold"))
        response_format_label.pack(anchor=tk.W, padx=10, pady=(15, 5))

        json_type_frame = ctk.CTkFrame(scroll_frame)
        json_type_frame.pack(fill=tk.X, padx=10, pady=5)

        self.use_json_var = tk.BooleanVar(value=False)
        self.use_json_check = ctk.CTkCheckBox(json_type_frame,
                                              text="Request JSON Response",
                                              variable=self.use_json_var,
                                              command=self.toggle_json_input)
        self.use_json_check.pack(side=tk.LEFT, padx=5)

        json_label = ctk.CTkLabel(scroll_frame, text="Custom JSON Format (when JSON response is enabled):")
        json_label.pack(anchor=tk.W, padx=10, pady=(5, 0))

        self.json_format_text = ctk.CTkTextbox(scroll_frame, height=100)
        self.json_format_text.pack(fill=tk.BOTH, padx=10, pady=5)
        # Insert placeholder example for Perplexity API response_format
        placeholder_json = '''{
  "type": "json_schema",
  "json_schema": {
    "schema": {
      "type": "object",
      "properties": {
        "answer": {
          "type": "number"
        }
      },
      "required": ["answer"]
    }
  }
}'''
        self.json_format_text.insert("1.0", placeholder_json)
        self.json_format_text.configure(state="disabled")

    def setup_output_panel(self, parent):
        output_label = ctk.CTkLabel(parent, text="Output Response",
                                   font=("Arial", 18, "bold"))
        output_label.pack(pady=10)

        stats_frame = ctk.CTkFrame(parent)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.token_label = ctk.CTkLabel(stats_frame, text="Tokens: N/A")
        self.token_label.pack(side=tk.LEFT, padx=10)

        self.time_label = ctk.CTkLabel(stats_frame, text="Response Time: N/A")
        self.time_label.pack(side=tk.LEFT, padx=10)

        self.validation_label = ctk.CTkLabel(stats_frame, text="JSON Valid: N/A")
        self.validation_label.pack(side=tk.LEFT, padx=10)

        response_label = ctk.CTkLabel(parent, text="Response:")
        response_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.response_text = ctk.CTkTextbox(parent, height=300)
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        raw_label = ctk.CTkLabel(parent, text="Raw API Response:")
        raw_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.raw_response_text = ctk.CTkTextbox(parent, height=200)
        self.raw_response_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def toggle_json_input(self):
        """Enable/disable JSON format input based on checkbox"""
        if self.use_json_var.get():
            self.json_format_text.configure(state="normal")
        else:
            self.json_format_text.configure(state="disabled")

    def setup_control_buttons(self):
        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.test_button = ctk.CTkButton(button_frame, text="Run Test",
                                        command=self.run_test,
                                        width=150, height=40)
        self.test_button.pack(side=tk.LEFT, padx=5)

        clear_button = ctk.CTkButton(button_frame, text="Clear All",
                                    command=self.clear_all,
                                    width=100, height=40)
        clear_button.pack(side=tk.LEFT, padx=5)

        save_button = ctk.CTkButton(button_frame, text="Save Test",
                                   command=self.save_test,
                                   width=100, height=40)
        save_button.pack(side=tk.LEFT, padx=5)

        load_button = ctk.CTkButton(button_frame, text="Load Test",
                                   command=self.load_test,
                                   width=100, height=40)
        load_button.pack(side=tk.LEFT, padx=5)

        export_button = ctk.CTkButton(button_frame, text="Export Results",
                                     command=self.export_results,
                                     width=120, height=40)
        export_button.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ctk.CTkProgressBar(button_frame, width=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.set(0)

    def load_api_key(self):
        api_key = os.getenv("PERPLEXITY_API_KEY")

        if not api_key:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if not os.path.exists(env_path):
                messagebox.showwarning("API Key Missing",
                                     "Please create a .env file with PERPLEXITY_API_KEY=your_key")
                self.api_status_label.configure(text="Not Found", text_color="red")
                return

        if api_key:
            self.api_client = PerplexityAPIClient(api_key)
            self.api_status_label.configure(text="Loaded", text_color="green")
        else:
            self.api_status_label.configure(text="Not Found", text_color="red")

    def run_test(self):
        if not self.api_client:
            messagebox.showerror("Error", "Please configure your API key first")
            return

        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt")
            return

        self.test_button.configure(state="disabled")
        self.progress_bar.set(0.5)
        self.progress_bar.start()

        thread = threading.Thread(target=self.execute_api_call, args=(prompt,))
        thread.daemon = True
        thread.start()

    def execute_api_call(self, prompt: str):
        try:
            start_time = datetime.now()

            # Prepare messages
            system_prompt = self.system_prompt_text.get("1.0", tk.END).strip()
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            # Prepare response format (JSON only)
            response_format = None
            if self.use_json_var.get():
                json_str = self.json_format_text.get("1.0", tk.END).strip()
                if json_str and json_str != '{}':
                    try:
                        # Parse the JSON format configuration
                        json_config = json.loads(json_str)
                        # Pass the entire configuration as response_format
                        response_format = json_config
                    except json.JSONDecodeError:
                        pass

            # Get URL
            url = self.url_entry.get().strip()
            if not url:
                url = None

            # Prepare search domain filter
            search_domain_filter = None
            domain_filter_str = self.domain_filter_entry.get().strip()
            if domain_filter_str:
                domains = [d.strip() for d in domain_filter_str.split(',') if d.strip()]
                if domains:
                    search_domain_filter = domains[:3]  # Limit to 3 domains

            # Get search recency filter
            search_recency_filter = None
            recency_value = self.recency_var.get()
            if recency_value != "none":
                search_recency_filter = recency_value

            # Get date filters
            search_after_date = self.after_date_entry.get().strip() or None
            search_before_date = self.before_date_entry.get().strip() or None

            # Get search context size
            search_context_size = self.context_var.get()

            # Get boolean options
            return_images = self.return_images_var.get() or None
            return_related_questions = self.return_questions_var.get() or None

            # Get location
            user_location = None
            lat = self.latitude_entry.get().strip()
            lon = self.longitude_entry.get().strip()
            if lat and lon:
                try:
                    user_location = {
                        "latitude": float(lat),
                        "longitude": float(lon)
                    }
                    country = self.country_entry.get().strip()
                    if country:
                        user_location["country"] = country
                except ValueError:
                    pass

            # Get LLM parameters
            temperature = self.temperature_slider.get()

            max_tokens = None
            max_tokens_str = self.max_tokens_entry.get().strip()
            if max_tokens_str:
                try:
                    max_tokens = int(max_tokens_str)
                except ValueError:
                    pass

            top_p = None
            top_p_str = self.top_p_entry.get().strip()
            if top_p_str:
                try:
                    top_p = float(top_p_str)
                except ValueError:
                    pass

            frequency_penalty = None
            freq_penalty_str = self.freq_penalty_entry.get().strip()
            if freq_penalty_str:
                try:
                    frequency_penalty = float(freq_penalty_str)
                except ValueError:
                    pass

            presence_penalty = None
            pres_penalty_str = self.pres_penalty_entry.get().strip()
            if pres_penalty_str:
                try:
                    presence_penalty = float(pres_penalty_str)
                except ValueError:
                    pass

            # Call API with all parameters
            response = self.api_client.chat_completion(
                model=self.model_var.get(),
                messages=messages,
                response_format=response_format,
                url=url,
                search_domain_filter=search_domain_filter,
                search_recency_filter=search_recency_filter,
                search_after_date_filter=search_after_date,
                search_before_date_filter=search_before_date,
                search_context_size=search_context_size,
                return_images=return_images,
                return_related_questions=return_related_questions,
                user_location=user_location,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False
            )

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            self.root.after(0, self.update_response, response, response_time)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_response(self, response: Dict[str, Any], response_time: float):
        self.current_response = response

        # Display raw response
        self.raw_response_text.delete("1.0", tk.END)
        self.raw_response_text.insert("1.0", json.dumps(response, indent=2))

        # Process main response content
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]

            # Build display text
            display_text = content

            # Add search results if available
            if "search_results" in response and response["search_results"]:
                display_text += "\n\n--- Search Results Used ---\n"
                for i, result in enumerate(response["search_results"], 1):
                    display_text += f"\n{i}. {result.get('title', 'Untitled')}\n"
                    display_text += f"   URL: {result.get('url', 'N/A')}\n"
                    if 'published_date' in result:
                        display_text += f"   Published: {result['published_date']}\n"

            # Add related questions if available
            if "related_questions" in response and response["related_questions"]:
                display_text += "\n\n--- Related Questions ---\n"
                for i, question in enumerate(response["related_questions"], 1):
                    display_text += f"{i}. {question}\n"

            self.response_text.delete("1.0", tk.END)
            self.response_text.insert("1.0", display_text)

            # Validate JSON if applicable
            if self.use_json_var.get():
                self.validate_json_response(content)

        # Update usage statistics
        if "usage" in response:
            usage = response["usage"]
            tokens_info = f"Tokens - Input: {usage.get('prompt_tokens', 0)}, " \
                         f"Output: {usage.get('completion_tokens', 0)}, " \
                         f"Total: {usage.get('total_tokens', 0)}"

            # Add search results count if available
            if "search_results" in response:
                tokens_info += f", Sources: {len(response.get('search_results', []))}"

            self.token_label.configure(text=tokens_info)

        self.time_label.configure(text=f"Response Time: {response_time:.2f}s")

        # Save to history with all parameters
        self.test_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": self.model_var.get(),
            "prompt": self.prompt_text.get("1.0", tk.END).strip(),
            "url": self.url_entry.get().strip(),
            "domain_filter": self.domain_filter_entry.get().strip(),
            "recency_filter": self.recency_var.get(),
            "context_size": self.context_var.get(),
            "response": response,
            "response_time": response_time
        })

        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.test_button.configure(state="normal")

    def validate_json_response(self, content: str):
        try:
            json_response = json.loads(content)
            self.validation_label.configure(text="JSON Valid: ✓",
                                          text_color="green")

            # Also validate against expected format if provided
            expected_format = self.json_format_text.get("1.0", tk.END).strip()
            if expected_format and expected_format != '{}':
                try:
                    expected = json.loads(expected_format)
                    # Simple structure validation - check if keys match
                    if isinstance(expected, dict) and isinstance(json_response, dict):
                        missing_keys = set(expected.keys()) - set(json_response.keys())
                        if missing_keys:
                            self.validation_label.configure(
                                text=f"JSON Valid: ⚠ Missing keys: {', '.join(list(missing_keys)[:3])}",
                                text_color="orange")
                        else:
                            self.validation_label.configure(text="JSON Valid: ✓ Structure matches",
                                                          text_color="green")
                except json.JSONDecodeError:
                    self.validation_label.configure(text="JSON Valid: ✓ (Invalid expected format)",
                                                  text_color="yellow")

        except json.JSONDecodeError:
            self.validation_label.configure(text="JSON Valid: ✗ - Response is not JSON",
                                          text_color="red")

    def show_error(self, error_message: str):
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.test_button.configure(state="normal")
        messagebox.showerror("API Error", error_message)

    def clear_all(self):
        self.prompt_text.delete("1.0", tk.END)
        self.response_text.delete("1.0", tk.END)
        self.raw_response_text.delete("1.0", tk.END)
        self.url_entry.delete(0, tk.END)
        self.token_label.configure(text="Tokens: N/A")
        self.time_label.configure(text="Response Time: N/A")
        self.validation_label.configure(text="JSON Valid: N/A", text_color="white")

    def save_test(self):
        if not self.current_response:
            messagebox.showwarning("Warning", "No test to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            test_data = {
                "model": self.model_var.get(),
                "prompt": self.prompt_text.get("1.0", tk.END).strip(),
                "system_prompt": self.system_prompt_text.get("1.0", tk.END).strip(),
                "url": self.url_entry.get().strip(),
                "search_params": {
                    "domain_filter": self.domain_filter_entry.get().strip(),
                    "recency_filter": self.recency_var.get(),
                    "context_size": self.context_var.get(),
                    "after_date": self.after_date_entry.get().strip(),
                    "before_date": self.before_date_entry.get().strip(),
                    "return_images": self.return_images_var.get(),
                    "return_questions": self.return_questions_var.get()
                },
                "location": {
                    "latitude": self.latitude_entry.get().strip(),
                    "longitude": self.longitude_entry.get().strip(),
                    "country": self.country_entry.get().strip()
                },
                "llm_params": {
                    "temperature": self.temperature_slider.get(),
                    "max_tokens": self.max_tokens_entry.get().strip(),
                    "top_p": self.top_p_entry.get().strip(),
                    "frequency_penalty": self.freq_penalty_entry.get().strip(),
                    "presence_penalty": self.pres_penalty_entry.get().strip()
                },
                "use_json": self.use_json_var.get(),
                "expected_json": self.json_format_text.get("1.0", tk.END).strip() if self.use_json_var.get() else "",
                "response": self.current_response,
                "timestamp": datetime.now().isoformat()
            }

            with open(file_path, 'w') as f:
                json.dump(test_data, f, indent=2)

            messagebox.showinfo("Success", f"Test saved to {file_path}")

    def load_test(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    test_data = json.load(f)

                # Load basic settings
                self.model_var.set(test_data.get("model", self.available_models[0]))
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert("1.0", test_data.get("prompt", ""))
                self.system_prompt_text.delete("1.0", tk.END)
                self.system_prompt_text.insert("1.0", test_data.get("system_prompt", ""))
                self.url_entry.delete(0, tk.END)
                self.url_entry.insert(0, test_data.get("url", ""))

                # Load search parameters
                search_params = test_data.get("search_params", {})
                self.domain_filter_entry.delete(0, tk.END)
                self.domain_filter_entry.insert(0, search_params.get("domain_filter", ""))
                self.recency_var.set(search_params.get("recency_filter", "none"))
                self.context_var.set(search_params.get("context_size", "medium"))
                self.after_date_entry.delete(0, tk.END)
                self.after_date_entry.insert(0, search_params.get("after_date", ""))
                self.before_date_entry.delete(0, tk.END)
                self.before_date_entry.insert(0, search_params.get("before_date", ""))
                self.return_images_var.set(search_params.get("return_images", False))
                self.return_questions_var.set(search_params.get("return_questions", False))

                # Load location
                location = test_data.get("location", {})
                self.latitude_entry.delete(0, tk.END)
                self.latitude_entry.insert(0, location.get("latitude", ""))
                self.longitude_entry.delete(0, tk.END)
                self.longitude_entry.insert(0, location.get("longitude", ""))
                self.country_entry.delete(0, tk.END)
                self.country_entry.insert(0, location.get("country", ""))

                # Load LLM parameters
                llm_params = test_data.get("llm_params", {})
                if "temperature" in llm_params:
                    self.temperature_slider.set(llm_params["temperature"])
                self.max_tokens_entry.delete(0, tk.END)
                self.max_tokens_entry.insert(0, llm_params.get("max_tokens", ""))
                self.top_p_entry.delete(0, tk.END)
                self.top_p_entry.insert(0, llm_params.get("top_p", ""))
                self.freq_penalty_entry.delete(0, tk.END)
                self.freq_penalty_entry.insert(0, llm_params.get("frequency_penalty", ""))
                self.pres_penalty_entry.delete(0, tk.END)
                self.pres_penalty_entry.insert(0, llm_params.get("presence_penalty", ""))

                # Load JSON settings
                self.use_json_var.set(test_data.get("use_json", False))
                self.toggle_json_input()
                if self.use_json_var.get():
                    self.json_format_text.delete("1.0", tk.END)
                    self.json_format_text.insert("1.0", test_data.get("expected_json", "{}"))

                messagebox.showinfo("Success", "Test loaded successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test: {str(e)}")

    def export_results(self):
        if not self.test_history:
            messagebox.showwarning("Warning", "No test history to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.test_history, f, indent=2)

            messagebox.showinfo("Success", f"Results exported to {file_path}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = LLMPromptTesterGUI()
    app.run()