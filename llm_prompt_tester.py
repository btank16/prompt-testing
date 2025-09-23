import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import customtkinter as ctk
import json
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import threading
from typing import Optional, Dict, Any
import jsonschema
from jsonschema import validate, ValidationError

load_dotenv()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PerplexityAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, model: str, messages: list, temperature: float = 0.7) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Request failed: {str(e)}")


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
        input_label = ctk.CTkLabel(parent, text="Input Configuration",
                                  font=("Arial", 18, "bold"))
        input_label.pack(pady=10)

        api_frame = ctk.CTkFrame(parent)
        api_frame.pack(fill=tk.X, padx=10, pady=5)

        api_label = ctk.CTkLabel(api_frame, text="API Key Status:")
        api_label.pack(side=tk.LEFT, padx=5)

        self.api_status_label = ctk.CTkLabel(api_frame, text="Not Loaded",
                                            text_color="red")
        self.api_status_label.pack(side=tk.LEFT, padx=5)

        reload_btn = ctk.CTkButton(api_frame, text="Reload API Key",
                                   command=self.load_api_key, width=100)
        reload_btn.pack(side=tk.RIGHT, padx=5)

        model_frame = ctk.CTkFrame(parent)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        model_label = ctk.CTkLabel(model_frame, text="Model:")
        model_label.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar(value=self.available_models[0])
        self.model_dropdown = ctk.CTkComboBox(model_frame,
                                             values=self.available_models,
                                             variable=self.model_var,
                                             width=200)
        self.model_dropdown.pack(side=tk.LEFT, padx=5)

        prompt_label = ctk.CTkLabel(parent, text="Prompt Message:")
        prompt_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.prompt_text = ctk.CTkTextbox(parent, height=150)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        json_label = ctk.CTkLabel(parent, text="Expected JSON Format (Optional):")
        json_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.json_format_text = ctk.CTkTextbox(parent, height=150)
        self.json_format_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.json_format_text.insert("1.0", '{\n  "type": "object",\n  "properties": {\n    \n  }\n}')

        system_prompt_label = ctk.CTkLabel(parent, text="System Prompt (Optional):")
        system_prompt_label.pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.system_prompt_text = ctk.CTkTextbox(parent, height=100)
        self.system_prompt_text.pack(fill=tk.BOTH, padx=10, pady=5)
        self.system_prompt_text.insert("1.0", "You are a helpful assistant. Please respond in valid JSON format.")

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

            system_prompt = self.system_prompt_text.get("1.0", tk.END).strip()
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.api_client.chat_completion(
                model=self.model_var.get(),
                messages=messages,
                temperature=0.7
            )

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            self.root.after(0, self.update_response, response, response_time)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_response(self, response: Dict[str, Any], response_time: float):
        self.current_response = response

        self.raw_response_text.delete("1.0", tk.END)
        self.raw_response_text.insert("1.0", json.dumps(response, indent=2))

        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            self.response_text.delete("1.0", tk.END)
            self.response_text.insert("1.0", content)

            self.validate_json_response(content)

        if "usage" in response:
            usage = response["usage"]
            tokens_info = f"Tokens - Input: {usage.get('prompt_tokens', 0)}, " \
                         f"Output: {usage.get('completion_tokens', 0)}, " \
                         f"Total: {usage.get('total_tokens', 0)}"
            self.token_label.configure(text=tokens_info)

        self.time_label.configure(text=f"Response Time: {response_time:.2f}s")

        self.test_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": self.model_var.get(),
            "prompt": self.prompt_text.get("1.0", tk.END).strip(),
            "response": response,
            "response_time": response_time
        })

        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.test_button.configure(state="normal")

    def validate_json_response(self, content: str):
        try:
            json_response = json.loads(content)

            expected_format = self.json_format_text.get("1.0", tk.END).strip()
            if expected_format and expected_format != '{\n  "type": "object",\n  "properties": {\n    \n  }\n}':
                try:
                    schema = json.loads(expected_format)
                    validate(instance=json_response, schema=schema)
                    self.validation_label.configure(text="JSON Valid: ✓",
                                                  text_color="green")
                except ValidationError as e:
                    self.validation_label.configure(text=f"JSON Valid: ✗ - {str(e)[:30]}",
                                                  text_color="orange")
                except json.JSONDecodeError:
                    self.validation_label.configure(text="JSON Valid: ✗ - Invalid schema",
                                                  text_color="red")
            else:
                self.validation_label.configure(text="JSON Valid: ✓ (No schema)",
                                              text_color="green")

        except json.JSONDecodeError:
            self.validation_label.configure(text="JSON Valid: ✗ - Not valid JSON",
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
                "expected_json": self.json_format_text.get("1.0", tk.END).strip(),
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

                self.model_var.set(test_data.get("model", self.available_models[0]))
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert("1.0", test_data.get("prompt", ""))
                self.system_prompt_text.delete("1.0", tk.END)
                self.system_prompt_text.insert("1.0", test_data.get("system_prompt", ""))
                self.json_format_text.delete("1.0", tk.END)
                self.json_format_text.insert("1.0", test_data.get("expected_json", ""))

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