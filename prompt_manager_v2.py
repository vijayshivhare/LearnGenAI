# prompt_manager_v1.py
import os
import yaml
from typing import Dict, List, Any

class PromptManagerV1:
    """Manages loading prompt templates from a YAML file for v1 application."""

    def __init__(self, config_path: str = "prompts_v1.yaml"):
        self.config_path = config_path
        self._prompts_data = self._load_prompts()

    def _load_prompts(self) -> Dict:
        """Loads the YAML config and returns the prompts dictionary."""
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"The prompt configuration file {self.config_path} does not exist.")
        try:
            with open(self.config_path, "r", encoding="UTF-8") as file:
                data = yaml.safe_load(file)
                if not isinstance(data, dict):
                    raise ValueError("Invalid YAML structure. Expected a dictionary.")
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get_general_chat_system_message(self) -> str:
        """Returns the general chat system message."""
        return self._prompts_data.get("general_chat", {}).get("system_message", "You are a helpful AI assistant.")

    def get_fraud_query_understanding_system_message(self) -> str:
        """Returns the system message for the FraudQueryUnderstandingTool."""
        return self._prompts_data.get("fraud_query_understanding", {}).get("system_message", "")

    def get_fraud_query_understanding_examples(self) -> List[Dict[str, str]]:
        """Returns example messages for the FraudQueryUnderstandingTool."""
        return self._prompts_data.get("fraud_query_understanding", {}).get("examples", [])

    def get_plot_code_generation_system_message(self) -> str:
        """Returns the system message for plot code generation."""
        return self._prompts_data.get("plot_code_generation", {}).get("system_message", "")

    def get_plot_code_generation_human_template(self) -> str:
        """Returns the human message template for plot code generation."""
        return self._prompts_data.get("plot_code_generation", {}).get("human_message_template", "")

    def get_general_code_writing_system_message(self) -> str:
        """Returns the system message for general code writing."""
        return self._prompts_data.get("general_code_writing", {}).get("system_message", "")

    def get_general_code_writing_human_template(self) -> str:
        """Returns the human message template for general code writing."""
        return self._prompts_data.get("general_code_writing", {}).get("human_message_template", "")

    def get_reasoning_curation_system_message(self) -> str:
        """Returns the system message for reasoning curation."""
        return self._prompts_data.get("reasoning_curation", {}).get("system_message", "")

    def get_reasoning_curation_human_template(self) -> str:
        """Returns the human message template for reasoning curation."""
        return self._prompts_data.get("reasoning_curation", {}).get("human_message_template", "")

    def get_rule_generation_system_message(self) -> str:
        """Returns the system message for rule generation."""
        return self._prompts_data.get("rule_generation", {}).get("system_message", "")

    def get_rule_generation_human_template(self) -> str:
        """Returns the human message template for rule generation."""
        return self._prompts_data.get("rule_generation", {}).get("human_message_template", "")