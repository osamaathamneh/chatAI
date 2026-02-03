import requests

from odoo import http
from odoo.http import request


class ChatAIBridgeController(http.Controller):
    @http.route("/chat_ai_bridge", type="http", auth="user")
    def chat_ai_page(self, **kwargs):
        return request.render("chat_ai_bridge.chat_ai_page", {})

    @http.route("/chat_ai_bridge/ask", type="json", auth="user")
    def ask(self, prompt=None):
        prompt = (prompt or "").strip()
        if not prompt:
            return {"error": "Prompt is required."}

        config = request.env["ir.config_parameter"].sudo()
        base_url = config.get_param("chat_ai_bridge.api_base_url")
        api_key = config.get_param("chat_ai_bridge.api_key")
        model = config.get_param("chat_ai_bridge.default_model")
        use_smart = config.get_param("chat_ai_bridge.use_smart_generation") == "True"

        if not base_url:
            return {"error": "Missing Chat AI API Base URL in settings."}

        endpoint = "api/smart-generate-sql" if use_smart else "api/generate-sql"
        url = f"{base_url.rstrip('/')}/{endpoint}"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {"prompt": prompt}
        if model:
            payload["model"] = model

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            return {"error": f"Chat AI request failed: {exc}"}

        return {"data": response.json()}
