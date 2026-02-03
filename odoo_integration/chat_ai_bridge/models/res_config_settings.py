from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    chat_ai_api_base_url = fields.Char(
        string="Chat AI API Base URL",
        config_parameter="chat_ai_bridge.api_base_url",
    )
    chat_ai_api_key = fields.Char(
        string="Chat AI API Key",
        config_parameter="chat_ai_bridge.api_key",
    )
    chat_ai_default_model = fields.Char(
        string="Default Model",
        config_parameter="chat_ai_bridge.default_model",
    )
    chat_ai_use_smart_generation = fields.Boolean(
        string="Use Smart SQL Generation",
        config_parameter="chat_ai_bridge.use_smart_generation",
        default=True,
    )
