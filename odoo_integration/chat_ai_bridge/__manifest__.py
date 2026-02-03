{
    "name": "Chat AI Bridge",
    "version": "18.0.1.0.0",
    "summary": "Integrate external Chat AI service with Odoo 18",
    "category": "Productivity",
    "author": "Chat AI",
    "website": "https://example.com",
    "license": "LGPL-3",
    "depends": ["base", "web"],
    "data": [
        "views/res_config_settings_view.xml",
        "views/chat_ai_templates.xml",
        "views/chat_ai_menu.xml",
    ],
    "assets": {
        "web.assets_backend": [
            "chat_ai_bridge/static/src/js/chat_ai_bridge.js",
            "chat_ai_bridge/static/src/css/chat_ai_bridge.css",
        ],
    },
    "application": True,
    "installable": True,
}
