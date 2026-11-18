# -*- coding: utf-8 -*-
"""Model and API configuration template.

Copy this file to config.py and fill in your real api_key.
IMPORTANT: config.py is excluded by .gitignore, so real keys never enter git;
always make sure no api_key is leaked before pushing to GitHub.
"""

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "api_key": "",   # <- fill in your key
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
        "api_key": "",   # <- fill in your key
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "api_key": "",   # <- fill in your key
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "",
    },
}
