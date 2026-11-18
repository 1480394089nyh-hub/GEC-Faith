# -*- coding: utf-8 -*-
"""模型与 API 配置模板。

把本文件复制为 config.py，并填入真实的 api_key 后即可使用。
【重要】config.py 已被 .gitignore 排除，真实的 key 不会进 git；
        提交到 GitHub 前务必确认仓库里没有 config.py、没有泄露任何 api_key。
"""

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "api_key": "",   # ← 填你的 key
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
        "api_key": "",   # ← 填你的 key
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "api_key": "",   # ← 填你的 key
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "",
    },
}
