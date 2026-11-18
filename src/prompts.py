# -*- coding: utf-8 -*-
"""Prompt strategy templates.

Each strategy is a function: strategy(call, src) -> corrected sentence.
  - call: a function call(messages) that calls the LLM and returns the reply text.
  - src: the original sentence to be corrected.
"""

SYSTEM = "你是一名专业的中文语法纠错助手。"

# Few-shot correction examples (error sentence -> corrected sentence)
EXAMPLES = [
    ("请把窗户观上", "请把窗户关上"),   # typo
    ("我吃饭在食堂", "我在食堂吃饭"),   # word order
    ("他跑步很快地", "他跑得很快"),     # word choice / collocation
]


def _clean(text):
    """Strip prefixes, explanations, newlines and quotes the model may add."""
    text = (text or "").strip()
    text = text.replace("\n", "").replace("\r", "").replace("\t", "")
    # Drop the "explanation + 改为" pattern; keep text after markers like 改为/改正.
    for marker in ("改为", "改正后", "改正为", "修改后", "应为"):
        if marker in text:
            tail = text.rsplit(marker, 1)[1].lstrip("：:，, ")
            if tail:
                text = tail
                break
    text = text.strip().strip("\"'“”")
    return text


# Strategy 1: direct correction (baseline)
def direct(call, src):
    out = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"请改正下面这句话中的语法错误。只输出改正后的完整句子本身，不要任何解释、说明、前缀或引号：\n{src}"},
    ])
    return _clean(out)


# Strategy 2: few-shot (with examples)
def few_shot(call, src):
    content = "下面是一些中文改错示例：\n"
    for a, b in EXAMPLES:
        content += f"错句：{a}\n改对：{b}\n"
    content += f"\n请改正：{src}\n只输出改正后的完整句子，不要任何解释。"
    out = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ])
    return _clean(out)


# Strategy 3: chain-of-thought (CoT)
def cot(call, src):
    content = (f"请先简要分析下面这句话里有哪些语法错误，"
               f"然后另起一行，以「改正后：」开头输出改正后的完整句子：\n{src}")
    out = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ])
    if "改正后：" in out:
        out = out.split("改正后：", 1)[1]
    return _clean(out)


# Strategy 4: iterative self-correction (correct once, then re-check)
def iterative(call, src):
    first = direct(call, src)
    out = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"原句：{src}\n你刚才的修改是：{first}\n"
            f"请再检查一次是否还有语法错误，如有请继续修正，"
            f"只输出最终改正后的完整句子。"},
    ])
    return _clean(out)


# Strategy 5: error-type-aware two-stage (our method)
def two_stage(call, src):
    # Stage 1: diagnose the error type
    diag = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"请指出下面这句话里的语法错误类型"
            f"（可选：错别字 / 用词不当 / 语序错误 / 搭配不当 / 缺字 / 多字），"
            f"只输出错误类型：\n{src}"},
    ])
    # Stage 2: correct based on the diagnosis
    out = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"这句话的错误类型是：{diag.strip()}\n"
            f"请根据以上错误类型改正这句话，只输出改正后的完整句子：\n{src}"},
    ])
    return _clean(out)


STRATEGIES = {
    "direct": direct,
    "few_shot": few_shot,
    "cot": cot,
    "iterative": iterative,
    "two_stage": two_stage,
}
