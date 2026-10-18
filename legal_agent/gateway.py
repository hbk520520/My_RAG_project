"""Shared, model-free admission routing for every durable user entry point."""
import re

_GREETING = re.compile(r"^(你好|在吗|哈喽|hello|hi|嗨|你是谁|谢谢|拜拜|再见|早安|晚安)[！。？~\s]*$", re.I)
_EXACT = re.compile(r"《.*法》第\d+条")
_SIMPLE = re.compile(r"^(什么是|解释一下|告诉我|.*的概念|.*的定义是|.*出台时间是|.*?属于什么法)[?？\s]*$")


def classify(query):
    query = query.strip()
    if _GREETING.fullmatch(query):
        return "CHITCHAT"
    if _EXACT.search(query) or _SIMPLE.fullmatch(query):
        return "SIMPLE_QA"
    return "COMPLEX_TASK"


GREETING_REPLY = "您好，我是法务智能体，请描述您的法律问题。"
