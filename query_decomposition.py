"""Split distinct legal intents into standalone retrieval questions."""

import json

from langchain_core.prompts import ChatPromptTemplate


DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "判断问题中有几个彼此不同、回答时都需要法律依据的法律问题。"
        "如果只有一个意图，返回一个问题；如果有多个，拆为最多四个独立、适合检索的问题。"
        "不要把同一意图改写成多种同义问法。每个子问题只保留解决该法律点所必需的事实，"
        "不要把其他诉求和无关背景重复塞入每个问题。使用明确的法律概念："
        "问仲裁申请期限时写成“仲裁时效期间”，问申请书内容时写成“仲裁申请书应载明的事项”。"
        "如果赔偿或补偿结论取决于行为是否合法、法定条件或期限上限，"
        "把这个必要前提单独列为检索问题。涉及工资倍数或比例时，直接询问具体倍数或比例。"
        "不要凭空增加用户未问的法律争议。"
        "保留用户明确提到且与该子问题相关的法律名称和法条号，不得添加用户没说的事实。"
        "只输出 JSON 对象，格式为"
        '{{"questions":["问题一？","问题二？"]}}，不得附加说明。',
    ),
    ("human", "{question}"),
])


class CompositeQuestionDecomposer:
    def __init__(self, model):
        self.model = model

    def decompose(self, question: str) -> list[str]:
        response = self.model.invoke(DECOMPOSITION_PROMPT.invoke({"question": question}))
        content = response.content if hasattr(response, "content") else response
        if not isinstance(content, str):
            return [question]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        try:
            questions = json.loads(content)["questions"]
        except (ValueError, TypeError, KeyError):
            return [question]
        if (
            not isinstance(questions, list)
            or not 1 <= len(questions) <= 4
            or any(not isinstance(item, str) or not item.strip() for item in questions)
        ):
            return [question]
        cleaned = [item.strip() for item in questions]
        return cleaned if len(set(cleaned)) == len(cleaned) else [question]
