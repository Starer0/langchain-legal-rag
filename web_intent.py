"""Conservative, zero-model-call routing for standalone answer-style preferences."""
import re


_STYLE_WORDS = (
    '具体例子', '通俗易懂', '先给结论', '先看结论', '再给例子', '再看例子',
    '再举例', '给结论', '看结论', '举例说明', '分点说明', '分点解释', '简单一点',
    '详细一点', '简短一点', '简单一些', '详细一些', '简短一些', '长篇大论',
    '专业术语', '回答', '回复', '解释', '说明', '结论', '例子', '举例', '分点',
    '中文', '英文', '简短', '简洁', '详细', '简单', '通俗', '具体', '清楚',
    '先', '再', '然后', '少用', '多用', '使用', '用', '看', '给', '的', '一些', '一点',
)
_LEAD = re.compile(r'^我(?:更)?(?:喜欢|习惯|偏好)')


def has_legal_request(question):
    """Question markers include elliptical follow-ups without a legal noun."""
    text=question.strip().rstrip('。.!！；;，, ')
    return bool(re.search(r'多久|多少|怎么办|怎么|如何|为什么|是什么|有哪些|能否|可否|[？?]|[吗呢]$',text)
                or re.search(r'[，,。；;]\s*(?:请)?(?:解释|说明|分析)(?:一下)?.*(?:法|合同|工资|仲裁|试用|实习|赔偿|社保)',question))


def is_style_preference(question):
    """Only a complete, narrow style statement qualifies; mixed/unknown text is RAG."""
    if not isinstance(question,str) or len(question)>500:return False
    if re.search(r'[“”「」『』"‘’\?？]',question):return False
    compact=re.sub(r'[\s，,。.!！；;、→]+','',question)
    lead=_LEAD.match(compact)
    if not lead:return False
    style=compact[lead.end():]
    if not style:return False
    # Bounded dynamic programming: overlapping words must not trigger regex
    # backtracking on adversarial near-matches.
    reachable={0}
    for position in range(len(style)):
        if position in reachable:
            for word in _STYLE_WORDS:
                if style.startswith(word,position):reachable.add(position+len(word))
    return len(style) in reachable
