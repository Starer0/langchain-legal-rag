"""Validated ownership metadata shared by laws and procedural guides."""

import re


def knowledge_base_id(entry):
    value = entry.get('knowledge_base_id')
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', value) is None:
        raise ValueError('资料必须登记有效的 knowledge_base_id（所属资料库）')
    return value
