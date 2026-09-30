"""One local date tool and the bounded model/tool exchange used for learning."""

import json
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from performance import measure

MAX_TOOL_CALLS = 2


class DateIntervalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="起始日期，YYYY-MM-DD")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="结束日期，YYYY-MM-DD，不能早于起始日期")


@tool(args_schema=DateIntervalInput)
def calculate_date_interval(start_date: str, end_date: str) -> dict:
    """计算两个明确日期之间相隔的自然日数，使用 end_date - start_date。

    只做日历运算，不判断法律期限、工作日、节假日或是否应计入首尾两天。
    """
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError as error:
        raise ValueError("请提供有效的 YYYY-MM-DD 日期。") from error
    if end < start:
        raise ValueError("结束日期不能早于起始日期。")
    return {
        "start_date": start_date, "end_date": end_date, "days": (end - start).days,
        "counting_rule": "end_date - start_date，按自然日计算，不额外加一天",
    }


TOOL_PLAN_PROMPT = """你为法律 RAG 助手选择是否需要日期计算工具。
仅当当前问题只要求计算两个明确日期相隔的自然日数时，调用 calculate_date_interval。
结合历史理解日期，但不能猜测缺失的年月日。最多提出一次工具调用。
法律条文、办事材料、法律期限、工作日、日期不完整或同时包含其他任务的问题，都不要调用工具。
不调用工具时不要回答问题，后续由知识库检索流程处理。
"""

TOOL_LOOP_PROMPT = """你为法律 RAG 助手处理纯日期间隔计算或多个日期区间的比较。
仅对完整明确的日期调用 calculate_date_interval，不猜测缺失的年月日。
每次只请求一个工具调用。拿到工具结果后，判断是否需要再计算一个区间，或直接用中文回答。
本轮最多执行两次工具。已有结果要复用，不重复请求；缺少结果的部分不得编造。
只做自然日间隔，不判断法律期限、工作日或节假日，不把结果解释为法律结论。
法律问题、缺失日期或同时含法律任务的问题不要调用工具，交给后续 RAG。
如果尚未使用任何工具，不要直接回答问题。
"""


def build_date_tool_nodes(model, *, agent_loop=False):
    planner = model.bind_tools(
        [calculate_date_interval], **({"parallel_tool_calls": False} if agent_loop else {}),
    )

    def tool_plan(state):
        messages = [SystemMessage(content=TOOL_LOOP_PROMPT if agent_loop else TOOL_PLAN_PROMPT), *state.get("history", []),
                    HumanMessage(content=state["question"])]
        response = measure(state, "tool_plan", lambda: planner.invoke(messages))
        requested = bool(response.tool_calls or response.invalid_tool_calls)
        return {
            "tool_request": response if requested else None,
            "tool_messages": [*messages, response] if requested else [],
            "tool_result": None,
            "tool_results": [], "tool_call_count": 0,
            "tool_call_limit": MAX_TOOL_CALLS if agent_loop else 1,
            "retrieval_question": state["question"] if requested else "",
        }

    def execute_tool(state):
        request = state["tool_request"]
        calls = request.tool_calls
        count = state.get("tool_call_count", 0)
        if count >= (MAX_TOOL_CALLS if agent_loop else 1):
            result = {"error": "已达到本轮工具调用上限。"}
        elif request.invalid_tool_calls or len(calls) != 1:
            result = {"error": "每次只接受一个格式正确的工具调用。"}
        elif calls[0]["name"] != calculate_date_interval.name:
            result = {"error": "请求的工具不在允许执行的工具列表中。"}
        else:
            try:
                result = measure(state, "tool", lambda: calculate_date_interval.invoke(calls[0]["args"]))
            except ValueError as error:
                result = {"error": str(error)}
        if "error" in result:
            return {"tool_result": result, "tool_messages": state["tool_messages"],
                    "tool_call_count": count, "tool_call_limit": MAX_TOOL_CALLS if agent_loop else 1}
        message = ToolMessage(
            content=json.dumps(result, ensure_ascii=False), tool_call_id=calls[0]["id"],
            name=calculate_date_interval.name,
        )
        return {
            "tool_result": result, "tool_messages": [*state["tool_messages"], message],
            "tool_results": [*state.get("tool_results", []), result],
            "tool_call_count": count + 1,
            "tool_call_limit": MAX_TOOL_CALLS if agent_loop else 1,
        }

    def tool_continue(state):
        response = measure(state, "tool_continue", lambda: planner.invoke(state["tool_messages"]))
        requested = bool(response.tool_calls or response.invalid_tool_calls)
        update = {
            "tool_request": response if requested else None,
            "tool_messages": [*state["tool_messages"], response],
        }
        if not requested:
            update.update(
                answer=StrOutputParser().invoke(response).strip() or "模型未给出答复，请参考已展示的工具结果。",
                candidates=[], docs=[], sources=[],
            )
        return update

    def tool_answer(state):
        result = state["tool_result"]
        if "error" in result:
            answer = f"工具调用失败：{result['error']} 请提供两个有效、完整且顺序正确的日期。"
        else:
            messages = [SystemMessage(content=(
                "请根据日期工具返回的结果用中文回答。明确日期、相隔的自然日数与计数规则，"
                "不要把自然日运算解释为法律期限或工作日计算。"
                + ("本轮已经达到两次工具调用上限。比较已有工具结果；"
                   "若原问题还有尚未计算的区间，明确该部分没有工具依据，不要自行编造数值。" if agent_loop else "")
            )), *state["tool_messages"][1:]]
            response = measure(state, "answer", lambda: model.invoke(messages))
            answer = StrOutputParser().invoke(response)
        return {"answer": answer, "candidates": [], "docs": [], "sources": []}

    return tool_plan, execute_tool, tool_answer, tool_continue
