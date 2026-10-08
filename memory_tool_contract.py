"""Bounded provider contracts. Parsing does not grant permission to write."""
from dataclasses import dataclass, asdict
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, StrictStr, StrictBool, Field,field_validator

VERSION = 'model-memory-tools-v1'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class SourceSchema(StrictModel):
    kind: Literal['current_input','message','manual_document','legacy_document']
    reference: StrictStr = Field(min_length=1,max_length=256)
    quote: StrictStr = Field(min_length=1,max_length=1000)


class OperationSchema(StrictModel):
    action: Literal['remember','merge','propose_change','request_delete','request_clear']
    layer: Literal['core','extended']
    category: Literal['preference','background','learning_focus']
    basis: Literal['declared','inferred']
    content: StrictStr = Field(max_length=1000)
    sources: list[SourceSchema] = Field(max_length=12)
    target_ids: list[StrictStr] = Field(max_length=12)
    relation: Literal['new','duplicate','complement','conflict']
    delta: StrictStr = Field(max_length=1000)


class UpdateSchema(StrictModel):
    operations: list[OperationSchema] = Field(max_length=12)


class AnswerSchema(StrictModel):
    route: Literal['rag','preference','general','clarify','out_of_scope']
    legal_question: StrictStr = Field(max_length=32000)
    retrieval_question: StrictStr = Field(max_length=32000)
    include_guide: StrictBool | None
    reply_plan: StrictStr = Field(max_length=1000)
    memory_request: StrictBool
    help_topic: Literal['greeting','memory','conversation','sources','scope',''] | None = None
    out_of_scope_request: StrictBool = False

    @field_validator('help_topic')
    @classmethod
    def empty_help_topic(cls,value):
        return None if value=='' else value


@dataclass(frozen=True)
class SourceRef:
    kind: str
    reference: str
    quote: str


@dataclass(frozen=True)
class MemoryOperation:
    action: str
    layer: str
    category: str
    basis: str
    content: str
    sources: tuple[SourceRef,...]
    target_ids: tuple[str,...]
    relation: str
    delta: str


@dataclass(frozen=True)
class MemoryToolCall:
    provider_call_id: str
    operations: tuple[MemoryOperation,...]


@dataclass(frozen=True)
class AnswerPlan:
    route: str
    legal_question: str
    retrieval_question: str
    include_guide: bool | None
    reply_plan: str
    memory_request: bool
    help_topic: str | None = None
    out_of_scope_request: bool = False


@dataclass(frozen=True)
class TurnDecision:
    answer: AnswerPlan
    memory_call: MemoryToolCall | None


@dataclass(frozen=True)
class ToolContext:
    owner: str
    scope: str
    execution_id: str
    revision: int
    policy_epoch: int
    input_snapshot: dict


@dataclass(frozen=True)
class PreparedUpdate:
    context: ToolContext
    call: MemoryToolCall
    facts: tuple[dict,...]
    suggestions: tuple[dict,...]
    prepared_document: dict
    argument_digest: str


@dataclass(frozen=True)
class ToolResult:
    status: str
    revision: int
    added: int
    merged: int
    suggestion_ids: tuple[str,...]
    message: str


def parse_memory_call(raw, provider_call_id=''):
    validated = UpdateSchema.model_validate(raw)
    operations=[]
    for op in validated.operations:
        values=op.model_dump()
        values['sources']=tuple(SourceRef(**s) for s in values['sources'])
        values['target_ids']=tuple(values['target_ids'])
        operations.append(MemoryOperation(**values))
    return MemoryToolCall(provider_call_id,tuple(operations))


def tool_schema(name, model, description):
    return {'type':'function','function':{'name':name,'description':description,
                                        'parameters':model.model_json_schema()}}


UPDATE_MEMORY = tool_schema('update_memory',UpdateSchema,'提出有原始用户来源的长期记忆更新；程序校验后提交，不代表已经保存。')
PLAN_ANSWER = tool_schema('plan_answer',AnswerSchema,'每轮一次规划回答路线，同时完成法律检索改写；不能授予资料权限。')


def decision_from_dict(raw):
    raw=json.loads(json.dumps(raw))
    if set(raw)!={'answer','memory_call'}:raise ValueError('Invalid decision')
    answer=AnswerPlan(**AnswerSchema.model_validate(raw['answer']).model_dump())
    call=raw['memory_call']
    return TurnDecision(answer,parse_memory_call({'operations':call['operations']},call['provider_call_id']) if call else None)


def prepared_from_dict(raw):
    raw=json.loads(json.dumps(raw))
    return PreparedUpdate(ToolContext(**raw['context']),
        parse_memory_call({'operations':raw['call']['operations']},raw['call']['provider_call_id']),
        tuple(raw['facts']),tuple(raw['suggestions']),raw['prepared_document'],raw['argument_digest'])
