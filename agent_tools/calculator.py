"""Deterministic Decimal arithmetic; legal applicability is supplied as evidence."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,31}$")
    operation: Literal["add", "subtract", "multiply", "divide"]
    left: str
    right: str


class CalculationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formula: str = Field(min_length=1, max_length=2000)
    inputs: dict[str, Decimal]
    steps: list[Step] = Field(min_length=1, max_length=30)
    result: str
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("inputs")
    @classmethod
    def finite_values(cls, value):
        if not value or len(value) > 50 or any(not n.is_finite() or abs(n.adjusted()) > 12 or len(n.as_tuple().digits) > 40 for n in value.values()):
            raise ValueError("Expected finite calculation inputs of bounded magnitude")
        return value


def calculate(plan):
    from agent_runtime.context import tool_call
    plan = CalculationPlan.model_validate(plan)

    def execute():
        values = dict(plan.inputs)
        with localcontext() as context:
            context.prec = 40
            for step in plan.steps:
                if step.name in values or step.left not in values or step.right not in values:
                    raise ValueError("Calculation steps must reference prior values and use unique names")
                left, right = values[step.left], values[step.right]
                if step.operation == "add":
                    value = left + right
                elif step.operation == "subtract":
                    value = left - right
                elif step.operation == "multiply":
                    value = left * right
                else:
                    value = left / right
                values[step.name] = value
            value = values[plan.result].quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {"amount": str(value), "formula": plan.formula, "assumptions": plan.assumptions}

    return tool_call("decimal.calculate", plan.model_dump(mode="json"), execute)
