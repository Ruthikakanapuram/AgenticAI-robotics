import os
import json
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from environment import ENVIRONMENT_OBJECTS
from action_registry import LIBRARY_ACTIONS


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)


class PlannerState(TypedDict):
    user_input: str
    parsed_intent: Dict[str, Any]

    intent_approved: bool
    intent_feedback: str

    abstract_plan: Dict[str, Any]
    abstract_plan_approved: bool
    abstract_plan_feedback: str

    primitive_plan: Dict[str, Any]
    primitive_plan_approved: bool
    primitive_plan_feedback: str

    error: str


# ── Node 1: Parse Request / Intent Analyzer ───────────────────────────────────
def parse_request(state: PlannerState) -> PlannerState:
    system = SystemMessage(content="""
You are an intent analyzer for a robotic planning system.

Your job is to read the user's natural language instruction and infer the underlying goal, task type, and relevant details.

Return ONLY valid JSON with this schema:
{
  "task_type": "<inferred action verb or short phrase>",
  "goal": "<one sentence describing what the robot should achieve>",
  "entities": {
    "<entity_name>": "<relevant property or description>"
  },
  "constraints": ["<any ordering, spatial, conditional, or physical constraints>"],
  "status": "parsed" | "unclear"
}

Rules:
- Infer task_type from the instruction.
- Infer the goal from context.
- Extract all mentioned objects, locations, or agents into entities.
- Extract any constraints.
- If unclear, set status to "unclear".
- Return only JSON.
""")

    human = HumanMessage(content=state["user_input"])
    response = llm.invoke([system, human])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {
            **state,
            "parsed_intent": parsed,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "parsed_intent": {},
            "error": f"Failed to parse intent: {str(e)}"
        }


# ── Node 2: Review Intent ─────────────────────────────────────────────────────
def review_intent(state: PlannerState) -> PlannerState:
    print("\nParsed Intent:")
    print(json.dumps(state["parsed_intent"], indent=2))

    choice = input("\nApprove this intent? (yes/no): ").strip().lower()

    if choice == "yes":
        return {
            **state,
            "intent_approved": True,
            "intent_feedback": ""
        }

    feedback = input("Enter feedback to revise the intent: ").strip()
    return {
        **state,
        "intent_approved": False,
        "intent_feedback": feedback
    }


# ── Node 3: Revise Intent ─────────────────────────────────────────────────────
def revise_intent(state: PlannerState) -> PlannerState:
    system = SystemMessage(content="""
You are a robotic intent reviser.

You will receive:
1. The original user instruction
2. The previously parsed intent JSON
3. Human feedback on what needs to change

Your job:
- revise the parsed intent based on the human feedback
- preserve the same JSON structure
- return ONLY valid JSON
""")

    human = HumanMessage(content=json.dumps({
        "user_input": state["user_input"],
        "previous_intent": state["parsed_intent"],
        "human_feedback": state["intent_feedback"]
    }))

    response = llm.invoke([system, human])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        revised = json.loads(raw)
        return {
            **state,
            "parsed_intent": revised,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to revise intent: {str(e)}"
        }


def route_after_intent_review(state: PlannerState) -> str:
    if state["intent_approved"]:
        return "approved"
    return "revise"


# ── Node 4: Plan Task (Abstract Plan) ─────────────────────────────────────────
def plan_task(state: PlannerState) -> PlannerState:
    system = SystemMessage(content=f"""
You are an abstract robotic task planner.

You will receive:
1. Approved parsed intent
2. Available environment objects
3. Allowed robot library actions

Environment objects:
{json.dumps(ENVIRONMENT_OBJECTS, indent=2)}

Allowed library actions:
{json.dumps(LIBRARY_ACTIONS, indent=2)}

Your job:
- Create an ABSTRACT plan only
- Do NOT generate low-level executable primitive action sequence yet
- Use the environment objects to map likely source/target objects where possible
- Keep the plan at operation level like pick / place / move / sort / group
- If exact object mapping is uncertain, mention that in assumptions
- Return ONLY valid JSON

Return JSON in this schema:
{{
  "plan_summary": "<short summary>",
  "assumptions": ["<assumption 1>", "<assumption 2>"],
  "abstract_steps": [
    {{
      "step": 1,
      "operation": "<abstract operation>",
      "object_id": "<object id if known>",
      "target_id": "<target id if known>",
      "details": {{}}
    }}
  ],
  "status": "planned"
}}
""")

    human = HumanMessage(content=json.dumps({
        "user_input": state["user_input"],
        "approved_intent": state["parsed_intent"]
    }))

    response = llm.invoke([system, human])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        abstract_plan = json.loads(raw)
        return {
            **state,
            "abstract_plan": abstract_plan,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "abstract_plan": {},
            "error": f"Failed to generate abstract plan: {str(e)}"
        }


# ── Node 5: Review Abstract Plan ──────────────────────────────────────────────
def review_abstract_plan(state: PlannerState) -> PlannerState:
    print("\nAbstract Plan:")
    print(json.dumps(state["abstract_plan"], indent=2))

    choice = input("\nApprove this abstract plan? (yes/no): ").strip().lower()

    if choice == "yes":
        return {
            **state,
            "abstract_plan_approved": True,
            "abstract_plan_feedback": ""
        }

    feedback = input("Enter feedback to revise the abstract plan: ").strip()
    return {
        **state,
        "abstract_plan_approved": False,
        "abstract_plan_feedback": feedback
    }


# ── Node 6: Revise Abstract Plan ──────────────────────────────────────────────
def revise_plan(state: PlannerState) -> PlannerState:
    system = SystemMessage(content=f"""
You are a robotic abstract plan reviser.

You will receive:
1. Original user instruction
2. Approved intent
3. Previous abstract plan
4. Human feedback
5. Environment objects
6. Allowed library actions

Environment objects:
{json.dumps(ENVIRONMENT_OBJECTS, indent=2)}

Allowed library actions:
{json.dumps(LIBRARY_ACTIONS, indent=2)}

Your job:
- revise the abstract plan based on human feedback
- keep it abstract, not primitive
- preserve the same JSON structure
- return ONLY valid JSON
""")

    human = HumanMessage(content=json.dumps({
        "user_input": state["user_input"],
        "approved_intent": state["parsed_intent"],
        "previous_abstract_plan": state["abstract_plan"],
        "human_feedback": state["abstract_plan_feedback"]
    }))

    response = llm.invoke([system, human])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        revised_plan = json.loads(raw)
        return {
            **state,
            "abstract_plan": revised_plan,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to revise abstract plan: {str(e)}"
        }


def route_after_abstract_plan_review(state: PlannerState) -> str:
    if state["abstract_plan_approved"]:
        return "approved"
    return "revise"


# ── Node 7: Compile Actions (Primitive Plan) ──────────────────────────────────
def compile_actions(state: PlannerState) -> PlannerState:
    system = SystemMessage(content=f"""
You are a robotic action compiler.

You will receive:
1. Approved abstract plan
2. Available environment objects
3. Allowed robot library actions

Environment objects:
{json.dumps(ENVIRONMENT_OBJECTS, indent=2)}

Allowed library actions:
{json.dumps(LIBRARY_ACTIONS, indent=2)}

Your job:
- Convert the abstract plan into executable primitive steps
- Use ONLY the allowed library actions
- Keep the sequence logically correct
- Return ONLY valid JSON

Return JSON in this schema:
{{
  "plan_summary": "<short summary>",
  "primitive_steps": [
    {{
      "step": 1,
      "action": "<library action name without parentheses>",
      "args": {{}},
      "description": "<short description>"
    }}
  ],
  "status": "compiled"
}}
""")

    human = HumanMessage(content=json.dumps({
        "approved_intent": state["parsed_intent"],
        "approved_abstract_plan": state["abstract_plan"]
    }))

    response = llm.invoke([system, human])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        primitive_plan = json.loads(raw)
        return {
            **state,
            "primitive_plan": primitive_plan,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "primitive_plan": {},
            "error": f"Failed to compile primitive plan: {str(e)}"
        }


# ── Node 8: Review Primitive Plan ─────────────────────────────────────────────
def review_primitive_plan(state: PlannerState) -> PlannerState:
    print("\nPrimitive Plan:")
    print(json.dumps(state["primitive_plan"], indent=2))

    choice = input("\nApprove this primitive plan? (yes/no): ").strip().lower()

    if choice == "yes":
        return {
            **state,
            "primitive_plan_approved": True,
            "primitive_plan_feedback": ""
        }

    feedback = input("Enter feedback to revise the primitive plan: ").strip()
    return {
        **state,
        "primitive_plan_approved": False,
        "primitive_plan_feedback": feedback
    }


# ── Node 9: Revise Primitive Plan ─────────────────────────────────────────────
def revise_compilation(state: PlannerState) -> PlannerState:
    system = SystemMessage(content=f"""
You are a robotic primitive plan reviser.

You will receive:
1. Approved intent
2. Approved abstract plan
3. Previous primitive plan
4. Human feedback
5. Available environment objects
6. Allowed library actions

Environment objects:
{json.dumps(ENVIRONMENT_OBJECTS, indent=2)}

Allowed library actions:
{json.dumps(LIBRARY_ACTIONS, indent=2)}

Your job:
- Revise the primitive plan based on human feedback
- Use ONLY the allowed library actions
- Preserve the same JSON structure
- Return ONLY valid JSON
""")

    human = HumanMessage(content=json.dumps({
        "approved_intent": state["parsed_intent"],
        "approved_abstract_plan": state["abstract_plan"],
        "previous_primitive_plan": state["primitive_plan"],
        "human_feedback": state["primitive_plan_feedback"]
    }))

    response = llm.invoke([system, human])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        revised_plan = json.loads(raw)
        return {
            **state,
            "primitive_plan": revised_plan,
            "error": ""
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to revise primitive plan: {str(e)}"
        }


def route_after_primitive_plan_review(state: PlannerState) -> str:
    if state["primitive_plan_approved"]:
        return "approved"
    return "revise"


# ── Node 10: Validate Plan ────────────────────────────────────────────────────
def validate_plan(state: PlannerState) -> PlannerState:
    primitive_plan = state.get("primitive_plan", {})
    steps = primitive_plan.get("primitive_steps", [])

    if not steps:
        return {
            **state,
            "error": "Validation failed: primitive_steps is empty"
        }

    allowed_action_names = {
        action.split("(")[0] for action in LIBRARY_ACTIONS
    }

    for step in steps:
        action_name = step.get("action", "")
        if action_name not in allowed_action_names:
            return {
                **state,
                "error": f"Validation failed: action '{action_name}' is not in registry"
            }

    return {
        **state,
        "error": ""
    }


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(PlannerState)

    graph.add_node("parse_request", parse_request)
    graph.add_node("review_intent", review_intent)
    graph.add_node("revise_intent", revise_intent)

    graph.add_node("plan_task", plan_task)
    graph.add_node("review_abstract_plan", review_abstract_plan)
    graph.add_node("revise_plan", revise_plan)

    graph.add_node("compile_actions", compile_actions)
    graph.add_node("review_primitive_plan", review_primitive_plan)
    graph.add_node("revise_compilation", revise_compilation)
    graph.add_node("validate_plan", validate_plan)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "review_intent")

    graph.add_conditional_edges(
        "review_intent",
        route_after_intent_review,
        {
            "approved": "plan_task",
            "revise": "revise_intent"
        }
    )

    graph.add_edge("revise_intent", "review_intent")

    graph.add_edge("plan_task", "review_abstract_plan")

    graph.add_conditional_edges(
        "review_abstract_plan",
        route_after_abstract_plan_review,
        {
            "approved": "compile_actions",
            "revise": "revise_plan"
        }
    )

    graph.add_edge("revise_plan", "review_abstract_plan")

    graph.add_edge("compile_actions", "review_primitive_plan")

    graph.add_conditional_edges(
        "review_primitive_plan",
        route_after_primitive_plan_review,
        {
            "approved": "validate_plan",
            "revise": "revise_compilation"
        }
    )

    graph.add_edge("revise_compilation", "review_primitive_plan")
    graph.add_edge("validate_plan", END)

    app = graph.compile()

    png_data = app.get_graph().draw_mermaid_png()
    with open("humanInTheLoop.png", "wb") as f:
        f.write(png_data)

    return app


if __name__ == "__main__":
    app = build_graph()
    user_query = input("Enter a natural language instruction for the robot: ").strip()
    # user_query = """
    # Sort these objects by height in ascending order:
    # obj_A: 45 cm
    # obj_B: 12 cm
    # obj_C: 30 cm
    # obj_D: 8 cm
    # obj_E: 22 cm
    # """
    initial_state: PlannerState = {
        # "user_input": "Pick the red cube and place it in the blue box",
        "user_input": user_query,
        "parsed_intent": {},

        "intent_approved": False,
        "intent_feedback": "",

        "abstract_plan": {},
        "abstract_plan_approved": False,
        "abstract_plan_feedback": "",

        "primitive_plan": {},
        "primitive_plan_approved": False,
        "primitive_plan_feedback": "",

        "error": ""
    }

    result = app.invoke(initial_state)

    print("\nFinal Approved Intent:")
    print(json.dumps(result["parsed_intent"], indent=2))

    print("\nFinal Approved Abstract Plan:")
    print(json.dumps(result["abstract_plan"], indent=2))

    print("\nFinal Approved Primitive Plan:")
    print(json.dumps(result["primitive_plan"], indent=2))

    print("\nValidation Error:")
    print(result["error"])