"""
============================================================
CASE 1: Agentic AI Robotic Arm — Pick & Place Planner
============================================================
Uses LangGraph
Pipeline:
  START → parse_instruction → map_objects → validate_objects
        → generate_plan → format_output → END
============================================================
"""

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── LLM Setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ── Environment: available objects ────────────────────────────────────────────
ENVIRONMENT_OBJECTS = {
    "red_cube_1":    {"type": "cube",     "color": "red",    "position": "A2"},
    "blue_box_1":    {"type": "box",      "color": "blue",   "position": "B4"},
    "green_cyl_1":   {"type": "cylinder", "color": "green",  "position": "C1"},
    "yellow_tray_1": {"type": "tray",     "color": "yellow", "position": "D3"},
    "storage_bin_1": {"type": "bin",      "color": "gray",   "position": "E5"},
    "purple_cube_1": {"type": "cube",     "color": "purple", "position": "A5"},
}

# ── Library of allowed robot actions ──────────────────────────────────────────
LIBRARY_ACTIONS = [
    "get_position(object_id)",
    "move_to_grasp_pos(position)",
    "close_gripper()",
    "get_position(object_id)",
    "move_to_release_pos(position)",
    "open_gripper()",
]

# ── State Schema ──────────────────────────────────────────────────────────────
class RobotState(TypedDict):
    task_instruction: str                  # raw user input
    messages: Annotated[List, "messages"]  # conversation memory
    parsed_intent: Dict[str, Any]          # what the agent understood
    mapped_objects: Dict[str, Any]         # matched env objects
    validation_result: Dict[str, Any]      # validation pass/fail
    execution_plan: List[Dict[str, str]]   # final action steps
    error: str                             # error message if any

# ── Node 1: Parse Instruction ─────────────────────────────────────────────────
def parse_instruction(state: RobotState) -> RobotState:
    print("\n[NODE 1] Parsing instruction...")

    system = SystemMessage(content="""You are a robotic task parser.
Extract the intent from the user's instruction.
Return ONLY valid JSON with these fields:
{
  "action": "pick_and_place" or "multi_pick_and_place",
  "source_description": {"color": "...", "type": "..."},
  "target_description": {"color": "...", "type": "..."},
  "multi_source": true/false
}
If 'all' or 'every' is mentioned, set multi_source to true and source_description to match all of that type.""")

    human = HumanMessage(content=state["task_instruction"])
    response = llm.invoke([system, human])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)

    print(f"   Parsed intent: {json.dumps(parsed, indent=2)}")

    return {
        **state,
        "parsed_intent": parsed,
        "messages": state["messages"] + [human, response],
    }

# ── Node 2: Map Objects ───────────────────────────────────────────────────────
def map_objects(state: RobotState) -> RobotState:
    print("\n[NODE 2] Mapping objects from environment...")

    intent = state["parsed_intent"]
    env_str = json.dumps(ENVIRONMENT_OBJECTS, indent=2)

    system = SystemMessage(content=f"""You are a robotic object mapper.
Environment objects:
{env_str}

Given parsed intent, find matching object IDs from the environment.
Return ONLY valid JSON:
{{
  "source_ids": ["object_id", ...],
  "target_id": "object_id",
  "match_reasoning": "brief explanation"
}}""")

    human = HumanMessage(content=f"Intent: {json.dumps(intent)}")
    response = llm.invoke([system, human])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    mapped = json.loads(raw)

    print(f"   Mapped objects: {json.dumps(mapped, indent=2)}")

    return {
        **state,
        "mapped_objects": mapped,
        "messages": state["messages"] + [human, response],
    }

# ── Node 3: Validate Objects ──────────────────────────────────────────────────
def validate_objects(state: RobotState) -> RobotState:
    print("\n[NODE 3] Validating objects exist in environment...")

    mapped = state["mapped_objects"]
    errors = []

    for sid in mapped.get("source_ids", []):
        if sid not in ENVIRONMENT_OBJECTS:
            errors.append(f"Source object '{sid}' not found in environment")

    tid = mapped.get("target_id", "")
    if tid not in ENVIRONMENT_OBJECTS:
        errors.append(f"Target object '{tid}' not found in environment")

    if errors:
        print(f"   Validation FAILED: {errors}")
        return {**state, "validation_result": {"valid": False, "errors": errors}, "error": "; ".join(errors)}

    print("   Validation PASSED")
    return {**state, "validation_result": {"valid": True, "errors": []}}

# ── Conditional Edge: check validation ───────────────────────────────────────
def check_validation(state: RobotState) -> str:
    if state["validation_result"].get("valid", False):
        return "generate_plan"
    return "handle_error"

# ── Node 4a: Generate Plan ────────────────────────────────────────────────────
def generate_plan(state: RobotState) -> RobotState:
    print("\n[NODE 4] Generating execution plan...")

    mapped = state["mapped_objects"]
    source_ids = mapped["source_ids"]
    target_id = mapped["target_id"]

    system = SystemMessage(content=f"""You are a robotic arm execution planner.
You MUST use ONLY these library actions:
{json.dumps(LIBRARY_ACTIONS, indent=2)}

For each pick-and-place operation, the sequence is always:
1. get_position(<source_id>)
2. move_to_grasp_pos(<source_id>_pos)
3. close_gripper()
4. get_position(<target_id>)
5. move_to_release_pos(<target_id>_pos)
6. open_gripper()

Return ONLY valid JSON:
{{
  "execution_plan": [
    {{"step": 1, "action": "get_position(red_cube_1)", "description": "Get position of source object"}},
    ...
  ],
  "summary": "brief plan summary"
}}""")

    human = HumanMessage(content=f"Source objects: {source_ids}\nTarget object: {target_id}")
    response = llm.invoke([system, human])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    plan_data = json.loads(raw)

    print(f"   Generated {len(plan_data['execution_plan'])} steps")

    return {
        **state,
        "execution_plan": plan_data["execution_plan"],
        "messages": state["messages"] + [human, response],
    }

# ── Node 4b: Handle Error ─────────────────────────────────────────────────────
def handle_error(state: RobotState) -> RobotState:
    print(f"\n[NODE ERROR] {state.get('error', 'Unknown error')}")
    return {
        **state,
        "execution_plan": [],
    }

# ── Node 5: Format Output ─────────────────────────────────────────────────────
def format_output(state: RobotState) -> RobotState:
    print("\n[NODE 5] Formatting final output...")
    return state

# ── Build Graph ───────────────────────────────────────────────────────────────
def build_pick_place_graph():
    graph = StateGraph(RobotState)

    graph.add_node("parse_instruction", parse_instruction)
    graph.add_node("map_objects", map_objects)
    graph.add_node("validate_objects", validate_objects)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("handle_error", handle_error)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "parse_instruction")
    graph.add_edge("parse_instruction", "map_objects")
    graph.add_edge("map_objects", "validate_objects")
    graph.add_conditional_edges(
        "validate_objects",
        check_validation,
        {"generate_plan": "generate_plan", "handle_error": "handle_error"}
    )
    graph.add_edge("generate_plan", "format_output")
    graph.add_edge("handle_error", "format_output")
    graph.add_edge("format_output", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    png_data = app.get_graph().draw_mermaid_png()
    with open("pick_place.png", "wb") as f:
            f.write(png_data)
    print("Graph saved as pick_place.png")
    return app


# ── Pretty Print Plan ─────────────────────────────────────────────────────────
def print_plan(result: RobotState, task: str):
    print("\n" + "═"*60)
    print("  AGENTIC AI ROBOTIC ARM — EXECUTION PLAN")
    print("═"*60)
    print(f"  Task : {task}")
    print(f"  Agent: GPT-4o-mini via LangGraph")
    print("─"*60)

    if result.get("error") and not result.get("execution_plan"):
        print(f"  ERROR: {result['error']}")
        print("═"*60)
        return

    mapped = result.get("mapped_objects", {})
    print(f"  Source(s): {mapped.get('source_ids', [])}")
    print(f"  Target   : {mapped.get('target_id', '')}")
    print(f"  Reason   : {mapped.get('match_reasoning', '')}")
    print("─"*60)
    print("  EXECUTION STEPS:")
    for step in result.get("execution_plan", []):
        print(f"  [{step['step']:02d}] {step['action']}")
        print(f"       → {step['description']}")
    print("═"*60)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_pick_place_graph()
    config = {"configurable": {"thread_id": "robot-session-1"}}

    tasks = [
        "Pick the red cube and place it in the blue box",
        # "Move all cubes to the storage bin",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"USER TASK: {task}")
        print("="*60)

        initial_state: RobotState = {
            "task_instruction": task,
            "messages": [],
            "parsed_intent": {},
            "mapped_objects": {},
            "validation_result": {},
            "execution_plan": [],
            "error": "",
        }

        result = app.invoke(initial_state, config=config)
        print_plan(result, task)