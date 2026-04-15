"""
============================================================
CASE 2: Agentic AI — Object Height Sorting Planner
============================================================
Input : List of object IDs with heights
Output: Sorted execution plan (ascending order by height)
        with robot pick-and-place steps for each sort swap

LangGraph Pipeline:
  START → ingest_objects → analyze_sort_order
        → plan_sort_moves → validate_plan
        → format_execution → END

Memory: MemorySaver (tracks multi-turn sessions)
============================================================
"""

import os
import json
import sys
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

# ── Library of allowed robot actions for sorting ──────────────────────────────
SORT_ACTIONS = [
    "get_position(object_id)",
    "move_to_grasp_pos(position)",
    "close_gripper()",
    "move_to_temp_pos(temp_slot)",
    "open_gripper()",
    "move_to_grasp_pos(position)",
    "close_gripper()",
    "move_to_release_pos(target_position)",
    "open_gripper()",
]

# ── State Schema ──────────────────────────────────────────────────────────────
class SortState(TypedDict):
    raw_input: str                         # user input string
    messages: Annotated[List, "messages"]  # conversation memory
    objects: List[Dict[str, Any]]          # parsed {id, height} list
    sorted_order: List[Dict[str, Any]]     # sorted objects ascending
    sort_algorithm: str                    # algorithm chosen by agent
    swap_sequence: List[Dict[str, Any]]    # swap operations needed
    execution_plan: List[Dict[str, Any]]   # full robot action plan
    validation: Dict[str, Any]             # validation result
    error: str

# ── Node 1: Ingest Objects ────────────────────────────────────────────────────
def ingest_objects(state: SortState) -> SortState:
    print("\n[NODE 1] Ingesting and parsing object list...")

    system = SystemMessage(content="""You are a robotic perception agent.
Parse the user's input to extract object IDs and their heights in original order only. Do Not rearrange or sort them.
Input can be in any format: JSON, comma-separated, natural language, etc.
Return ONLY valid JSON:
{
  "objects": [
    {"id": "obj_id", "height": <number_in_cm>},
    ...
  ],
  "unit": "cm" or "mm" or "m",
  "count": <integer>
}
Ensure heights are numeric values. Normalize to cm if needed.""")

    human = HumanMessage(content=state["raw_input"])
    response = llm.invoke([system, human])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    print(f"   Parsed {data['count']} objects in {data['unit']}")
    for obj in data["objects"]:
        print(f"   {obj['id']:20s} → {obj['height']} cm")

    return {
        **state,
        "objects": data["objects"],
        "messages": state["messages"] + [human, response],
    }

# ── Node 2: Analyze Sort Order ────────────────────────────────────────────────
def analyze_sort_order(state: SortState) -> SortState:
    print("\n[NODE 2] Analyzing optimal sort order (ascending by height)...")

    objects_str = json.dumps(state["objects"], indent=2)

    system = SystemMessage(content="""You are a sorting algorithm agent for robotics.
Given a list of objects with heights, determine:
1. The correct ascending sort order
2. The best algorithm for a robotic arm (insertion sort is preferred for physical sorting)
3. The exact swap/move sequence needed

Return ONLY valid JSON:
{
  "sorted_order": [{"id": "...", "height": ..., "final_position": 1}, ...],
  "algorithm": "insertion_sort",
  "rationale": "why this algorithm suits a robotic arm",
  "swap_sequence": [
    {
      "step": 1,
      "operation": "move",
      "object_id": "...",
      "from_slot": 3,
      "to_slot": 1,
      "reason": "..."
    },
    ...
  ]
}
Slots are numbered 1..N from left to right. Final sorted state must be ascending by height.""")

    human = HumanMessage(content=f"Objects to sort:\n{objects_str}")
    response = llm.invoke([system, human])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    sort_data = json.loads(raw)
    # print("sort_data",sort_data);
    # sys.exit()
    print(f"   Algorithm: {sort_data['algorithm']}")
    print(f"   Rationale: {sort_data['rationale']}")
    print(f"   Sorted order:")
    for obj in sort_data["sorted_order"]:
        print(f"   Pos {obj['final_position']}: {obj['id']:20s} → {obj['height']} cm")

    return {
        **state,
        "sorted_order": sort_data["sorted_order"],
        "sort_algorithm": sort_data["algorithm"],
        "swap_sequence": sort_data["swap_sequence"],
        "messages": state["messages"] + [human, response],
    }

# ── Node 3: Plan Sort Moves ───────────────────────────────────────────────────
def plan_sort_moves(state: SortState) -> SortState:
    print("\n[NODE 3] Generating robot execution plan for sort operations...")

    swap_str = json.dumps(state["swap_sequence"], indent=2)
    sorted_str = json.dumps(state["sorted_order"], indent=2)

    system = SystemMessage(content=f"""You are a robotic arm motion planner.
You must translate sorting swaps into robot action sequences.
Use ONLY these actions:
{json.dumps(SORT_ACTIONS, indent=2)}

For each swap/move operation:
- pick up object from its current slot
- place it in the temp slot if needed
- pick from temp and place in target slot

Return ONLY valid JSON:
{{
  "execution_plan": [
    {{
      "phase": "swap N",
      "object_id": "...",
      "from_slot": N,
      "to_slot": N,
      "steps": [
        {{"step": 1, "action": "get_position(obj_id)", "description": "..."}},
        ...
      ]
    }},
    ...
  ],
  "final_state": [
    {{"slot": 1, "object_id": "...", "height": ...}},
    ...
  ]
}}""")

    human = HumanMessage(content=f"Swap sequence:\n{swap_str}\n\nTarget sorted order:\n{sorted_str}")
    response = llm.invoke([system, human])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    plan_data = json.loads(raw)

    total_steps = sum(len(p["steps"]) for p in plan_data["execution_plan"])
    # print("plan_data", plan_data)
    # sys.exit()
    print(f"   Generated {len(plan_data['execution_plan'])} phases, {total_steps} total steps")

    return {
        **state,
        "execution_plan": plan_data["execution_plan"],
        "messages": state["messages"] + [human, response],
    }

# ── Node 4: Validate Plan ─────────────────────────────────────────────────────
def validate_plan(state: SortState) -> SortState:
    print("\n[NODE 4] Validating execution plan...")

    sorted_heights = [obj["height"] for obj in state["sorted_order"]]
    is_ascending = all(sorted_heights[i] <= sorted_heights[i+1]
                       for i in range(len(sorted_heights) - 1))

    if not is_ascending:
        return {
            **state,
            "validation": {"valid": False, "reason": "Sort order is not ascending"},
            "error": "Validation failed: heights not in ascending order"
        }

    if not state["execution_plan"]:
        return {
            **state,
            "validation": {"valid": False, "reason": "Empty execution plan"},
            "error": "Validation failed: no execution steps generated"
        }

    print("   Validation PASSED — ascending order confirmed")
    return {**state, "validation": {"valid": True, "reason": "OK"}, "error": ""}

# ── Conditional Edge ──────────────────────────────────────────────────────────
def check_sort_valid(state: SortState) -> str:
    return "format_execution" if state["validation"].get("valid") else "handle_sort_error"

# ── Node 5a: Format Execution ─────────────────────────────────────────────────
def format_execution(state: SortState) -> SortState:
    print("\n[NODE 5] Finalizing execution output...")
    return state

# ── Node 5b: Handle Error ─────────────────────────────────────────────────────
def handle_sort_error(state: SortState) -> SortState:
    print(f"\n[NODE ERROR] {state.get('error', 'Unknown error')}")
    return state

# ── Build Graph ───────────────────────────────────────────────────────────────
def build_sort_graph():
    graph = StateGraph(SortState)

    graph.add_node("ingest_objects", ingest_objects)
    graph.add_node("analyze_sort_order", analyze_sort_order)
    graph.add_node("plan_sort_moves", plan_sort_moves)
    graph.add_node("validate_plan", validate_plan)
    graph.add_node("format_execution", format_execution)
    graph.add_node("handle_sort_error", handle_sort_error)

    graph.add_edge(START, "ingest_objects")
    graph.add_edge("ingest_objects", "analyze_sort_order")
    graph.add_edge("analyze_sort_order", "plan_sort_moves")
    graph.add_edge("plan_sort_moves", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        check_sort_valid,
        {"format_execution": "format_execution", "handle_sort_error": "handle_sort_error"}
    )
    graph.add_edge("format_execution", END)
    graph.add_edge("handle_sort_error", END)

    memory = MemorySaver()
    # return graph.compile(checkpointer=memory)
    app = graph.compile(checkpointer=memory)
    png_data = app.get_graph().draw_mermaid_png()
    with open("sorting_planner.png", "wb") as f:
            f.write(png_data)
    print("Graph saved as sorting_planner.png")
    return app

# ── Pretty Print ──────────────────────────────────────────────────────────────
def print_sort_plan(result: SortState):
    print("\n" + "═"*65)
    print("  AGENTIC AI — OBJECT HEIGHT SORT EXECUTION PLAN")
    print("═"*65)

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        print("═"*65)
        return

    print(f"  Algorithm : {result.get('sort_algorithm', 'N/A')}")
    print("─"*65)

    print("  SORTED ORDER (ascending by height):")
    print(f"  {'Slot':<6} {'Object ID':<25} {'Height (cm)':<12}")
    print(f"  {'─'*5} {'─'*24} {'─'*11}")
    for obj in result["sorted_order"]:
        bar = "█" * int(obj["height"] // 5)
        print(f"  {obj['final_position']:<6} {obj['id']:<25} {obj['height']:<8} {bar}")

    print("\n  EXECUTION PHASES:")
    for phase in result["execution_plan"]:
        print(f"\n  ┌── {phase['phase'].upper()} — {phase['object_id']} (slot {phase['from_slot']} → {phase['to_slot']})")
        for s in phase["steps"]:
            print(f"  │  [{s['step']:02d}] {s['action']}")
            print(f"  │       → {s['description']}")
        print("  └" + "─"*50)
    print("═"*65)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_sort_graph()
    config = {"configurable": {"thread_id": "sort-session-1"}}

    # Example input — can be JSON, natural language, etc.
    user_input = """
    Sort these objects by height in ascending order:
    obj_A: 45 cm
    obj_B: 12 cm
    obj_C: 30 cm
    obj_D: 8 cm
    obj_E: 22 cm
    """

    print("="*65)
    print("USER INPUT:")
    print(user_input.strip())
    print("="*65)

    initial_state: SortState = {
        "raw_input": user_input,
        "messages": [],
        "objects": [],
        "sorted_order": [],
        "sort_algorithm": "",
        "swap_sequence": [],
        "execution_plan": [],
        "validation": {},
        "error": "",
    }

    result = app.invoke(initial_state, config=config)
    print_sort_plan(result)

    # ── Memory Demo: follow-up in same session ─────────────────────────────
    print("\n[MEMORY DEMO] Follow-up query in same session...")
    followup_input = """
    Now add obj_F with height 5 cm and obj_G with height 55 cm.
    Re-sort all objects.
    """

    followup_state: SortState = {
        "raw_input": followup_input,
        "messages": result.get("messages", []),  # carry over memory
        "objects": [],
        "sorted_order": [],
        "sort_algorithm": "",
        "swap_sequence": [],
        "execution_plan": [],
        "validation": {},
        "error": "",
    }

    result2 = app.invoke(followup_state, config={"configurable": {"thread_id": "sort-session-1"}})
    # with open("case_Result.txt", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(result2, indent=2))
    print_sort_plan(result2)
