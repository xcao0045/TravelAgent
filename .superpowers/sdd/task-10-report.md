# Task 10: 主控Agent + 汇总Agent + LangGraph 图组装

## Status: Completed

## Files Created

| File | Purpose |
|------|---------|
| `agents/orchestrator.py` | `orchestrator_node(TravelPlanState) -> dict` — validates destination field, returns validated fields |
| `agents/synthesizer.py` | `synthesizer_node(TravelPlanState) -> dict` — integrates weather/attractions/restaurants/hotels into Markdown report |
| `agents/graph.py` | `build_graph(Settings) -> CompiledStateGraph`, `run_travel_plan(dict, Settings) -> TravelPlanState`, module-level singletons `_get_llm()`, `_get_tools()`, `_get_retriever()` |
| `tests/test_graph.py` | 2 tests: graph compilation and orchestrator pass-through |

## Implementation Notes

- **agents/orchestrator.py**: Simple validation node — checks `destination` is non-empty, appends to `error_log` if missing, returns validated fields on success.
- **agents/synthesizer.py**: Invokes LLM with structured prompt combining weather, attractions, restaurants, hotels, and any error warnings into a full Markdown travel plan. Uses `_format_list()` helper.
- **agents/graph.py**: Assembles the 5-node LangGraph state graph:
  - `orchestrator` (entry point) fan-out via `Send` to 3 parallel sub-agents
  - `weather_agent`, `attraction_agent`, `hotel_agent` run in parallel
  - All three converge into `synthesizer`, then `END`
  - Module-level singletons (`_llm`, `_tools`, `_retriever`) initialized by `_init_dependencies()` on first `build_graph()` call — closes the circular dependency with sub-agent lazy imports.
- Used `from langgraph.types import Send` (current API) instead of deprecated `langgraph.constants`.
- All sub-agent mock-based tests continue to work by injecting `sys.modules["agents.graph"]`.

## Test Results

```
tests/test_graph.py::test_build_graph_compiles            PASSED
tests/test_graph.py::test_orchestrator_node_parses_input  PASSED
```

Full project: **39 passed in 1.42s** (no regressions).

## Task 10 Review Fixes

### Issues Fixed

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | CRITICAL | `agents/graph.py` | Changed `Send("...", {})` to `Send("...", state)` to pass actual state dict |
| 2 | HIGH | `agents/graph.py` | Added empty-destination guard: returns `[]` to skip fan-out when destination is blank |
| 3 | MEDIUM | `agents/graph.py` | Removed redundant `path_map` from `add_conditional_edges` call |
| 4 | LOW | `agents/synthesizer.py` | Removed unnecessary tool binding — uses `llm.invoke(prompt)` directly |
| 5 | LOW | `tests/test_graph.py` | Added explicit node assertions (filters out `__start__`/`__end__` internal nodes) |

### Test Results (post-fix)

```
tests/test_graph.py::test_build_graph_compiles            PASSED
tests/test_graph.py::test_orchestrator_node_parses_input  PASSED

Full project: **39 passed in 1.52s** (no regressions).
```

## Commit

```
git commit -m "feat: 主控Agent+汇总Agent+LangGraph状态图组装"
```
