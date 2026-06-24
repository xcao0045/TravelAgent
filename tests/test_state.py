from agents.state import TravelPlanState


def test_travel_plan_state_has_required_fields():
    state = TravelPlanState(
        destination="成都",
        travel_date="2026-07-15",
        days=3,
        preferences="亲子、安静",
        budget_total=5000.0,
        weather_report="",
        attractions=[],
        restaurants=[],
        hotels=[],
        routes=[],
        final_report="",
        error_log=[],
        conversation=[],
        is_finalized=False,
    )
    assert state["destination"] == "成都"
    assert state["days"] == 3
    assert state["is_finalized"] is False


def test_state_is_mutable():
    state = TravelPlanState(
        destination="北京",
        travel_date="2026-08-01",
        days=5,
        preferences="历史文化",
        budget_total=8000.0,
        weather_report="",
        attractions=[],
        restaurants=[],
        hotels=[],
        routes=[],
        final_report="",
        error_log=[],
        conversation=[],
        is_finalized=False,
    )
    state["weather_report"] = "晴，28°C"
    assert state["weather_report"] == "晴，28°C"
