import json
import pytest
from src.router import Router
from src.agent import Reflector, Searcher, DeepReasoner

@pytest.fixture
def setup_router():
    router = Router()
    return router

def test_router_initialization(setup_router):
    router = setup_router
    assert router is not None

def test_router_process_sllm_results(setup_router):
    router = setup_router
    with open('data/sllm_results.jsonl', 'r') as file:
        for line in file:
            result = json.loads(line)
            output = router.process_sllm_result(result)
            assert output is not None

def test_router_agent_dispatch(setup_router):
    router = setup_router
    sample_result = {
        "type": "Network Camera",
        "vendor": "Hikvision",
        "model": "DS-2CD2142FWD-I",
        "confidence": 0.92
    }
    output = router.dispatch_agent(sample_result)
    assert isinstance(output, (Reflector, Searcher, DeepReasoner))

def test_router_output_format(setup_router):
    router = setup_router
    sample_result = {
        "type": "Network Camera",
        "vendor": "Hikvision",
        "model": "DS-2CD2142FWD-I",
        "confidence": 0.92
    }
    output = router.process_sllm_result(sample_result)
    assert isinstance(output, dict)
    assert "prediction" in output
    assert "confidence" in output

def test_router_error_handling(setup_router):
    router = setup_router
    invalid_result = {"invalid_key": "invalid_value"}
    output = router.process_sllm_result(invalid_result)
    assert output is None  # Assuming the router returns None for invalid input