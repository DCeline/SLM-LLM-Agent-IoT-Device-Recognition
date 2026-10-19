import jsonlines
import pytest
from src.agent import Reflector, Searcher, DeepReasoner
from src.router import Router

@pytest.fixture
def setup_agents():
    reflector = Reflector()
    searcher = Searcher()
    deep_reasoner = DeepReasoner()
    return reflector, searcher, deep_reasoner

@pytest.fixture
def setup_router(setup_agents):
    reflector, searcher, deep_reasoner = setup_agents
    router = Router(reflector, searcher, deep_reasoner)
    return router

def test_reflector_agent(setup_agents):
    reflector, _, _ = setup_agents
    input_data = {"text": "Sample input for reflection."}
    expected_output = {"status": "success", "reflection": "Reflected output based on input."}
    
    output = reflector.process(input_data)
    assert output == expected_output

def test_searcher_agent(setup_agents):
    _, searcher, _ = setup_agents
    input_data = {"query": "What is the model of device X?"}
    expected_output = {"model": "Device X", "confidence": 0.95}
    
    output = searcher.search(input_data)
    assert output == expected_output

def test_deep_reasoner_agent(setup_agents):
    _, _, deep_reasoner = setup_agents
    input_data = {"data": "Complex reasoning data."}
    expected_output = {"reasoning": "Deep reasoning output."}
    
    output = deep_reasoner.reason(input_data)
    assert output == expected_output

def test_router_functionality(setup_router):
    router = setup_router
    input_data = {"text": "Sample input for routing."}
    expected_output = {"final_result": "Processed result from appropriate agent."}
    
    output = router.route(input_data)
    assert output == expected_output

def test_sllm_results_import():
    with jsonlines.open('data/sllm_results.jsonl') as reader:
        for obj in reader:
            assert "model" in obj
            assert "confidence" in obj
            assert isinstance(obj["model"], str)
            assert isinstance(obj["confidence"], float)