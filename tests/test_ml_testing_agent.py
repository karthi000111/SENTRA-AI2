"""Unit tests for MLTestingAgent dynamic 10 testcase runner without API calls."""
from __future__ import annotations

import pytest
from app.testing.ml_testing_agent import MLTestingAgent, TestSuiteResult, TestCaseResult


@pytest.fixture
def sample_transformer_code():
    return """
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)
"""


@pytest.fixture
def broken_syntax_code():
    return """
import torch
class BrokenModel(nn.Module
    def forward(self, x):
        return x * 2
"""


def test_ml_testing_agent_all_10_cases_pass(sample_transformer_code):
    agent = MLTestingAgent()
    specs = {"d_model": {"value": "64", "status": "SUPPORTED"}}
    result: TestSuiteResult = agent.run_test_suite(sample_transformer_code, specs=specs)

    assert result.total_tests == 10
    assert len(result.results) == 10
    
    # Verify all 10 test case IDs are present
    test_ids = [r.test_id for r in result.results]
    assert test_ids == [f"TEST-{i:02d}" for i in range(1, 11)]

    # Check individual results
    for r in result.results:
        assert isinstance(r, TestCaseResult)
        assert r.passed, f"Test {r.test_id} ({r.name}) failed: {r.error_message}"

    assert result.passed_count == 10
    assert result.failed_count == 0
    assert result.all_passed is True


def test_ml_testing_agent_syntax_failure_graceful_handling(broken_syntax_code):
    agent = MLTestingAgent()
    result: TestSuiteResult = agent.run_test_suite(broken_syntax_code)

    assert result.total_tests == 10
    assert len(result.results) == 10
    assert result.results[0].test_id == "TEST-01"
    assert result.results[0].passed is False
    assert "Syntax error" in (result.results[0].error_message or "")
    assert result.passed_count == 0
    assert result.failed_count == 10
    assert result.all_passed is False


def test_ml_testing_agent_multifile_input():
    files_dict = {
        "layers.py": """
import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
    def forward(self, x):
        return self.fc(x)
""",
        "model.py": """
import torch
import torch.nn as nn
from layers import CustomLayer

class FullModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.layer = CustomLayer(dim)
    def forward(self, x):
        return self.layer(x)
"""
    }
    agent = MLTestingAgent()
    result = agent.run_test_suite(files_dict)

    assert result.total_tests == 10
    assert result.passed_count == 10
    assert result.all_passed is True
