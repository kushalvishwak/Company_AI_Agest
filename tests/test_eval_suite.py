from agent_stack.layer7_observability import run_eval_suite

from .eval_suite import build_regression_cases


def test_regression_suite_all_pass():
    report = run_eval_suite(build_regression_cases())
    assert report.all_passed, report.summary()
