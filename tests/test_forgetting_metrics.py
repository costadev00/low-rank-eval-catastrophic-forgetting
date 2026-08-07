from low_rank_eval.evaluation.metrics import continual_learning_metrics


def test_two_task_metrics() -> None:
    metrics = continual_learning_metrics(
        base={"ifeval": 20.0, "math": 30.0},
        stage1={"ifeval": 60.0, "math": 28.0},
        stage2={"ifeval": 45.0, "math": 70.0},
        task_order=("ifeval", "math"),
    )
    assert metrics["forgetting_first"] == 15.0
    assert metrics["tasks"]["ifeval"]["gain"] == 40.0
    assert metrics["tasks"]["ifeval"]["bwt"] == -15.0
    assert metrics["tasks"]["math"]["plasticity"] == 70.0
    assert metrics["tasks"]["math"]["net"] == 40.0
