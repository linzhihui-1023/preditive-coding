import torch
from torch import nn

from predify.modules import DynamicErrorPCoderN


class IdentityModule(nn.Module):
    def forward(self, x):
        return x


def assert_close(name, actual, expected, atol=1e-6, rtol=1e-6):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(f"[ok] {name}")


def run_formula_check():
    sample_time = 0.1
    tau = 0.5
    gamma = sample_time / tau

    pcoder = DynamicErrorPCoderN(
        IdentityModule(),
        has_feedback=False,
        random_init=False,
        sample_time=sample_time,
        tau=tau,
    )

    # This check targets the dynamic-error formula, not the gradient scaling.
    pcoder.C_sqrt.fill_(1.0)

    ff1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target1 = torch.tensor([[[[2.0, 0.0], [1.0, 6.0]]]])
    residual1 = target1 - ff1
    expected_e1 = gamma * residual1
    expected_loss1 = torch.mean(expected_e1.pow(2))

    pcoder(ff=ff1, fb=None, target=target1, build_graph=True, ffm=1.0, fbm=0.0, erm=0.0)
    assert_close("step1.dynamic_error", pcoder.dynamic_error.detach(), expected_e1)
    assert_close("step1.prediction_error", pcoder.prediction_error.detach(), expected_loss1)

    ff2 = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    target2 = torch.tensor([[[[1.0, 3.0], [2.0, 2.0]]]])
    residual2 = target2 - ff2
    expected_e2 = gamma * residual2 + (1.0 - gamma) * expected_e1
    expected_loss2 = torch.mean(expected_e2.pow(2))

    pcoder(ff=ff2, fb=None, target=target2, build_graph=True, ffm=1.0, fbm=0.0, erm=0.0)
    assert_close("step2.dynamic_error", pcoder.dynamic_error.detach(), expected_e2)
    assert_close("step2.prediction_error", pcoder.prediction_error.detach(), expected_loss2)

    pcoder.reset()
    assert pcoder.dynamic_error is None
    print("[ok] reset.dynamic_error_is_none")


def run_ts_equals_tau_check():
    sample_time = 0.1
    tau = 0.1

    pcoder = DynamicErrorPCoderN(
        IdentityModule(),
        has_feedback=False,
        random_init=False,
        sample_time=sample_time,
        tau=tau,
    )

    pcoder.C_sqrt.fill_(1.0)

    ff = torch.tensor([[[[1.0, -1.0], [0.5, 2.5]]]])
    target = torch.tensor([[[[2.0, 1.0], [-0.5, 3.5]]]])
    residual = target - ff
    expected_dynamic_error = residual
    expected_loss = torch.mean(residual.pow(2))

    pcoder(ff=ff, fb=None, target=target, build_graph=True, ffm=1.0, fbm=0.0, erm=0.0)
    assert_close("ts_equals_tau.dynamic_error", pcoder.dynamic_error.detach(), expected_dynamic_error)
    assert_close("ts_equals_tau.prediction_error", pcoder.prediction_error.detach(), expected_loss)


if __name__ == "__main__":
    torch.manual_seed(0)
    run_formula_check()
    run_ts_equals_tau_check()
    print("dynamic error formula checks passed")
