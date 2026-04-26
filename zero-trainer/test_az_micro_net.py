from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import torch
import pytest

from az_micro_net import TaflAlphaZeroNet
from dataset import SelfPlayDataset
from export_onnx import export_model_to_onnx
from training_utils import NEG_LARGE, alpha_zero_loss, masked_policy_logits, normalize_visit_counts


def test_model_output_shapes() -> None:
    model = TaflAlphaZeroNet()
    x = torch.randn(4, 6, 11, 11)
    policy_logits, value = model(x)

    assert policy_logits.shape == (4, 4840)
    assert value.shape == (4, 1)


def test_masked_logits_block_illegal_moves() -> None:
    logits = torch.zeros(2, 4840)
    legal_mask = torch.zeros(2, 4840, dtype=torch.bool)
    legal_mask[:, :3] = True

    masked = masked_policy_logits(logits, legal_mask)
    probs = torch.softmax(masked, dim=1)

    assert torch.all(masked[:, 10:] == NEG_LARGE)
    assert torch.all(probs[:, 10:] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_alpha_zero_loss_is_finite() -> None:
    model = TaflAlphaZeroNet()
    x = torch.randn(3, 6, 11, 11)
    policy_logits, value_pred = model(x)

    legal_mask = torch.rand(3, 4840) < 0.1
    legal_mask[:, 0] = True

    visit_counts = torch.randint(0, 100, (3, 4840), dtype=torch.float32)
    visit_counts[:, 0] += 1
    pi_target = normalize_visit_counts(visit_counts, legal_mask)

    value_target = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32).unsqueeze(1)

    total, p_loss, v_loss = alpha_zero_loss(
        policy_logits=policy_logits,
        value_pred=value_pred,
        pi_target=pi_target,
        value_target=value_target,
        legal_mask=legal_mask,
    )

    assert torch.isfinite(total)
    assert torch.isfinite(p_loss)
    assert torch.isfinite(v_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_forward_on_cuda() -> None:
    device = torch.device("cuda")
    model = TaflAlphaZeroNet().to(device)

    x = torch.randn(2, 6, 11, 11, device=device)
    policy_logits, value = model(x)

    assert policy_logits.is_cuda
    assert value.is_cuda
    assert policy_logits.shape == (2, 4840)
    assert value.shape == (2, 1)


class _DummyPolicyHead(torch.nn.Module):
    def __init__(self, out: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_out", out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self._out.shape[0]:
            raise ValueError("Batch size mismatch for dummy policy head")
        return self._out


def test_policy_logits_square_major_indexing() -> None:
    # Build a deterministic policy tensor so we can verify flatten order.
    policy = torch.zeros(1, 40, 11, 11)
    for move_type in range(40):
        for square in range(121):
            row, col = divmod(square, 11)
            policy[0, move_type, row, col] = move_type * 1000 + square

    model = TaflAlphaZeroNet()
    model.policy_head = _DummyPolicyHead(policy)

    x = torch.zeros(1, 6, 11, 11)
    policy_logits, _ = model(x)

    expected = torch.empty(4840)
    for square in range(121):
        row, col = divmod(square, 11)
        for move_type in range(40):
            expected[square * 40 + move_type] = policy[0, move_type, row, col]

    assert torch.allclose(policy_logits[0], expected)


def test_onnx_export_matches_pytorch(tmp_path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")

    torch.manual_seed(0)
    model = TaflAlphaZeroNet()
    model.eval()

    x = torch.randn(2, 6, 11, 11)
    with torch.no_grad():
        pt_policy, pt_value = model(x)

    onnx_path = tmp_path / "model.onnx"
    export_model_to_onnx(model, onnx_path)

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_policy, ort_value = sess.run(None, {"input": x.numpy()})

    assert ort_policy.shape == tuple(pt_policy.shape)
    assert ort_value.shape == tuple(pt_value.shape)

    np.testing.assert_allclose(ort_policy, pt_policy.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ort_value, pt_value.numpy(), rtol=1e-4, atol=1e-5)


def _find_engine_binary(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "target" / "release" / "taflzero.exe",
        repo_root / "target" / "debug" / "taflzero.exe",
        repo_root / "target" / "release" / "taflzero",
        repo_root / "target" / "debug" / "taflzero",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def test_rust_dump_sample_roundtrip(tmp_path) -> None:
    pytest.importorskip("numpy")
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sample.bin"

    engine = _find_engine_binary(repo_root)
    cargo = shutil.which("cargo")

    def run_cmd(cmd, cwd):
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    result = None
    if engine is not None:
        result = run_cmd([str(engine), "--dump-sample", str(out_path)], repo_root)
        if result.returncode != 0 and "Unknown arg: --dump-sample" in (result.stderr + result.stdout):
            result = None  # Fall back to cargo run if available

    if result is None:
        if cargo is None:
            pytest.skip("cargo not found and engine binary missing or outdated")
        result = run_cmd([cargo, "run", "--quiet", "--", "--dump-sample", str(out_path)], repo_root)

    if result.returncode != 0:
        pytest.fail(f"engine failed: {result.stderr or result.stdout}")

    match = re.search(r"DUMP_SAMPLE index=(\d+)", result.stdout + result.stderr)
    assert match, f"expected DUMP_SAMPLE in output, got: {result.stdout} {result.stderr}"
    expected_index = int(match.group(1))

    dataset = SelfPlayDataset(out_path)
    assert len(dataset) == 1

    planes, legal_mask, pi_target, value = dataset[0]

    assert legal_mask[expected_index].item() is True
    assert torch.isclose(pi_target.sum(), torch.tensor(1.0), atol=1e-6)
    assert int(torch.argmax(pi_target).item()) == expected_index
    assert float(value.item()) == 1.0
