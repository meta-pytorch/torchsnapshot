# TorchSnapshot (Beta Release) TEST

<p align="center">
<a href="https://github.com/pytorch/torchsnapshot/actions?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/pytorch/torchsnapshot/.github/workflows/run_tests.yaml?branch=main" alt="build status"></a>
<a href="https://pypi.org/project/torchsnapshot"><img src="https://img.shields.io/pypi/v/torchsnapshot" alt="pypi version"></a>
<a href="https://anaconda.org/conda-forge/torchsnapshot"><img src="https://img.shields.io/conda/vn/conda-forge/torchsnapshot" alt="conda version"></a>
<a href="https://pypi.org/project/torchsnapshot-nightly"><img src="https://img.shields.io/pypi/v/torchsnapshot-nightly?label=nightly" alt="pypi nightly version"></a>
<a href="https://codecov.io/gh/pytorch/torchsnapshot"><img src="https://codecov.io/gh/pytorch/torchsnapshot/branch/main/graph/badge.svg?token=DR67Q6T7YF" alt="codecov"></a>
<a href="https://github.com/pytorch/torchsnapshot/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/torchsnapshot" alt="bsd license"></a>
</div>

A performant, memory-efficient checkpointing library for PyTorch applications, designed with large, complex distributed workloads in mind.


## Install

Requires Python >= 3.8 and PyTorch >= 2.0.0

From pip:

```bash
# Stable
pip install torchsnapshot
# Or, using conda
conda install -c conda-forge torchsnapshot

# Nightly
pip install --pre torchsnapshot-nightly
```


From source:

```bash
git clone https://github.com/pytorch/torchsnapshot
cd torchsnapshot
pip install -r requirements.txt
python setup.py install
```

## Why TorchSnapshot

**Performance**
- TorchSnapshot provides a fast checkpointing implementation employing various optimizations, including zero-copy serialization for most tensor types, overlapped device-to-host copy and storage I/O, parallelized storage I/O.
- TorchSnapshot greatly speeds up checkpointing for DistributedDataParallel workloads by distributing the write load across all ranks ([benchmark](https://github.com/pytorch/torchsnapshot/tree/main/benchmarks/ddp)).
- When host memory is abundant, TorchSnapshot allows training to resume before all storage I/O completes, reducing the time blocked by checkpoint saving.

**Memory Usage**
- TorchSnapshot's memory usage adapts to the host's available resources, greatly reducing the chance of out-of-memory issues when saving and loading checkpoints.
- TorchSnapshot supports efficient random access to individual objects within a snapshot, even when the snapshot is stored in a cloud object storage.

**Usability**
- Simple APIs that are consistent between distributed and non-distributed workloads.
- Out of the box integration with commonly used cloud object storage systems.
- Automatic resharding (elasticity) on world size change for supported workloads ([more details](https://pytorch.org/torchsnapshot/getting_started.html#elasticity-experimental)).

**Security**
- Secure tensor serialization without pickle dependency [WIP].


## Getting Started

```python
from torchsnapshot import Snapshot

# Taking a snapshot
app_state = {"model": model, "optimizer": optimizer}
snapshot = Snapshot.take(path="/path/to/snapshot", app_state=app_state)

# Restoring from a snapshot
snapshot.restore(app_state=app_state)
```

See the [documentation](https://pytorch.org/torchsnapshot/main/getting_started.html) for more details.


## License

torchsnapshot is BSD licensed, as found in the [LICENSE](LICENSE) file.
