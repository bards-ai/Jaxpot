from __future__ import annotations

import jax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def create_device_mesh(device_count: int | None = None) -> Mesh:
    """
    Create a device mesh for data-parallel training.

    Parameters
    ----------
    device_count : int | None
        Number of devices to use. If None, uses all available devices.

    Returns
    -------
    Mesh
        JAX device mesh with a single 'data' axis for data parallelism.
        Uses Auto axis type to let the compiler handle sharding propagation,
        which is needed for operations like embedding lookups (gather/scatter).
    """
    num_devices = device_count or len(jax.devices())
    return jax.make_mesh((num_devices,), ("data",), axis_types=(AxisType.Auto,))


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """
    Get sharding spec for replicated data (e.g., model parameters).

    Parameters
    ----------
    mesh : Mesh
        Device mesh to use.

    Returns
    -------
    NamedSharding
        Sharding that replicates data across all devices.
    """
    return NamedSharding(mesh, P())


def get_data_sharding(mesh: Mesh) -> NamedSharding:
    """
    Get sharding spec for batch-sharded data.

    Parameters
    ----------
    mesh : Mesh
        Device mesh to use.

    Returns
    -------
    NamedSharding
        Sharding that partitions data along the first (batch) axis.
    """
    return NamedSharding(mesh, P("data"))
