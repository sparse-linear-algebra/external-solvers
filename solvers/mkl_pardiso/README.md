# MKL PARDISO Solver

This directory contains the planned MKL PARDISO implementation of the
`sparse_solver_interface` plugin ABI. The implementation is intentionally
deferred until the mapping below is accepted as the first-pass contract.

The local Intel oneAPI install is expected under `~/intel/oneapi`, with MKL
currently present at `~/intel/oneapi/mkl/2025.3`.

## First-Pass Scope

The first implementation should be a host-only direct solver plugin.

Supported:

- square sparse systems
- CSR-like row-oriented graphs
- `fp32`, `fp64`, `c64`, and `c128` values
- `i32` through `pardiso`
- `i64` through `pardiso_64`
- dense host RHS and solution matrices

Initially disabled or rejected:

- nonsquare systems
- device placement
- column-oriented graph input
- user permutations
- user-visible `iparm` tuning
- PARDISO handle store/restore
- symbolic reuse across multiple numeric factorizations
- complex symmetric modes other than the temporary Hermitian interpretation

## Object Mapping

`graph_t` owns the sparsity pattern. The PARDISO implementation stores SSI
zero-based CSR and rejects duplicate ids while preserving the caller's edge
order. Before calling PARDISO, the adapter translates to one-based CSR and sorts
column ids within each PARDISO row together with their values.

`sparse_matrix_t` owns values in the graph edge order. The first implementation
should require row-oriented values matching the graph's orientation.

`symbolic_t` should store graph ownership and the selected solver policy, but
should not own live PARDISO `pt` memory initially.

`numeric_factorization_t` should own the live PARDISO state:

- initialize `pt` and `iparm`
- run phase `11` for analysis/reordering
- run phase `22` for numeric factorization
- run phase `33` in `solve`
- run phase `-1` in the destructor to release PARDISO memory

This duplicates symbolic analysis when multiple numeric factorizations are made
from the same `symbolic_t`, but avoids shared mutable `pt` state in the first
implementation.

## Matrix Type Selection

`sparse_problem_properties_t` should select PARDISO `mtype`.

| SSI properties | PARDISO `mtype` |
| --- | ---: |
| real, unsymmetric or unknown symmetry | `11` |
| real, symmetric storage, numerically symmetric, positive definite | `2` |
| real, symmetric storage, numerically symmetric, not known positive definite | `-2` |
| real, symmetric storage, structurally symmetric only | `1` |
| complex, unsymmetric or unknown symmetry | `13` |
| complex, symmetric storage, numerically symmetric, positive definite | `4` |
| complex, symmetric storage, numerically symmetric, not known positive definite | `-4` |

For now, `numerically_symmetric` is treated as Hermitian when the dtype is
complex. This is a temporary interface interpretation and should be revisited
once the sparse solver interface can distinguish symmetric from Hermitian.

## Symmetric Storage

`symmetric_storage_t::unsymmetric` should use unsymmetric PARDISO modes.

`symmetric_storage_t::lower` and `symmetric_storage_t::upper` should only be
accepted when the selected `mtype` is symmetric or Hermitian.

`symmetric_storage_t::full` is normalized to the upper triangle for symmetric or
Hermitian PARDISO modes.

## Dense Solve

`numeric_factorization_t::solve(rhs, solution)` should:

- require `rhs.nrows() == n`
- use `rhs.ncols()` as PARDISO `nrhs`
- copy RHS to the dense host layout expected by PARDISO when needed
- copy the PARDISO output back into `solution`

Zero-copy dense solve support can come later.

## Validation

The sparse solver gym is installed at:

```text
/home/reidatcheson/sparse_linear_algebra/installs/bin/sparse-solver-gym
```

Once the plugin is implemented, use the gym as the primary integration test for
the shared library exported from this directory.
