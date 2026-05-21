# external-solvers

External implementations of the
`/home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0`
solver interface.

This is a CMake project. Each solver lives in its own subdirectory under
`solvers/`.

## Configure

```sh
cmake -S . -B build
```

The sparse solver interface install is discovered through
`SPARSE_SOLVER_INTERFACE_ROOT`, which defaults to:

```text
/home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0
```

Override it when configuring if needed:

```sh
cmake -S . -B build -DSPARSE_SOLVER_INTERFACE_ROOT=/path/to/sparse_solver_interface
```

## Solvers

- `solvers/mkl_pardiso`: optional MKL PARDISO plugin. Enable it with
  `-DEXTERNAL_SOLVERS_ENABLE_MKL_PARDISO=ON`.
