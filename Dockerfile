# Reproducibility image for the Block Method vs hp-FEM comparison.
#
#   docker build -t pyblockmethod-repro .
#   docker run --rm pyblockmethod-repro
#
# Builds the pyBlockGrid C++ backend and the deal.II hp-FEM benchmark solver,
# then runs benchmarks/reproduce_comparison.py. No MATLAB required: the
# method-independent references are committed to the repo.

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# deal.II 9.3 + Eigen + build tools + Python are all in the Ubuntu 22.04 repos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libdeal.ii-dev libeigen3-dev liblapack-dev \
        python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# NumPy 2.0 and pybind11 >= 2.12 (older pybind11 corrupts bool arrays under NumPy 2.0).
RUN pip3 install --no-cache-dir "numpy>=2.0" "pybind11>=2.12" matplotlib scipy pytest

WORKDIR /pyblockmethod
COPY . .

# Build the pyBlockGrid C++ backend and install the extension into the package.
RUN cmake -S cpp -B cpp/build \
        -DPYTHON_EXECUTABLE="$(which python3)" \
        -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
    && cmake --build cpp/build -j"$(nproc)" \
    && cp cpp/build/_pyblockgrid_cpp*.so src/pyBlockGrid/

# Build the deal.II hp-FEM benchmark solver.
RUN cmake -S benchmarks/dealii -B benchmarks/dealii/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build benchmarks/dealii/build -j"$(nproc)"

# Build the C++ Lightning and AAA-LS spectral solvers (Eigen + LAPACK, R1.4).
RUN cmake -S benchmarks/spectral -B benchmarks/spectral/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build benchmarks/spectral/build -j"$(nproc)"

# Make the src-layout package importable (matches the editable install used in
# development) for pytest and the comparison driver.
ENV PYTHONPATH=/pyblockmethod/src

# Sanity: the full test suite should pass in the image.
RUN python3 -m pytest tests/ -q

CMD ["python3", "benchmarks/reproduce_comparison.py", "--reps", "6"]
