import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution, Extension

import numpy
from Cython.Build import cythonize

link_args = []
include_dirs = [numpy.get_include()]
libraries = []

INPUT_DIR = "bot/utils/"

def build():
    source_files = []
    for root, directories, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith("pyx"):
                source_files.append(os.path.join(root, file))

    extensions = cythonize(
        Extension(
            name="cy_dijkstra",
            sources=source_files,
            include_dirs=include_dirs,
        ),
        compiler_directives={"binding": True, "language_level": 3},
    )

    distribution = Distribution({"name": "extended", "ext_modules": extensions})
    distribution.package_dir = "extended"

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        output_path = os.path.join(INPUT_DIR, relative_extension)
        shutil.copyfile(output, output_path)
        mode = os.stat(output_path).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(output_path, mode)


if __name__ == "__main__":
    build()
