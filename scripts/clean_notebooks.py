import subprocess
import sys

args = sys.argv[1:]

for file in args:
    subprocess.run(["poetry", "run", "nbdev_clean", "--fname", file])
