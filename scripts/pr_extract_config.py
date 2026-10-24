"""Extract the FLAGS.config YAML block from a run's output.log into a file."""
import re
import sys

log_path, out_path = sys.argv[1], sys.argv[2]
lines = open(log_path, errors="replace").read().splitlines()
start = next(i for i, l in enumerate(lines) if "FLAGS.config:" in l)
block = []
for l in lines[start + 1:]:
    if re.match(r"^[IWEF]\d{4} ", l):
        break
    block.append(l)
open(out_path, "w").write("\n".join(block) + "\n")
print(f"{len(block)} lines -> {out_path}")
