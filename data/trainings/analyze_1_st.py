import os
import re
import ast
from statistics import mean


def parse_first_stage(file_path: str):
	"""Parse first_stage.txt to extract:
	- Averages of consecutive [GrandAlignDebug] grand_loss blocks
	- Raw 'loss' values from summary dict lines

	Returns (grand_loss_avgs, loss_values)
	"""
	re_grand = re.compile(r"^\[GrandAlignDebug\]\s+grand_loss=(\d+\.\d+)\s*$")

	grand_loss_avgs = []  # averaged per consecutive block
	loss_values = []      # raw from dict lines
	current_block = []    # accumulating grand_loss values until dict line

	with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			s = line.strip()
			if not s:
				continue

			m = re_grand.match(s)
			if m:
				try:
					current_block.append(float(m.group(1)))
				except ValueError:
					# Skip malformed numeric values
					pass
				continue

			# Try to capture dict summary lines like:
			# {'loss': 2.2384, 'learning_rate': 0.00099..., 'epoch': 0.07}
			if s.startswith("{") and "'loss':" in s:
				try:
					d = ast.literal_eval(s)
					if isinstance(d, dict) and "loss" in d:
						# finalize the block average up to this dict
						if current_block:
							grand_loss_avgs.append(mean(current_block))
							current_block = []
						else:
							# keep alignment; NaN will be ignored in plots
							grand_loss_avgs.append(float("nan"))

						# store raw loss
						try:
							loss_values.append(float(d["loss"]))
						except (TypeError, ValueError):
							# keep alignment; NaN if malformed
							loss_values.append(float("nan"))
						continue
				except (SyntaxError, ValueError):
					# Not a clean dict line; ignore
					pass

			# ignore all other lines

	# Handle trailing block (if file ended without a dict line)
	if current_block:
		grand_loss_avgs.append(mean(current_block))
		# keep alignment with dict-based series
		loss_values.append(float("nan"))

	return grand_loss_avgs, loss_values


def plot_losses(gl_avgs, raw_losses, out_path: str):
	try:
		import matplotlib.pyplot as plt
		import math
	except ImportError:
		print("matplotlib is not installed. Install it with: pip install matplotlib")
		return

	# Ensure same length for indexing; pad shorter with NaN
	n = max(len(gl_avgs), len(raw_losses))
	def pad(xs, length):
		if len(xs) < length:
			return xs + [float("nan")] * (length - len(xs))
		return xs

	gl_avgs = pad(gl_avgs, n)
	raw_losses = pad(raw_losses, n)

	x = list(range(n))

	fig, ax1 = plt.subplots(figsize=(10, 5))

	l1, = ax1.plot(x, raw_losses, color="tab:blue", marker="o", linestyle="-", linewidth=1.2, markersize=3,
				   label="CE Loss")
	ax1.set_xlabel("Batch")
	ax1.set_ylabel("CE Loss", color="tab:blue")
	ax1.tick_params(axis='y', labelcolor='tab:blue')

	ax2 = ax1.twinx()
	l2, = ax2.plot(x, gl_avgs, color="tab:orange", marker="x", linestyle="--", linewidth=1.2, markersize=3,
				   label="Alignment Loss")
	ax2.set_ylabel("Alignment Loss", color="tab:orange")
	ax2.tick_params(axis='y', labelcolor='tab:orange')

	# Build a combined legend
	lines = [l1, l2]
	labels = [str(l.get_label()) for l in lines]
	ax1.legend(lines, labels, loc="upper right")

	ax1.grid(True, linestyle=":", alpha=0.4)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	print(f"Saved plot to: {out_path}")


def main():
	here = os.path.dirname(os.path.abspath(__file__))
	in_path = os.path.join(here, "first_stage.txt")
	if not os.path.isfile(in_path):
		print(f"Could not find file: {in_path}")
		return

	print(f"Parsing: {in_path}")
	gl_avgs, raw_losses = parse_first_stage(in_path)

	print(f"GrandAlign blocks parsed: {len(gl_avgs)}")
	print(f"Raw loss entries parsed: {len(raw_losses)}")

	out_path = os.path.join(here, "first_stage_losses.png")
	plot_losses(gl_avgs, raw_losses, out_path)


if __name__ == "__main__":
	main()

