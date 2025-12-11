import os
import ast
import argparse


def parse_raw_losses(file_path: str):
	"""Parse lines like {'loss': 1.9581, 'learning_rate': ..., 'epoch': ...}
	and return a list of loss floats in appearance order.
	"""
	losses = []
	if not os.path.isfile(file_path):
		raise FileNotFoundError(file_path)

	with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			s = line.strip()
			if not s:
				continue
			# quick filter to avoid expensive evals
			if not s.startswith("{") or "'loss':" not in s:
				continue
			try:
				d = ast.literal_eval(s)
				if isinstance(d, dict) and "loss" in d:
					try:
						losses.append(float(d["loss"]))
					except (TypeError, ValueError):
						# skip malformed values
						continue
			except Exception:
				# ignore unparsable lines
				continue

	return losses


def plot_losses(losses, out_path: str):
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		print("matplotlib not installed. Install with: pip install matplotlib")
		return

	if not losses:
		print("No loss values found; nothing to plot.")
		return

	x = list(range(len(losses)))

	plt.figure(figsize=(10, 4.5))
	plt.plot(x, losses, marker='o', markersize=3, linewidth=1.2, color='tab:blue')
	plt.xlabel('Batch')
	plt.ylabel('Loss')
	plt.title('CE Loss')
	plt.grid(alpha=0.35, linestyle=':')
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()
	print(f"Saved raw loss plot to: {out_path}")


def main():
	parser = argparse.ArgumentParser(description="Parse raw dict loss lines and plot them.")
	parser.add_argument('--input', '-i', default='second_stage.txt', help='Input log file (default: second_stage.txt)')
	parser.add_argument('--output', '-o', default='second_stage_losses.png', help='Output image path')
	args = parser.parse_args()

	here = os.path.dirname(os.path.abspath(__file__))
	in_path = args.input if os.path.isabs(args.input) else os.path.join(here, args.input)
	out_path = args.output if os.path.isabs(args.output) else os.path.join(here, args.output)

	try:
		losses = parse_raw_losses(in_path)
	except FileNotFoundError:
		print(f"Input file not found: {in_path}")
		return

	print(f"Parsed {len(losses)} loss entries from {in_path}")
	plot_losses(losses, out_path)


if __name__ == '__main__':
	main()
