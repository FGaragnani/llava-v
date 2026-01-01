from pathlib import Path
import plotext

LOGS_DIR = Path("/leonardo_scratch/large/userexternal/fgaragna/logs")

def main():
    out_files = sorted(LOGS_DIR.glob("*.out"))
    for out_file in out_files:
        print(f"<--- {out_file.name} --->")
        grand_align_data = []
        ce_loss_data = []
        with open(out_file, "r", encoding="utf-8") as f:
            grand_align_debug = []
            found = False
            for line in f:
                if "[GrandAlignDebug]" in line:
                    found = True
                    try:
                        grand_align_debug.append(float(line.split("=")[1]))
                    except ValueError:
                        first_part = line.split("=")[1]
                        grand_align_debug.append(float(first_part.split("[")[0]))
                        grand_align_debug.append(float(line.split("=")[2]))
                else:
                    if not found:
                        continue
                    if line.strip() == "":
                        continue
                    try:
                        ce_loss_data.append(float(line.split(":")[1].split(",")[0]))
                    except IndexError:
                        print("Could not parse CE Loss line:", line)
                    grand_align_data.append(sum(grand_align_debug)/len(grand_align_debug) if grand_align_debug else 0.0)
                    grand_align_debug = []
        plotext.clf()
        plotext.plot(grand_align_data, label="GrandAlignDebug")
        # plotext.plot(ce_loss_data, label="CE Loss")
        plotext.title(f"Logs from {out_file.name}")
        plotext.xlabel("Iterations")
        plotext.ylabel("Value")
        plotext.show()

if __name__ == "__main__":
    main()