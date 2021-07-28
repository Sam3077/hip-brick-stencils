import csv
import sys

if len(sys.argv) != 2:
    print("Filename must be specified")
    exit

def duration(a):
    flops = pow(512, 3) * 13 / int(a)
    return f"Duration: {a} ns\nFLOPS: {'{:.3f}'.format(flops)} GFLOPS/s"

proc_dict = {
    "KernelName": lambda a: f"===== {a} =====",
    "FETCH_SIZE": lambda a: f"Fetch size: {a} KB",
    "WRITE_SIZE": lambda a: f"Write size: {a} KB",
    "SQ_INSTS_SMEM": lambda a: f"SMEM instructions issued: {a}",
    "DurationNs": duration,
    "TCC_HIT_sum": lambda a: f"Total cache hits: {a}",
    "TCC_EA_RDREQ_sum": lambda a: f"Read requests: {a}",
    "WRITE_REQ_32B": lambda a: f"32B write requests: {a}",
    "TCC_HIT_sum": lambda a: f"Cache hits: {a}",
    "TCC_MISS_sum": lambda a: f"Cache misses: {a}"
}

with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    title = reader.__next__()
    
    # name_index = title.index("KernelName")
    # duration_index = title.index("DurationNs")
    # fetch_index = title.index("FETCH_SIZE")
    # write_index = title.index("WRITE_SIZE")
    
    for row in reader:
        for item in zip(title, row):
            item_name = item[0]
            a = proc_dict.get(item_name, lambda a: None)(item[1])
            if a is not None:
                print(a)
        print("\n")
        # flops = pow(512, 3) * 13 / int(row[duration_index])
        # print(f"\n===== {row[name_index]} =====\nDuration: {row[duration_index]} ns\nFLOPS: {'{:.3f}'.format(flops)} GFlops/s\nFetch size: {row[fetch_index]} kb\nWrite size: {row[write_index]} kb\n")