import re
# Before pandas: importing slurm_gpustat pins the BLAS thread pools, which only
# takes effect if numpy has not been loaded yet.
from slurm_gpustat import (
    MONITORED_PARTITIONS, partition_nodes, node_states, is_inaccessible,
    reserved_for_others, node_records, parse_gres_gpus, alloc_gpu_count,
    avail_stats_for_node,
)
import pandas as pd


def get_node_names():
    """Every node of the monitored partitions, warning about partitions with none."""
    nodes = partition_nodes()
    for partition in MONITORED_PARTITIONS.split(","):
        if partition not in nodes:
            print("No nodes found for partition:", partition)
    return set().union(*nodes.values())


def get_node_resources(node_name, record):
    """What is still free on one node, from its `scontrol show node` record."""
    stats = avail_stats_for_node(node_name)
    cpus_available = stats["cpu"][0]
    memory_available_gb = (stats["mem"][0] // 1024 ** 2 // 102.4) / 10

    gpus_total = sum(count or 0 for _, count in parse_gres_gpus(record.get("Gres", "")))
    gpus_available = gpus_total - alloc_gpu_count(record.get("AllocTRES", "")) if gpus_total else 0

    features = re.match(r"[\w,]*", record.get("AvailableFeatures", "")).group(0)
    gpu_type = features.split(",") if features else []
    gpu_type = [tp for tp in gpu_type if "gb" in tp] if len(gpu_type) > 1 else gpu_type
    gpu_type_str = ", ".join(gpu_type) if gpu_type else None

    return {
        'Node': node_name,
        'Available CPUs': cpus_available,
        'Available Memory (GB)': memory_available_gb,
        'Available GPUs': gpus_available,
        'GPU Type': gpu_type_str
    }


def main():
    # One set of nodes: the union partitions (gpu, interactive) hold the same nodes
    # as the typed ones, so checking partition by partition would list them twice.
    all_nodes = get_node_names()

    # A drained node or one reserved for another group shows idle GPUs that no job
    # of ours can get, so leave those out, the same way the web page does.
    states = node_states(partition=MONITORED_PARTITIONS)
    reserved = reserved_for_others()
    closed = {node: states.get(node, "missing from sinfo") for node in all_nodes
              if node not in states or is_inaccessible(states[node])}
    held = {node: reserved[node] for node in all_nodes if node in reserved and node not in closed}
    if closed:
        print(f"Skipping {len(closed)} nodes not taking jobs: "
              + ", ".join(f"{n} ({closed[n]})" for n in sorted(closed)))
    if held:
        print(f"Skipping {len(held)} nodes reserved for other groups: "
              + ", ".join(f"{n} ({held[n]})" for n in sorted(held)))

    records = node_records()
    all_resources = []
    for node in sorted(all_nodes - set(closed) - set(held)):
        if node not in records:
            print(f"Warning: no scontrol record for node {node}, skipping")
            continue
        all_resources.append(get_node_resources(node, records[node]))

    if all_resources:
        df = pd.DataFrame(all_resources)
        df_filtered = df[(df['Available GPUs'] > 0) & (df['Available CPUs'] > 0) & (df['Available Memory (GB)'] >= 6)]
        print(df_filtered)


if __name__ == "__main__":
    main()
