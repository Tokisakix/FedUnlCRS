from partition import run_partition

task = "all"
dataset = "opendialkg"
partition_model = "mlp"

if __name__ == "__main__":
    if task in ["partition", "all"]:
        run_partition(dataset, partition_model)