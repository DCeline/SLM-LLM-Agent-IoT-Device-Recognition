import jsonlines

def read_sllm_results(file_path: str) -> list:
    results = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            results.append(obj)
    return results

def write_results(file_path: str, results: list) -> None:
    with jsonlines.open(file_path, mode='a') as writer:
        writer.write_all(results)

def print_results(results: list) -> None:
    for result in results:
        print(result)