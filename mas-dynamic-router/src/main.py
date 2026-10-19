import jsonlines
from router import Router
from config import Config

def main():
    # Load configuration
    config = Config()
    
    # Initialize the Router
    router = Router(config)

    # Load SLM results from the data file
    with jsonlines.open('data/sllm_results.jsonl') as reader:
        for sample in reader:
            # Process each sample through the Router
            result = router.route(sample)
            print(result)

if __name__ == "__main__":
    main()