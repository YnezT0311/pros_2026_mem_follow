from .config import parse_eval_config
from .runner import run_evaluation, write_evaluation_results


def main() -> None:
    config = parse_eval_config()
    results = run_evaluation(config)
    print(write_evaluation_results(config, results))


if __name__ == "__main__":
    main()
