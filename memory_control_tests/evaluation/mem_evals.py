from .cost import log_cost_delta, snapshot_openrouter_usage
from .config import parse_eval_config
from .runner import run_evaluation, write_evaluation_results


def main() -> None:
    config = parse_eval_config()
    cost_start = snapshot_openrouter_usage(config.api_key_file)
    try:
        results = run_evaluation(config)
        print(write_evaluation_results(config, results))
    finally:
        log_cost_delta(cost_start, snapshot_openrouter_usage(config.api_key_file))


if __name__ == "__main__":
    main()
