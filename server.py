import flwr as fl
import os
import requests
import json
# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def send_to_blockchain(round_num, metrics):
def send_to_blockchain(round_num, acc):
    """Send aggregated metrics to the API endpoint."""
    url = "http://host.docker.internal:3001/store-metrics"  # Replace with actual API URL
    
    data = {
        "round": round_num,
        "accuracy": int(acc*100),  # Convert to fixed-point integer
        "loss": 0
    }
    print(data)
    headers = {"Content-Type": "application/json"}
    
    try:
        # response = requests.post(url, data=json.dumps(data), headers=headers)
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Metrics sent to API successfully: {response.json()}")
        else:
            print(f"Error sending metrics: {response.text}")
    except Exception as e:
        print(f"Exception occurred while sending metrics: {e}")

def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / total_examples for k, v in federated_metrics.items()}

def get_server_strategy():
    return fl.server.strategy.FedAvg(
            min_fit_clients=3,
            min_evaluate_clients = 3,
            min_available_clients=3,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
    # print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
    print(history.metrics_distributed)
    send_to_blockchain(final_round, acc)
