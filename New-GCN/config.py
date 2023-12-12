import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cora")
parser.add_argument("--method", default="AGE+FDS+SDS")
parser.add_argument("--learning_rate", default=0.01)
parser.add_argument("--epochs", default=300)
parser.add_argument("--hidden1", default=16)
parser.add_argument("--dropout", default=0.5)
parser.add_argument("--weight_decay", default=5e-4)
parser.add_argument("--early_stopping", default=10)
parser.add_argument("--max_degree", default=3)
parser.add_argument("--inicount", default=4)

args = parser.parse_args()
