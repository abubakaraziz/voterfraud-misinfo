
#!/usr/local/bin/python3
import argparse, os
import pandas as pd
from models import ClassificationModel, ClassificationArgs


# Global argument.
parser = argparse.ArgumentParser()

parser.add_argument("--train_data", type=str, default="label",
                    help="Which dataset to train?")

parser.add_argument("--dev_data", type=str, default="label",
                    help="Which dataset to develop?")
parser.add_argument("--test_data", type=str, default="label",
                    help="Which dataset to test?")
parser.add_argument("--split", type=str, default="0.85/0/0.15",
                    help="How to split train/dev/test sets?")
parser.add_argument("--model_type", type=str, default="roberta",
                    help="What type of model to use?")
parser.add_argument("--model_name", type=str, default="roberta-base",
                    help="Which model to use?")
parser.add_argument("--train_epochs", type=int, default=2,
                    help="How many epochs to train?")
args, _ = parser.parse_known_args()

event="voter_fraud"
# Read train/dev/test sets.
train_path = os.path.join("..", "data", event , args.train_data, "test.csv")
dev_path = os.path.join("..", "data", event , args.dev_data, "test.csv")
test_path = os.path.join("..", "data", event, args.test_data, "test.csv")
train_r, dev_r, test_r = [float(r) for r in args.split.split("/")]
_train_r, _dev_r, _test_r = train_r, dev_r, test_r
train = pd.read_csv(train_path)



train = train.sample(frac=train_r)
print("Train data loaded:\t", args.train_data, "\t", len(train))
if dev_r > 0:
    dev = pd.read_csv(dev_path)
    if dev_path == train_path:
        dev = dev.drop(train.index)
        dev_r = min(dev_r / (1 - _train_r), 1)
    dev = dev.sample(frac=dev_r)
else:
    dev = pd.DataFrame()
print("Dev data loaded:\t", args.dev_data, "\t", len(dev))

if test_r > 0:
    test = pd.read_csv(test_path)
    if test_path == train_path and test_path == dev_path:
        test = test.drop(train.index).drop(dev.index)
        test_r = min(test_r / (1 - _train_r - _dev_r), 1)
    test = test.sample(frac=test_r)
else:
    test = pd.DataFrame()
print("Test data loaded:\t", args.test_data, "\t", len(test))

# Model configuration.
model_args = ClassificationArgs()
model_args.num_train_epochs=args.train_epochs
model_args.labels_list = [0, 1]
model_args.overwrite_output_dir = True

# Configure the model.
model = ClassificationModel(args.model_type, args.model_name, args=model_args)

# Train the model.
model.train_model(train)


# Evaluate the model.
if len(dev) > 0:
    result, model_outputs, wrong_predictions = model.eval_model(dev)
    print("Dev set:\t", result)


# Test the model.
if len(test) > 0:
    result, model_outputs, wrong_predictions = model.eval_model(test)
    print("Test set:\t", result)
    #print("Model_output:\t",model_outputs)
    #print("Wrong_Predictions:\t",wrong_predictions)
    # for output, truth, text in zip(model_outputs, test["labels"], test["text"]):
    #     print(output, truth, text[:200])

