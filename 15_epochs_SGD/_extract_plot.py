import re

log_data = """
Epoch 1/15
Train Loss: 0.43101 | Train Acc: 0.80288 | Val Loss: 0.44182 | Val Acc: 0.79738 | Test Loss: 0.41484 | Test Acc: 0.81099
Saved best model by loss at epoch 1 with test loss 0.41484
Saved best model by accuracy at epoch 1 with test acc 0.81099
Saved best model by validation accuracy at epoch 1 with val acc 0.79738
Epoch 2/15
Train Loss: 0.34552 | Train Acc: 0.85381 | Val Loss: 0.34735 | Val Acc: 0.84778 | Test Loss: 0.32559 | Test Acc: 0.86946
Saved best model by loss at epoch 2 with test loss 0.32559
Saved best model by accuracy at epoch 2 with test acc 0.86946
Saved best model by validation accuracy at epoch 2 with val acc 0.84778
Epoch 3/15
Train Loss: 0.32733 | Train Acc: 0.85965 | Val Loss: 0.31095 | Val Acc: 0.86946 | Test Loss: 0.29739 | Test Acc: 0.88306
Saved best model by loss at epoch 3 with test loss 0.29739
Saved best model by accuracy at epoch 3 with test acc 0.88306
Saved best model by validation accuracy at epoch 3 with val acc 0.86946
Epoch 4/15
Train Loss: 0.31364 | Train Acc: 0.86754 | Val Loss: 0.29577 | Val Acc: 0.87500 | Test Loss: 0.28615 | Test Acc: 0.88810
Saved best model by loss at epoch 4 with test loss 0.28615
Saved best model by accuracy at epoch 4 with test acc 0.88810
Saved best model by validation accuracy at epoch 4 with val acc 0.87500
Epoch 5/15
Train Loss: 0.30273 | Train Acc: 0.87316 | Val Loss: 0.29074 | Val Acc: 0.87298 | Test Loss: 0.28088 | Test Acc: 0.88861
Saved best model by loss at epoch 5 with test loss 0.28088
Saved best model by accuracy at epoch 5 with test acc 0.88861
Epoch 6/15
Train Loss: 0.29638 | Train Acc: 0.87327 | Val Loss: 0.27804 | Val Acc: 0.87903 | Test Loss: 0.27219 | Test Acc: 0.89466
Saved best model by loss at epoch 6 with test loss 0.27219
Saved best model by accuracy at epoch 6 with test acc 0.89466
Saved best model by validation accuracy at epoch 6 with val acc 0.87903
Epoch 7/15
Train Loss: 0.28423 | Train Acc: 0.88203 | Val Loss: 0.27541 | Val Acc: 0.87298 | Test Loss: 0.26886 | Test Acc: 0.89113
Saved best model by loss at epoch 7 with test loss 0.26886
Epoch 8/15
Train Loss: 0.28936 | Train Acc: 0.87900 | Val Loss: 0.27015 | Val Acc: 0.88357 | Test Loss: 0.26659 | Test Acc: 0.89415
Saved best model by loss at epoch 8 with test loss 0.26659
Saved best model by validation accuracy at epoch 8 with val acc 0.88357
Epoch 9/15
Train Loss: 0.28503 | Train Acc: 0.88214 | Val Loss: 0.26623 | Val Acc: 0.88609 | Test Loss: 0.26560 | Test Acc: 0.89264
Saved best model by loss at epoch 9 with test loss 0.26560
Saved best model by validation accuracy at epoch 9 with val acc 0.88609
Epoch 10/15
Train Loss: 0.28871 | Train Acc: 0.87738 | Val Loss: 0.27064 | Val Acc: 0.88206 | Test Loss: 0.26427 | Test Acc: 0.89264
Saved best model by loss at epoch 10 with test loss 0.26427
Epoch 11/15
Train Loss: 0.27807 | Train Acc: 0.88549 | Val Loss: 0.26615 | Val Acc: 0.88760 | Test Loss: 0.26402 | Test Acc: 0.89365
Saved best model by loss at epoch 11 with test loss 0.26402
Saved best model by validation accuracy at epoch 11 with val acc 0.88760
Epoch 12/15
Train Loss: 0.27388 | Train Acc: 0.88744 | Val Loss: 0.26589 | Val Acc: 0.88659 | Test Loss: 0.26444 | Test Acc: 0.89970
Saved best model by accuracy at epoch 12 with test acc 0.89970
Epoch 13/15
Train Loss: 0.28063 | Train Acc: 0.88397 | Val Loss: 0.26635 | Val Acc: 0.88760 | Test Loss: 0.26642 | Test Acc: 0.89113
Epoch 14/15
Train Loss: 0.27484 | Train Acc: 0.88538 | Val Loss: 0.27003 | Val Acc: 0.87954 | Test Loss: 0.26326 | Test Acc: 0.89062
Saved best model by loss at epoch 14 with test loss 0.26326
Epoch 15/15
Train Loss: 0.26928 | Train Acc: 0.88862 | Val Loss: 0.27320 | Val Acc: 0.88155 | Test Loss: 0.26918 | Test Acc: 0.89163
"""

# Initialize empty lists to store the metrics
train_loss = []
train_acc = []
val_loss = []
val_acc = []
test_loss = []
test_acc = []

# Regular expression to capture the floating-point numbers for each metric
# We use named capturing groups for clarity
pattern = re.compile(
    r"Train Loss: (?P<train_loss>\d+\.\d+)"
    r" \| Train Acc: (?P<train_acc>\d+\.\d+)"
    r" \| Val Loss: (?P<val_loss>\d+\.\d+)"
    r" \| Val Acc: (?P<val_acc>\d+\.\d+)"
    r" \| Test Loss: (?P<test_loss>\d+\.\d+)"
    r" \| Test Acc: (?P<test_acc>\d+\.\d+)"
)

# Split the log data into lines and iterate
for line in log_data.splitlines():
    match = pattern.search(line)
    if match:
        # Extract the values from the named groups and convert to float
        train_loss.append(float(match.group('train_loss')))
        train_acc.append(float(match.group('train_acc')))
        val_loss.append(float(match.group('val_loss')))
        val_acc.append(float(match.group('val_acc')))
        test_loss.append(float(match.group('test_loss')))
        test_acc.append(float(match.group('test_acc')))

# Print the extracted lists
print("Train_Loss=", train_loss)
print("Train_Acc=", train_acc)
print("Val_Loss=", val_loss)
print("Val_Acc=", val_acc)
print("Test_Loss=", test_loss)
print("Test_Acc=", test_acc)