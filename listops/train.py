import os 
import math 
import torch 
from sys import argv

import numpy as np 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 

from data import ListOpsDataset
from model import ListOpsClassifier

LOG_PATH  = './log/ListOps/mem/'+argv[1]
DATA_PATH = '/cluster/project/sachan/LRA/ListOps/'

TRAIN_FILE = 'train.tsv'
VALID_FILE = 'valid.tsv'
TEST_FILE  = 'test.tsv'

EPOCHS = 4
LOG_FREQ = 100
BATCH_SIZE = 32
WARMUP_STEPS = 1000 
LEARNING_RATE = 5e-3

def get_learning_rate_scheduler(optimizer, num_warmup_steps, last_epoch=-1): 
    """
    Create learning rate scheduler. 

    Arguments:
    ----------
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:
    --------
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int): 
        r = 1.0 

        # Warmup phase
        r *= min(1.0, float(current_step) / float(num_warmup_steps))

        # Decay phase
        r /= math.sqrt(max(current_step, num_warmup_steps))

        return r

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate(model, data_loader, criterion, device): 
    # Turn on the evaluation mode
    model.eval() 

    total_loss = 0.0 

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader): 
            # Unpack batch 
            y = batch['label'].to(device)
            x = batch['tokens'].to(device)
            length_mask = batch['length'].to(device)

            # Call model 
            logits = model(x, length_mask)

            # Accumulate loss 
            total_loss += len(y) * criterion(logits, y).item()

    return total_loss / len(data_loader.dataset)

def predict(model, data_loader, device): 
    # Turn on the evaluation mode
    model.eval() 

    # Initialize outputs
    labels = np.array([])
    predictions = np.array([])
     
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader): 
            # Unpack batch 
            y = batch['label'].to(device)
            x = batch['tokens'].to(device)
            length_mask = batch['length'].to(device)

            # Call model 
            logits = model(x, length_mask)
            probs = nn.functional.softmax(logits, dim=1)
            y_hat = torch.argmax(probs, dim=1)

            labels = np.concatenate((labels, \
                                    y.detach().cpu().numpy()))

            predictions = np.concatenate((predictions, \
                                         y_hat.detach().cpu().numpy()))

    return labels, predictions

if __name__ == '__main__':
    # Specify GPU option 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Specify model parameters
    params = {
        'classifier_dim': 2048, 
        'd_model': 512, 
        'attention_type': argv[1], 
        'n_layers': 6, 
        'n_heads' : 8,
        'd_ff': 2048, 
        'd_query': 64, 
        'd_values': 64,
        'activation': 'gelu',
        'output_norm': True
    }

    # Load datasets 
    train_dataset = ListOpsDataset(
        os.path.join(DATA_PATH, TRAIN_FILE),
        training_set=True 
    )

    valid_dataset = ListOpsDataset(
        os.path.join(DATA_PATH, VALID_FILE),
        vocab=train_dataset.vocab 
    )

    # Dataloaders 
    train_loader = DataLoader(
        train_dataset, 
        BATCH_SIZE, 
        shuffle=True, 
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, 
        BATCH_SIZE, 
        shuffle=False, 
        drop_last=False
    )

    # Model 
    model = ListOpsClassifier(
        n_classes=train_dataset.n_classes,
        vocab_size=train_dataset.vocab_size, 
        max_len=train_dataset.sequence_len,
        **params
    )
    model = torch.nn.DataParallel(model)

    # Set precision 
    model.float()

    # Send to GPU 
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.9, 0.98),
        eps=1e-9, 
        weight_decay=1e-1
    )

    # Scheduler 
    scheduler = get_learning_rate_scheduler(
        optimizer, 
        WARMUP_STEPS
    )

    # Tensorboard
    best_valid_acc = float('-inf')    
    best_valid_loss = float('inf')    
    writer = SummaryWriter(LOG_PATH)

    # Repeat for each epoch 
    for epoch in range(EPOCHS): 
        # Turn on the training mode
        model.train()

        # Loop over batches 
        for train_batch_idx, train_batch in tqdm(enumerate(train_loader)):
            # Unpack batch 
            y = train_batch['label'].to(device)
            x = train_batch['tokens'].to(device)
            length_mask = train_batch['length'].to(device)

            # Reset graph 
            optimizer.zero_grad()

            # Call model 
            logits = model(x, length_mask)

            # Loss 
            loss = criterion(logits, y)
            loss.backward()

            # Optimization step 
            optimizer.step()

            # Update learning rate 
            scheduler.step()

            # Report results
            if train_batch_idx>0 and train_batch_idx % LOG_FREQ == 0:  
                # Learning rate 
                print("Logging Started")
                writer.add_scalar(
                    'Learning Rate', 
                    scheduler.get_last_lr()[0],
                    scheduler._step_count
                )

                # Training loss
                writer.add_scalar(
                    'Training Loss', 
                    loss.item(), 
                    scheduler._step_count
                )

                # Training accuracy 
                y_pred = torch.argmax(
                    nn.functional.softmax(logits, dim=1), 
                    dim=1
                )

                writer.add_scalar(
                    'Training Accuracy', 
                    100*accuracy_score(
                        y.detach().cpu().numpy(), 
                        y_pred.detach().cpu().numpy()
                    ), 
                    scheduler._step_count
                )
                            
                # Validation loss
                valid_loss = evaluate(
                    model, 
                    valid_loader, 
                    criterion, 
                    device
                )
                writer.add_scalar(
                    'Validation Loss', 
                    valid_loss,
                    scheduler._step_count
                )

                # Validation accuracy 
                y_valid, y_valid_pred = predict(
                    model, 
                    valid_loader, 
                    device
                )
                writer.add_scalar(
                    'Validation Accuracy', 
                    100*accuracy_score(y_valid, y_valid_pred), 
                    scheduler._step_count
                )
                valid_accuracy = 100*accuracy_score(y_valid, y_valid_pred)
                if valid_accuracy > best_valid_acc:
                    best_valid_acc = valid_accuracy
                    torch.save(
                        model.state_dict(), 
                        os.path.join(LOG_PATH, 'accuracy-best.pt')
                    )

                # Reset training mode 
                model.train()
        # Evaluate the model 
        valid_loss = evaluate(
            model, 
            valid_loader, 
            criterion, 
            device
        )
        y_valid, y_valid_pred = predict(
            model, 
            valid_loader, 
            device
        )
        valid_accuracy = 100*accuracy_score(y_valid, y_valid_pred)

        # Save best validation loss model 
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(), 
                os.path.join(LOG_PATH, 'loss-best.pt')
            )

        # Save best validation accuracy model 
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(
                model.state_dict(), 
                os.path.join(LOG_PATH, 'accuracy-best.pt')
            )

    writer.close()

    # Save final model 
    torch.save(
        model.state_dict(), 
        os.path.join(LOG_PATH, 'final.pt')
    )

    memory_stats = torch.cuda.memory_stats()
    max_memory_reserved = torch.cuda.max_memory_reserved(device=device)
    max_memory_allocated = torch.cuda.max_memory_allocated(device=device)

    # Test dataset 
    test_dataset = ListOpsDataset(
        os.path.join(DATA_PATH, TEST_FILE),
        vocab=train_dataset.vocab 
    )

    test_loader = DataLoader(
        test_dataset, 
        BATCH_SIZE, 
        shuffle=False, 
        drop_last=False
    )

    print('=' * 45, ' Final Checkpoint ', '=' * 45)

    # Training set 
    train_loss = evaluate(model, train_loader, criterion, device)
    y_train, y_train_pred = predict(model, train_loader, device)
    C_train, C_train_counts = np.unique(y_train, return_counts=True)

    # Validation set 
    valid_loss = evaluate(model, valid_loader, criterion, device)
    y_valid, y_valid_pred = predict(model, valid_loader, device)
    C_valid, C_valid_counts = np.unique(y_valid, return_counts=True)

    # Testing set 
    test_loss = evaluate(model, test_loader, criterion, device)
    y_test, y_test_pred = predict(model, test_loader, device)
    C_test, C_test_counts = np.unique(y_test, return_counts=True)

    print('| Training Loss {:5.4f} | Training Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        train_loss, 100*accuracy_score(y_train, y_train_pred), 
        100*f1_score(y_train, y_train_pred, average='micro')))

    print('| Validation Loss {:5.4f} | Validation Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        valid_loss, 100*accuracy_score(y_valid, y_valid_pred), 
        100*f1_score(y_valid, y_valid_pred, average='micro')))

    print('| Testing Loss {:5.4f} | Testing Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        test_loss, 100*accuracy_score(y_test, y_test_pred), 
        100*f1_score(y_test, y_test_pred, average='micro')))

    print('=' * 45, ' Loss Checkpoint ', '=' * 45)

    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(LOG_PATH, 'loss-best.pt')))

    # Training set 
    train_loss = evaluate(model, train_loader, criterion, device)
    y_train, y_train_pred = predict(model, train_loader, device)
    C_train, C_train_counts = np.unique(y_train, return_counts=True)

    # Validation set 
    valid_loss = evaluate(model, valid_loader, criterion, device)
    y_valid, y_valid_pred = predict(model, valid_loader, device)
    C_valid, C_valid_counts = np.unique(y_valid, return_counts=True)

    # Testing set 
    test_loss = evaluate(model, test_loader, criterion, device)
    y_test, y_test_pred = predict(model, test_loader, device)
    C_test, C_test_counts = np.unique(y_test, return_counts=True)

    print('| Training Loss {:5.4f} | Training Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        train_loss, 100*accuracy_score(y_train, y_train_pred), 
        100*f1_score(y_train, y_train_pred, average='micro')))

    print('| Validation Loss {:5.4f} | Validation Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        valid_loss, 100*accuracy_score(y_valid, y_valid_pred), 
        100*f1_score(y_valid, y_valid_pred, average='micro')))

    print('| Testing Loss {:5.4f} | Testing Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        test_loss, 100*accuracy_score(y_test, y_test_pred), 
        100*f1_score(y_test, y_test_pred, average='micro')))

    print('=' * 45, ' Accuracy Checkpoint ', '=' * 45)

    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(LOG_PATH, 'accuracy-best.pt')))

    # Training set 
    train_loss = evaluate(model, train_loader, criterion, device)
    y_train, y_train_pred = predict(model, train_loader, device)
    C_train, C_train_counts = np.unique(y_train, return_counts=True)

    # Validation set 
    valid_loss = evaluate(model, valid_loader, criterion, device)
    y_valid, y_valid_pred = predict(model, valid_loader, device)
    C_valid, C_valid_counts = np.unique(y_valid, return_counts=True)

    # Testing set 
    test_loss = evaluate(model, test_loader, criterion, device)
    y_test, y_test_pred = predict(model, test_loader, device)
    C_test, C_test_counts = np.unique(y_test, return_counts=True)

    print('| Training Loss {:5.4f} | Training Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        train_loss, 100*accuracy_score(y_train, y_train_pred), 
        100*f1_score(y_train, y_train_pred, average='micro')))

    print('| Validation Loss {:5.4f} | Validation Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        valid_loss, 100*accuracy_score(y_valid, y_valid_pred), 
        100*f1_score(y_valid, y_valid_pred, average='micro')))

    print('| Testing Loss {:5.4f} | Testing Accuracy {:8.2f} | F1 Score {:8.2f}'.format(\
        test_loss, 100*accuracy_score(y_test, y_test_pred), 
        100*f1_score(y_test, y_test_pred, average='micro')))

    print('=' * 45, ' Memory Statistics ', '=' * 45)

    print('Max Memory Reserved: ', max_memory_reserved)
    print('Max Memory Allocated: ', max_memory_allocated)

    peak_bytes_requirement = memory_stats["allocated_bytes.all.peak"]

    print('Peak Allocated Bytes: ', peak_bytes_requirement)




