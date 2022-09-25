import config
import torch
import torch.optiom as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset.dataset import TextDataset


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets = [d.to(device) for d in data]

    target_lengths = torch.LongTensor([i.size()[0] for i in targets])

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

    target_lengths = torch.flatten(target_lengths)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()


def train_loop(data_path, imgs_list, model, optimizer, criterion=nn.CTCLoss(blank=0)):
    epochs = config.epochs
    device = config.device

    image_names_train, image_names_test = train_test_split(imgs_list, random_state=config.SEED)
    trainset = TextDataset(data_path, image_names_train)
    testset = TextDataset(data_path, image_names_test)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=True)

    epoch_losses = []
    num_updates_epochs = []

    for epoch in range(1, epochs + 1):
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:

            loss = train_batch(model, train_data, optimizer, criterion, device)

            epoch_losses.append(loss)
            num_updates_epochs.append(epoch)

            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size

        # lr_scheduler.step(tot_train_loss / tot_train_count)
        # scheduler.step()
        print(f'epoch: {epoch}', end='\t')
        print('train_loss: ', tot_train_loss / tot_train_count)