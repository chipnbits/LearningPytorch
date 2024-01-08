from torch import no_grad

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += (logits.argmax(dim=1) == y).float().mean().item()

    return running_loss / len(train_loader), running_accuracy / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    with no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item()
            running_accuracy += (logits.argmax(dim=1) == y).float().mean().item()

    return running_loss / len(val_loader), running_accuracy / len(val_loader)