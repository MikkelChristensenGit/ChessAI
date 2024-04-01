from utils import *

# Load data from Kaggle - Data is games played by players on LiChess
raw_chess_data = pd.read_csv('chess_games.csv', usecols=['AN', 'WhiteElo'])
chess_data_train, chess_data_test = select_data(raw_chess_data, 2000, 2500, 50000, 55000) # 50000 training games 2000-2500 elo
del raw_chess_data # Delete big file from memory when we don't need it anymore
gc.collect()
# Check dimensions
print(f"Training data shape: {chess_data_train.shape}")
print(f"Training data shape: {chess_data_test.shape}")

# PyTorch Dataset & DataLoader
class ChessDataset(Dataset):

    def __init__(self, games):
        super(ChessDataset, self).__init__()
        self.games = games

    # Select the amount of moves the CNN will process
    def __len__(self):
        return 150000

    def __getitem__(self, index):
        """
        Picks a random match from the dataset and a random move from that match.
        Performs all the moves up to the randomly chosen move.
        Converts the final board state to matrices
        """
        game_i = np.random.randint(self.games.shape[0])
        random_game = chess_data_train['AN'].values[game_i]
        moves = create_move_list(random_game)
        game_state_i = np.random.randint(len(moves)-1)
        next_move = moves[game_state_i]
        moves = moves[:game_state_i]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        x = board_2_rep(board)
        y = move_2_rep(next_move, board)
        if game_state_i % 2 == 1: # True if Black's turn
            x *= -1
        return x, y

# Learning Parameters
batch_size = 32 # Batch size
learning_rate = 0.001 # Learning rate
num_epochs = 10  # Number of epochs
# Define loss functions for both metrics
metric_from = nn.CrossEntropyLoss() # The piece that is moved
metric_to = nn.CrossEntropyLoss() # The square it is moved to

data_train = ChessDataset(chess_data_train['AN'])
data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

# Define the neural network model
class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()

    def forward(self, x):
        x_input = torch.clone(x) # Copy of input tensor -> Used in residual connection
        x = self.conv1(x) # Convolutional layer
        x = self.bn1(x) # Batch Normalization
        x = self.activation1(x) # Activation function
        x = self.conv2(x) # Convolutional layer
        x = self.bn2(x) # Batch Normalization
        x = x + x_input # Residual Connection
        x = self.activation2(x) # Activation function
        return x

class chessCNN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=128):
        super(chessCNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x

# Instantiate your model
model = chessCNN()
# If GPU is available, move the model to it
if torch.cuda.is_available():
    to_device(model, device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

correct_from = 0
correct_to = 0
total_len = 0
# Track loss and accuracy
history_loss = []
history_acc = []
history_batch = []
history_epoch = []


# Training loop
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for i, data in enumerate(data_train_loader, 0):
        X, y = data  # Assuming your data loader returns inputs and labels
        X = X.float() # Convert to float
        if torch.cuda.is_available():  # If GPU is availble, store tensors on it
            X, y = X.to(device), y.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        pred = model(X)
    
        # Check accuracy for both metrics
        for j in range(0, batch_size):
            max_pred_from = torch.argmax(pred[j,0,:,:]) # Find index of max output
            max_pred_to = torch.argmax(pred[j,1,:,:])
            
            max_y_from = torch.argmax(y[j,0,:,:]) # Find index of max target
            max_y_to = torch.argmax(y[j,1,:,:])
            
            if max_y_from == max_pred_from: # If it found the right one
                correct_from += 1
            if max_y_to == max_pred_to:
                correct_to += 1
            
        total_len += batch_size #
        accuracy = (correct_from + correct_to) / (total_len*2)


            
        # Compute the loss for the "from" and "to" predictions
        loss_from = metric_from(pred[:, 0, :], y[:, 0, :]) # .argmax(dim=1))
        loss_to = metric_to(pred[:, 1, :], y[:, 1, :]) # .argmax(dim=1))
        
        # Combine the losses
        loss = loss_from + loss_to
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss) + '  Accuracy: ' + str(accuracy))

            # Append and reset training history
            history_batch.append(i+1)
            history_epoch.append(epoch+1)
            history_loss.append(running_loss)
            history_acc.append(accuracy)
            running_loss = 0.0
            correct_from = 0
            correct_to = 0
            total_len = 0
            
print('Finished Training')
torch.save(model.state_dict(), "chessCNN_weights.pt")
print('Model successfully saved!')
