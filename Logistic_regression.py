import sys
from a1_p1_last import wordTokenizer, spacelessBPETokenize,spacelessBPELearn
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F


#----------Checkpoint 2.1-------------------------------------------------
# Define a method to extract lexical features
def extractLexicalFeatures(tokens, target, vocabulary, tokenizer_type):
    # Initialize feature vectors
    vocabulary_size = 501
    unknown_word_token = vocabulary_size - 1
    one_hot_encoding = [0] * vocabulary_size
    multi_hot_encoding = [0] * vocabulary_size

    # Tokenize input sentence based on the tokenizer type
    tokenizer = wordTokenizer
    if tokenizer_type == 'bpe':
        tokenizer = spacelessBPETokenize
    tokenized_sentence = tokenizer(tokens[target])

    document_tokens = tokens

    # Create one-hot encoding of the word preceding the target word
    if target - 3 > 0:
        preceding_word = document_tokens[target - 3]
        if preceding_word in vocabulary:
            preceding_word_index = vocabulary[preceding_word]
            one_hot_encoding[preceding_word_index] = 1

    # Create one-hot encoding of the word following the target word
    if target < len(document_tokens) - 1:
        following_word = document_tokens[target + 3]
        if following_word in vocabulary:
            following_word_index = vocabulary[following_word]
            one_hot_encoding[following_word_index] = 1

    # Create multi-hot encoding of all words in the tokenized sentence
    for word in tokenized_sentence:
        if word in vocabulary:
            word_index = vocabulary[word]
            multi_hot_encoding[word_index] = 1

    # Combine the one-hot encodings
    features = one_hot_encoding + multi_hot_encoding

    return features

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# ------------Checkpoint 2.2------------------------------------------
def trainLogReg(train_corpus):
    # Extract input features and labels from train_corpus
    X_train = torch.tensor([instance[0] for instance in train_corpus], dtype=torch.float32)

    # Ensure labels are numeric
    labels = [instance[1] for instance in train_corpus]
    for label in labels:
        if not isinstance(label, (int, float)):
            raise ValueError("Labels must be numeric")

    y_train = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Add a dimension for batch

    # Initialize logistic regression model
    input_size = X_train.shape[1]
    model = LogisticRegression(input_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=10)

    # Train the model
    num_epochs =30
    train_losses = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_losses.append(loss.item())
    return model,train_losses
def plot_loss_curve(train_losses, dev_losses, epochs, target_word):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label=f'Training Loss ({target_word})')
    plt.plot(epochs, dev_losses, label=f'Dev Set Loss ({target_word})')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Dev Set Loss Curve')
    plt.legend()
    plt.show()

# Example usage:
# plot_loss_curve(all_train_losses[0], all_dev_losses[0], range(1, len(all_train_losses[0]) + 1), "receptor")

# ------------Checkpoint 2.3-------------------------------------------
def crossVal(model, dev_set):

    # Evaluate the model on the dev set
    X_dev = torch.tensor([instance[0] for instance in dev_set], dtype=torch.float32)
    y_dev = torch.tensor([instance[1] for instance in dev_set], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_dev)
        predictions = (outputs >= 0.5).float()

    # Calculate F1 score for each class
    f1_scores = f1_score(y_dev, predictions, average=None)

    # Calculate average F1 score across all classes
    average_f1 = np.mean(f1_scores)

    return average_f1

#-------------Checkpoint 2.4--------------------------------------------
def train_logistic_regression(train_corpus, dev_corpus, l2_penalty, dropout):
    # Extract input features and labels from train_corpus
    X_train = torch.tensor([instance[0] for instance in train_corpus], dtype=torch.float32)

    # Ensure labels are numeric
    labels = [instance[1] for instance in train_corpus]
    for label in labels:
        if not isinstance(label, (int, float)):
            raise ValueError("Labels must be numeric")

    y_train = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Add a dimension for batch

    # Initialize logistic regression model
    input_size = X_train.shape[1]
    model = nn.Sequential( nn.Dropout(dropout),LogisticRegression(input_size))

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=l2_penalty)

    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the dev set
    X_dev = torch.tensor([instance[0] for instance in dev_corpus], dtype=torch.float32)
    y_dev = torch.tensor([instance[1] for instance in dev_corpus], dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        dev_outputs = model(X_dev)
        dev_predictions = (dev_outputs >= 0.5).float()

    f1_scores = f1_score(y_dev, dev_predictions, average=None)

    return np.mean(f1_scores)

# -----------Checkpoint 2.5---------------------------------------------
class ImprovedLogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImprovedLogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% dropout rate

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Apply batch normalization after the first hidden layer
        x = F.relu(self.bn2(self.fc2(x)))  # Apply batch normalization after the second hidden layer
        x = self.dropout(x)  # Apply dropout
        x = torch.sigmoid(self.fc3(x))
        return x

def improvement(train_corpus, dev_corpus, l2_penalty, dropout):
    # Extract input features and labels from train_corpus
    X_train = torch.tensor([instance[0] for instance in train_corpus], dtype=torch.float32)

    # Ensure labels are numeric
    labels = [instance[1] for instance in train_corpus]
    for label in labels:
        if not isinstance(label, (int, float)):
            raise ValueError("Labels must be numeric")

    y_train = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Add a dimension for batch

    # Initialize improved logistic regression model
    input_size = X_train.shape[1]
    hidden_size = 4  # Define the size of the hidden layer
    model = ImprovedLogisticRegression(input_size, hidden_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=l2_penalty)

    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the dev set
    X_dev = torch.tensor([instance[0] for instance in dev_corpus], dtype=torch.float32)
    y_dev = torch.tensor([instance[1] for instance in dev_corpus], dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        dev_outputs = model(X_dev)
        dev_predictions = (dev_outputs >= 0.5).float()



    # Calculate F1 score for each class
    scores = f1_score(y_dev, dev_predictions, average=None)

    # Calculate average F1 score across all classes
    average_f1 = np.mean(scores)

    return average_f1


if __name__ == "__main__":
    # Taking input argument-----
    if len(sys.argv) != 2:
        print("Usage: python3 script_name.py input_file_name")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize a dictionary to count word frequencies
    word_freq_dict = {}

    # Tokenize each document and count words
    # vocabulary =[]
    bps_vocabulary = spacelessBPELearn(lines)
    for line in lines:
        tokens = wordTokenizer(line)
        # vocabulary = spacelessBPELearn(line)
        for token in tokens:
            word_freq_dict[token] = word_freq_dict.get(token, 0) + 1


    # Select the top 500 most frequent words
    most_common_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:500]

    # Create a vocabulary where each word is mapped to its index
    vocabulary = {word: idx for idx, (word, _) in enumerate(most_common_words)}
    # vocabulary=spacelessBPELearn()
    # Add an out-of-vocabulary index for marking any remaining words
    vocabulary['<UNK>'] = len(vocabulary)
    train_corpus = []
    dev_corpus = []
    bps_train_corpus=[]
    bps_dev_corpus=[]
    # Read each line and extract features
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            print("Invalid line format:", line)
            continue
        text_id, sent_id, label, sentence = parts
        tokens = wordTokenizer(sentence)
        bps_token=spacelessBPETokenize(sentence,bps_vocabulary)
        label = label.split('%')[0]  # Extract the label without sense keys

        # Search for variations of the label in the tokens
        label_found = False
        for token in tokens:
            if label.lower() in token.lower():
                target = tokens.index(token)
                label_found = True
                break
        if not label_found:
            print("Label not found in sentence:", label)
            continue
        if target == 0 or target == len(tokens) - 1:
            print("Target index out of range for sentence:", sentence)
            continue

        # -----------------Extract features---------------------------------------------
        features = extractLexicalFeatures(tokens, target, vocabulary,"word")
        bps_features=extractLexicalFeatures(tokens, target,vocabulary,"bps")
        # Assuming train_corpus contains the training data

        # --------------Word train and dev dataset-----------------------------------------------
        if len(train_corpus) < 3968:
            train_corpus.append([features,label])
        elif len(dev_corpus) < 1000:
            dev_corpus.append([features,label])

        if len(bps_train_corpus) < 3968:
            bps_train_corpus.append([bps_features,label])
        elif len(bps_dev_corpus) < 1000:
            bps_dev_corpus.append([bps_features,label])
        #----------------------------------------------------------------------------------------
    print("-----------------Checkpoint 2.1-----------------------------------")
    with open('_OUTPUT.txt', 'w', encoding='utf-8') as out:
        out.write("Checkpoint 2.1 - Feature vector for the first 2 and last documents of the training set.:\n")

        for i, instance in enumerate(train_corpus[:2] + train_corpus[-2:]):
            features, label = instance
            print(f"Document {i + 1}:")
            print(" Labels:-", label, "--Feature vector:", features)
            out.write(" Labels:- " + str(label) + " --Feature vector: " + str(features) + '\n')

    # --------------Checkpoint 2.2-----------------------------------------------------

    target_words = ['receptor', 'reduction', 'rate', 'reserve', 'reason', 'return']
    # Create a dictionary mapping each target word to its index (starting from 0)
    target_word_index = {word: index for index, word in enumerate(target_words)}

    models = []
    all_train_losses = []
    all_dev_losses=[]
    word_train_corpus = []
    word_dev_corpus = []
    bps_word_train_corpus = []
    bps_word_dev_corpus = []
    # ----Checkpoint 2.3--------------------------------
    f1_scores_word_tokenizer = []
    f1_scores_bps_tokenizer = []
    # ----Checkpoint 2.4---------------------------------
    best_f1_scores_word_tokenizer = []
    best_f1_scores_bps_tokenizer = []
    best_f1_word_hyperparameter=[]
    best_f1_bps_hyperparameter = []
    # -----Checkpoint 2.5--------------------------------
    improve_f1_scores_word_tokenizer = []
    improve_f1_scores_bps_tokenizer = []
    # ---------------------------------------------------
    for word in target_words:
        print(f'Training model for word: {word}')

        # Prepare training and dev data for the current target word
        for instance in train_corpus:
            if word in instance[1]:
                word_train_corpus.append([instance[0], 1])  # Positive class label
            else:
                word_train_corpus.append([instance[0], 0])  # Negative class label
        for instance in dev_corpus:
            if word in instance[1]:
                word_dev_corpus.append([instance[0], 1])  # Positive class label
            else:
                word_dev_corpus.append([instance[0], 0])  # Negative class label
        for instance in bps_train_corpus:
            if word in instance[1]:
                bps_word_train_corpus.append([instance[0], 1])  # Positive class label
            else:
                bps_word_train_corpus.append([instance[0], 0])  # Negative class label

        for instance in bps_dev_corpus:
            if word in instance[1]:
                bps_word_dev_corpus.append([instance[0], 1])  # Positive class label
            else:
                bps_word_dev_corpus.append([instance[0], 0])  # Negative class label
        # ------------Checkpoint 2.2-------------------------------------------------

        print("-----------------Checkpoint 2.2---------------------------------")
        model ,train_loss= trainLogReg(word_train_corpus)
        dev_model,dev_loss=trainLogReg(word_dev_corpus)
        models.append(model)

        all_train_losses.append(train_loss)
        all_dev_losses.append(dev_loss)
        print("Model trained for word:", word)

        print("Models------------",model,"Train Losses:------",train_loss)

        # -----Checkpoint 2.3---------------------------------------------------------
        # Evaluate the model with word tokenizer features
        f1_word_tokenizer = crossVal(model, word_dev_corpus)
        f1_scores_word_tokenizer.append(f1_word_tokenizer)
        # print(f'F1 score for word tokenizer model ({word}): {f1_word_tokenizer:.4f}')
        f1_bps_tokenizer = crossVal(model, bps_word_dev_corpus)
        f1_scores_bps_tokenizer.append(f1_bps_tokenizer)
        print("--------------End of Checkpoint 2.3------------------------")
        # -----Checkpoint 2.4-------------------------------------------------------------
        # Define L2 penalties and dropout percentages to try
        l2_penalties = [0.001, 0.01, 0.1, 1, 10, 100]
        dropout_percentages = [0, 0.1, 0.2, 0.5]

        # Initialize a dictionary to store F1 scores for different hyperparameter combinations
        f1_scores_table = {}
        bps_f1_scores_table={}
        # Iterate through all combinations of L2 penalties and dropout percentages
        for l2_penalty in l2_penalties:
            for dropout in dropout_percentages:
                # Train logistic regression model with the current hyperparameters
                avg_f1 = train_logistic_regression(word_train_corpus, word_dev_corpus, l2_penalty, dropout)
                # Store the F1 score in the table
                f1_scores_table[(l2_penalty, dropout)] = avg_f1

        for l2_penalty in l2_penalties:
            for dropout in dropout_percentages:
                # Train logistic regression model with the current hyperparameters
                bps_avg_f1 = train_logistic_regression(bps_word_train_corpus, bps_word_dev_corpus, l2_penalty, dropout)
                # Store the F1 score in the table
                bps_f1_scores_table[(l2_penalty, dropout)] = bps_avg_f1

        # Print the F1 scores table
        print("----------------Checkpoint 2.4------------------------------------")
        print("F1 Scores Table for Word Tokenization for ", word, ":---------------")
        print("------------------------------------------------------------------")
        print("L2 Penalty\t\tDropout\t\tAverage F1")

        with open('a.txt', 'a', encoding='utf-8') as out:
            out.write(f"Checkpoint 2.4 - F1 Scores Table for Word Tokenization for {word}:\n")
            for (l2_penalty, dropout), _f1_score in f1_scores_table.items():
                print(f"{l2_penalty}\t\t{dropout}\t\t{_f1_score}")
                out.write(f"{l2_penalty}\t\t{dropout}\t\t{_f1_score}\n")
        print("-----------------------------------------------------------")

        # Find the combination that yielded the best F1 score
        best_hyperparameters = max(f1_scores_table, key=f1_scores_table.get)
        best_f1_score = f1_scores_table[best_hyperparameters]

        #---------------- BPS Print the F1 scores table------------------------------------------
        print("F1 Scores Table for BPS Tokenization for ", word, ":---------------")
        print("-----------------------------------------------------------")
        print("L2 Penalty\t\tDropout\t\tAverage F1")
        for (l2_penalty, dropout), _f1_score in bps_f1_scores_table.items():
            print(f"{l2_penalty}\t\t{dropout}\t\t{_f1_score}")
        print("-----------------------------------------------------------")
        with open('OUTPUT2.txt', 'a', encoding='utf-8') as out:
            out.write(f"Checkpoint 2.4 - F1 Scores Table for BPS Tokenization for {word}:\n")
            for (l2_penalty, dropout), _f1_score in bps_f1_scores_table.items():
                print(f"{l2_penalty}\t\t{dropout}\t\t{_f1_score}")
                out.write(f"{l2_penalty}\t\t{dropout}\t\t{_f1_score}\n")
        # Find the combination that yielded the best F1 score
        bps_best_hyperparameters = max(bps_f1_scores_table, key=bps_f1_scores_table.get)
        bps_best_f1_score = bps_f1_scores_table[bps_best_hyperparameters]
        best_f1_scores_word_tokenizer.append(best_f1_score)
        best_f1_scores_bps_tokenizer.append(bps_best_f1_score)
        best_f1_word_hyperparameter.append(best_hyperparameters)
        best_f1_bps_hyperparameter.append(bps_best_hyperparameters)
        # -----------Checkpoint 2.5-------------------------------------------------------
        print("-------------------Checkpoint 2.5----------------------------------------")
        # Call the improvement function
        improved_f1 = improvement(word_train_corpus, word_dev_corpus, best_hyperparameters[0], best_hyperparameters[1])
        improve_f1_scores_word_tokenizer.append(improved_f1)
        improved_f2 = improvement(bps_word_train_corpus, bps_word_dev_corpus, best_hyperparameters[0], best_hyperparameters[1])
        improve_f1_scores_bps_tokenizer.append(improved_f2)

    print("--------------plot------------")
    # epochs = len(all_train_losses[0]) + 1
    epochs = range(1, len(all_train_losses[0]) + 1)
    plot_loss_curve(all_train_losses[0], all_dev_losses[0], epochs, "receptor")

    # plot_loss_curve(all_train_losses[0], all_dev_losses[0],epochs ,"receptor")
    print("--------------plot end------------")
    #----------------Checkpoint 2.3--------------------------------------------------
    print("----------------Checkpoint 2.3----------------------------")
    # Calculate the average F1 score across all words for both tokenizer types
    print("Word_Tokenization_F1_score:----",f1_scores_word_tokenizer)
    print("BPS_Tokenization_F1_score:-----",f1_scores_bps_tokenizer)
    with open('a1_p2_kumar_115317804_OUTPUT.txt', 'a', encoding='utf-8') as out:
        out.write("Checkpoint 2.3 - Word_Tokenization_F1_score:----")
        out.write(str(f1_scores_word_tokenizer) + ":\n")
        out.write("Checkpoint 2.3 - BPS_Tokenization_F1_score:----")
        out.write(str(f1_scores_bps_tokenizer) + ":\n")

    # ----------------Checkpoint 2.4------------------------------------------
    print("----------------Checkpoint 2.4-------------------------------------")
    print("Best_Word_Hyperparameter_F1_score:----", best_f1_word_hyperparameter)
    print("Best_BPS_Hyperparameter_F1_score:-----", best_f1_bps_hyperparameter)
    print("Best_Word_Tokenization_F1_score:----", best_f1_scores_word_tokenizer)
    print("Best_BPS_Tokenization_F1_score:-----", best_f1_scores_bps_tokenizer)
    with open('a1_p2_kumar_115317804_OUTPUT.txt', 'a', encoding='utf-8') as out:
        out.write("Checkpoint 2.4 - Best_Word_Hyperparameter_F1_score:----")
        out.write(str(best_f1_word_hyperparameter) + ":\n")
        out.write("Checkpoint 2.4 - Best_BPS_Hyperparameter_F1_score:-----")
        out.write(str(best_f1_bps_hyperparameter) + ":\n")
        out.write("Checkpoint 2.4 - Best_Word_Tokenization_F1_score:----")
        out.write(str(best_f1_scores_word_tokenizer) + ":\n")
        out.write("Checkpoint 2.4 - Best_BPS_Tokenization_F1_score:-----")
        out.write(str(best_f1_scores_bps_tokenizer) + ":\n")
    # # -----------Checkpoint 2.5---------------------------------------------
    print("-------------------Checkpoint 2.5---------------------------------")
    print("Improved_Word_Tokenization_F1_score:----", improve_f1_scores_word_tokenizer)
    print("Improved_BPS_Tokenization_F1_score:-----", improve_f1_scores_bps_tokenizer)
    with open('a1_p2_kumar_115317804_OUTPUT.txt', 'a', encoding='utf-8') as out:
        out.write("Checkpoint 2.5 - Improved_Word_Tokenization_F1_score:----")
        out.write(str(improve_f1_scores_word_tokenizer) + ":\n")
        out.write("Checkpoint 2.5 - Improved_BPS_Tokenization_F1_score:-----")
        out.write(str(improve_f1_scores_bps_tokenizer) + ":\n")


