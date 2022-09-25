import string
import torch


epochs = 50
batch_size = 16
SEED = 0

# vocabs = ' ' + string.digits + string.ascii_letters + string.punctuation + '°£€¥¢฿'
vocabs = ' ' + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&'()+,-.;=@[]^_`{}~°£€¥¢"

CHAR2LABEL = {char: i for i, char in enumerate(vocabs)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
