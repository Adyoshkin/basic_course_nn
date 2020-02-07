TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


IMAGE_SIZE = 224
BATCH_SIZE = 4

NUM_CLASSES = 2
THRESHOLD = 0.55
NUM_EPOCHS = 100 

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)